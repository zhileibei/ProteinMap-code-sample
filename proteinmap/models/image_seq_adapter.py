"""
ImageSeq baseline adapter — wraps the model trained by script 10 behind BasePredictor.

Architecture: SmallCNN(landmark_stains) + Linear(esm2) → concat → Linear(29)

Trained with script 10_train_image_seq_baseline.py.
Evaluated with script 11_run_benchmark_image_seq_baseline.py.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from proteinmap.models.base import BasePredictor
from proteinmap.data.constants import CLASSES


# ---------------------------------------------------------------------------
# Model architecture (shared with script 10)
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # → 64×64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # → 32×32
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → (128, 1, 1)
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ImageSeqClassifier(nn.Module):
    def __init__(self, esm2_dim: int = 1280, img_dim: int = 256, n_classes: int = 29):
        super().__init__()
        self.img_enc  = SmallCNN(img_dim)
        self.seq_proj = nn.Linear(esm2_dim, img_dim)
        self.head     = nn.Linear(img_dim * 2, n_classes)

    def forward(self, img, esm2):
        img_feat = self.img_enc(img)                  # (B, img_dim)
        seq_feat = F.relu(self.seq_proj(esm2))        # (B, img_dim)
        return self.head(torch.cat([img_feat, seq_feat], dim=-1))  # (B, n_classes)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class ImageSeqPredictor(BasePredictor):
    """
    Multimodal baseline: SmallCNN(landmark_stains) + Linear(mean-pooled ESM2) → 29 classes.

    predict_compartment uses both image and sequence — unlike ESM2BaselinePredictor
    which ignores the image.

    If pooled_cache_path is supplied, run_on_dataloader looks up pre-pooled ESM2
    vectors by ensembl_id instead of deserializing the large per-protein pkl files,
    which is the main I/O bottleneck when evaluating 200K+ cell crops.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        threshold: float = None,
        pooled_cache_path: str = None,
    ):
        import pickle
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)

        self._pooled_cache: dict | None = None
        if pooled_cache_path:
            print(f"  Loading pooled ESM2 cache from {pooled_cache_path} ...")
            with open(pooled_cache_path, "rb") as f:
                self._pooled_cache = pickle.load(f)

        img_dim = ckpt.get("img_dim", 256)
        self.threshold = threshold if threshold is not None else ckpt.get("threshold", 0.5)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImageSeqClassifier(esm2_dim=1280, img_dim=img_dim, n_classes=len(CLASSES[0]))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    def predict_compartment(
        self,
        esm2_embedding: np.ndarray,   # (1, 2000, 1280)
        protein_length: int,
        landmark_stains: np.ndarray,  # (3, 128, 128)
    ) -> dict:
        pooled = esm2_embedding[0, :protein_length, :].mean(axis=0)  # (1280,)
        img_t  = torch.from_numpy(landmark_stains).unsqueeze(0).to(self.device)  # (1, 3, 128, 128)
        esm_t  = torch.from_numpy(pooled).unsqueeze(0).to(self.device)           # (1, 1280)

        with torch.no_grad():
            logits = self.model(img_t, esm_t)                      # (1, 29)
            proba  = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (29,)

        binary = (proba >= self.threshold).astype(np.int32)
        if binary.sum() == 0:
            binary[np.argmax(proba)] = 1
        return {"proba": proba.astype(np.float32), "binary": binary}

    def predict_inpainting(self, esm2_embedding, protein_length, landmark_stains):
        return None

    @property
    def supports_inpainting(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "ImageSeqBaseline"

    # ------------------------------------------------------------------
    # Batch inference (called by Evaluator)
    # ------------------------------------------------------------------

    def run_on_dataloader(self, dataloader, limit_batches: int = None) -> list[dict]:
        rows = []
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{self.name} inference")):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            if batch is None:
                continue

            if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[1], list):
                datums, metas = batch
            else:
                datums = batch
                metas = [{}] * datums[0].shape[0]

            X_esm2, X_len, X_stains, y_multi, _y_ab = datums
            B = X_esm2.shape[0]

            # Mean-pool ESM2: use pre-pooled cache if available (avoids pkl I/O)
            if self._pooled_cache is not None:
                pooled = []
                for i in range(B):
                    eid = (metas[i].get("ensembl_ids") or "").split(",")[0].strip()
                    vec = self._pooled_cache.get(eid)
                    if vec is None:
                        l = int(X_len[i].item())
                        vec = X_esm2[i, 0, :l, :].mean(0).numpy()
                    pooled.append(torch.from_numpy(vec))
                pooled_t = torch.stack(pooled).to(self.device)  # (B, 1280)
            else:
                pooled = []
                for i in range(B):
                    l = int(X_len[i].item())
                    pooled.append(X_esm2[i, 0, :l, :].mean(0))  # (1280,)
                pooled_t = torch.stack(pooled).to(self.device)   # (B, 1280)
            img_t    = X_stains.float().to(self.device)       # (B, 3, 128, 128)

            with torch.no_grad():
                logits = self.model(img_t, pooled_t)              # (B, 29)
                probas = torch.sigmoid(logits).cpu().numpy()       # (B, 29)

            for i in range(B):
                proba  = probas[i].astype(np.float32)
                binary = (proba >= self.threshold).astype(np.int32)
                if binary.sum() == 0:
                    binary[np.argmax(proba)] = 1
                meta = metas[i] if i < len(metas) else {}
                rows.append({
                    "model": self.name,
                    "antibody":    meta.get("antibody"),
                    "ensembl_ids": meta.get("ensembl_ids"),
                    "if_plate_id": meta.get("if_plate_id"),
                    "position":    meta.get("position"),
                    "sample":      meta.get("sample"),
                    "cell_id":     meta.get("cell_id"),
                    "true_locations": meta.get("locations"),
                    "atlas_name":  meta.get("atlas_name"),
                    "inpainting_mse": np.nan,
                    "y_true_multilabel": y_multi[i].cpu().numpy().astype(np.int32),
                    "y_pred_proba":  proba,
                    "y_pred_binary": binary,
                    "y_antibody_pred": None,
                    "y_antibody_true": None,
                })
        return rows


# ---------------------------------------------------------------------------
# Inpainting extension
# ---------------------------------------------------------------------------

class SimpleDecoder(nn.Module):
    """
    Pure bottleneck → image decoder, no skip connections.

    fused (B, in_dim) → Linear → reshape(B, 32, 4, 4)
    → 5× ConvTranspose2d blocks → (B, 1, 128, 128)

    Output is unbounded (no activation) — MSE-friendly, same convention as PUPS.
    """

    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, 32 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # → (16, 8, 8)
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16,  8, kernel_size=4, stride=2, padding=1),  # → (8, 16, 16)
            nn.BatchNorm2d(8),  nn.ReLU(),
            nn.ConvTranspose2d( 8,  4, kernel_size=4, stride=2, padding=1),  # → (4, 32, 32)
            nn.BatchNorm2d(4),  nn.ReLU(),
            nn.ConvTranspose2d( 4,  2, kernel_size=4, stride=2, padding=1),  # → (2, 64, 64)
            nn.BatchNorm2d(2),  nn.ReLU(),
            nn.ConvTranspose2d( 2,  1, kernel_size=4, stride=2, padding=1),  # → (1, 128, 128)
        )

    def forward(self, fused):
        x = self.proj(fused)           # (B, 512)
        x = x.view(x.size(0), 32, 4, 4)
        return self.decoder(x)         # (B, 1, 128, 128)


class ImageSeqInpaintingModel(nn.Module):
    """
    Joint classification + inpainting model.

    Shares the SmallCNN image encoder and ESM2 projection with ImageSeqClassifier,
    and adds a SimpleDecoder head for antibody stain reconstruction.

    forward(img, esm2) → (cls_logits (B, n_classes), inp_pred (B, 1, 128, 128))
    """

    def __init__(self, esm2_dim: int = 1280, img_dim: int = 256, n_classes: int = 29):
        super().__init__()
        fused_dim = img_dim * 2
        self.img_enc  = SmallCNN(img_dim)
        self.seq_proj = nn.Linear(esm2_dim, img_dim)
        self.cls_head = nn.Linear(fused_dim, n_classes)
        self.decoder  = SimpleDecoder(fused_dim)

    def forward(self, img, esm2):
        img_feat = self.img_enc(img)                          # (B, img_dim)
        seq_feat = F.relu(self.seq_proj(esm2))                # (B, img_dim)
        fused    = torch.cat([img_feat, seq_feat], dim=-1)    # (B, fused_dim)
        cls_logits = self.cls_head(fused)                     # (B, n_classes)
        inp_pred   = self.decoder(fused)                      # (B, 1, 128, 128)
        return cls_logits, inp_pred


class ImageSeqInpaintingPredictor(ImageSeqPredictor):
    """
    Wraps ImageSeqInpaintingModel behind BasePredictor.

    Supports both compartment classification and antibody stain inpainting.
    """

    def __init__(self, checkpoint_path: str, device: str = None, threshold: float = None,
                 pooled_cache_path: str = None):
        import pickle
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)

        img_dim = ckpt.get("img_dim", 256)
        self.threshold = threshold if threshold is not None else ckpt.get("threshold", 0.5)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ImageSeqInpaintingModel(
            esm2_dim=1280, img_dim=img_dim, n_classes=len(CLASSES[0])
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self._pooled_cache: dict | None = None
        if pooled_cache_path:
            print(f"  Loading pooled ESM2 cache from {pooled_cache_path} ...")
            with open(pooled_cache_path, "rb") as f:
                self._pooled_cache = pickle.load(f)

    @property
    def supports_inpainting(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ImageSeqInpainting"

    def predict_compartment(self, esm2_embedding, protein_length, landmark_stains):
        pooled = esm2_embedding[0, :protein_length, :].mean(axis=0)
        img_t  = torch.from_numpy(landmark_stains).unsqueeze(0).to(self.device)
        esm_t  = torch.from_numpy(pooled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cls_logits, _ = self.model(img_t, esm_t)
            proba = torch.sigmoid(cls_logits).squeeze(0).cpu().numpy().astype(np.float32)
        binary = (proba >= self.threshold).astype(np.int32)
        if binary.sum() == 0:
            binary[np.argmax(proba)] = 1
        return {"proba": proba, "binary": binary}

    def predict_inpainting(self, esm2_embedding, protein_length, landmark_stains):
        pooled = esm2_embedding[0, :protein_length, :].mean(axis=0)
        img_t  = torch.from_numpy(landmark_stains).unsqueeze(0).to(self.device)
        esm_t  = torch.from_numpy(pooled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, inp_pred = self.model(img_t, esm_t)
        return inp_pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)  # (128, 128)

    def run_on_dataloader(self, dataloader, limit_batches: int = None) -> list[dict]:
        rows = []
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{self.name} inference")):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            if batch is None:
                continue

            if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[1], list):
                datums, metas = batch
            else:
                datums = batch
                metas = [{}] * datums[0].shape[0]

            X_esm2, X_len, X_stains, y_multi, y_ab_true = datums
            B = X_esm2.shape[0]

            if self._pooled_cache is not None:
                pooled = []
                for i in range(B):
                    eid = (metas[i].get("ensembl_ids") or "").split(",")[0].strip()
                    vec = self._pooled_cache.get(eid)
                    if vec is None:
                        l = int(X_len[i].item())
                        vec = X_esm2[i, 0, :l, :].mean(0).numpy()
                    pooled.append(torch.from_numpy(vec))
                pooled_t = torch.stack(pooled).to(self.device)
            else:
                pooled = []
                for i in range(B):
                    l = int(X_len[i].item())
                    pooled.append(X_esm2[i, 0, :l, :].mean(0))
                pooled_t = torch.stack(pooled).to(self.device)
            img_t = X_stains.float().to(self.device)

            with torch.no_grad():
                cls_logits, inp_pred = self.model(img_t, pooled_t)
                probas   = torch.sigmoid(cls_logits).cpu().numpy()
                inp_pred = inp_pred.squeeze(1).cpu().numpy()  # (B, 128, 128)

            y_ab_true_np = y_ab_true.cpu().numpy() if y_ab_true is not None else None

            for i in range(B):
                proba  = probas[i].astype(np.float32)
                binary = (proba >= self.threshold).astype(np.int32)
                if binary.sum() == 0:
                    binary[np.argmax(proba)] = 1
                ab_true = y_ab_true_np[i].astype(np.float32) if y_ab_true_np is not None else None
                ab_pred = inp_pred[i].astype(np.float32)
                mse = float(np.mean((ab_true - ab_pred) ** 2)) if ab_true is not None else np.nan
                meta = metas[i] if i < len(metas) else {}
                rows.append({
                    "model": self.name,
                    "antibody":    meta.get("antibody"),
                    "ensembl_ids": meta.get("ensembl_ids"),
                    "if_plate_id": meta.get("if_plate_id"),
                    "position":    meta.get("position"),
                    "sample":      meta.get("sample"),
                    "cell_id":     meta.get("cell_id"),
                    "true_locations": meta.get("locations"),
                    "atlas_name":  meta.get("atlas_name"),
                    "inpainting_mse": mse,
                    "y_true_multilabel": y_multi[i].cpu().numpy().astype(np.int32),
                    "y_pred_proba":  proba,
                    "y_pred_binary": binary,
                    "y_antibody_pred": ab_pred,
                    "y_antibody_true": ab_true,
                })
        return rows
