"""
Unified training script — replaces scripts 07 (ESM2+LR), 10 (ImageSeq), 13 (ImageSeq+Inpainting).

Usage:
    python scripts/train.py \\
        --config configs/models/image_seq_classifier.yaml \\
        --output_path results/image_seq_v2/checkpoint.pkl \\
        --data_root data/czi \\
        --pooled_cache data/esm2_cache/pooled_esm2.joblib \\
        --train_pairs_csv .../split_v2/train_pairs.csv \\
        --val_pairs_csv   .../split_v2/val_pairs.csv \\
        [--rna_tsv .../rna_celline.tsv]          # seq_rna_baseline
        [--subcell_vit_ckpt .../vit.pth]         # subcell only
        [--input_size 128]                        # image resize (128 default; 448 for SubCell)
        [--epochs 10] [--batch_size 128] [--lr 1e-3] \\
        [--cls_weight 1.0] [--inp_weight 1.0] \\
        [--train_frac 1.0] [--val_every 2000] [--patience 5] \\
        [--num_workers 4] [--wandb_run_name my_run]
"""

import argparse
import dataclasses
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from proteinmap.models.components.localization_model import LocalizationModel, ModelConfig
from proteinmap.data.sequence_loader import SequenceLoader
from proteinmap.data.hpa_train_dataset import HPATrainDataset
from proteinmap.data.constants import DatasetType


def parse_args():
    p = argparse.ArgumentParser("ProteinMap unified training")
    p.add_argument("--config",            required=True,  help="Path to model YAML config")
    p.add_argument("--output_path",       required=True,  help="Path to save checkpoint (.pkl)")
    p.add_argument("--data_root",         required=True,  help="HPA data root directory")
    p.add_argument("--pooled_cache",      default=None,   help="Path to pooled_esm2.pkl or .joblib")
    p.add_argument("--tsv_path",          default=None,   help="SwissProt TSV (for one_hot mode)")
    # split_v2 pairs CSV args (replaces legacy antibody-list mode)
    p.add_argument("--train_pairs_csv",   default=None,
                   help="Path to split_v2/train_pairs.csv")
    p.add_argument("--val_pairs_csv",     default=None,
                   help="Path to split_v2/val_pairs.csv")
    # RNA expression (seq_rna_baseline)
    p.add_argument("--rna_tsv",           default=None,
                   help="Path to rna_celline.tsv (required for seq_rna_baseline)")
    # SubCell ViT checkpoint (subcell only)
    p.add_argument("--subcell_vit_ckpt",  default=None,
                   help="Path to SubCell ViT .pth checkpoint")
    # Image resize resolution
    p.add_argument("--input_size",        type=int, default=128,
                   help="Image resize resolution passed to dataset (default: 128)")
    p.add_argument("--epochs",            type=int,   default=10)
    p.add_argument("--batch_size",        type=int,   default=128)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--cls_weight",        type=float, default=1.0)
    p.add_argument("--inp_weight",        type=float, default=1.0)
    p.add_argument("--train_frac",        type=float, default=1.0,
                   help="Fraction of training data to use (quick smoke test)")
    p.add_argument("--val_every",         type=int,   default=2000,
                   help="Run validation every N gradient steps")
    p.add_argument("--patience",          type=int,   default=5,
                   help="Stop if no improvement for this many val checks")
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--threshold",         type=float, default=0.5)
    p.add_argument("--wandb_run_name",    default=None)
    p.add_argument("--device",            default=None)
    return p.parse_args()


def _make_collate(return_antibody: bool, has_rna: bool = False):
    """Returns a collate_fn that filters None items and stacks tensors."""
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        if return_antibody and has_rna:
            landmarks, seq_inputs, labels, antibodies, rna_vecs = zip(*batch)
            return (
                torch.from_numpy(np.stack(landmarks)).float(),
                torch.from_numpy(np.stack(seq_inputs)).float(),
                torch.from_numpy(np.stack(labels)).float(),
                torch.from_numpy(np.stack(antibodies)).float().unsqueeze(1),
                torch.from_numpy(np.stack(rna_vecs)).float(),
            )
        elif return_antibody:
            landmarks, seq_inputs, labels, antibodies = zip(*batch)
            return (
                torch.from_numpy(np.stack(landmarks)).float(),
                torch.from_numpy(np.stack(seq_inputs)).float(),
                torch.from_numpy(np.stack(labels)).float(),
                torch.from_numpy(np.stack(antibodies)).float().unsqueeze(1),
            )
        elif has_rna:
            landmarks, seq_inputs, labels, rna_vecs = zip(*batch)
            return (
                torch.from_numpy(np.stack(landmarks)).float(),
                torch.from_numpy(np.stack(seq_inputs)).float(),
                torch.from_numpy(np.stack(labels)).float(),
                torch.from_numpy(np.stack(rna_vecs)).float(),
            )
        else:
            landmarks, seq_inputs, labels = zip(*batch)
            return (
                torch.from_numpy(np.stack(landmarks)).float(),
                torch.from_numpy(np.stack(seq_inputs)).float(),
                torch.from_numpy(np.stack(labels)).float(),
            )
    return collate_fn


def _validate(model, val_loader, cfg, device, return_antibody, has_rna=False):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            if return_antibody and has_rna:
                landmarks, seq_inputs, labels, _, rna_vecs = batch
                rna_vecs = rna_vecs.to(device)
            elif return_antibody:
                landmarks, seq_inputs, labels, _ = batch
                rna_vecs = None
            elif has_rna:
                landmarks, seq_inputs, labels, rna_vecs = batch
                rna_vecs = rna_vecs.to(device)
            else:
                landmarks, seq_inputs, labels = batch
                rna_vecs = None

            seq_in = seq_inputs.to(device) if cfg.seq_encoder.type != "none" else None
            vis_in = landmarks.to(device)  if cfg.vis_encoder.type != "none" else None
            out = model(seq_input=seq_in, vis_input=vis_in, rna_input=rna_vecs)
            all_logits.append(out["cls_logits"].cpu())
            all_labels.append(labels)
    if not all_logits:
        return 0.0
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probas = torch.sigmoid(logits).numpy()
    y_true = labels.numpy().astype(int)
    try:
        from sklearn.metrics import roc_auc_score
        valid = y_true.sum(axis=0) > 0
        return float(roc_auc_score(y_true[:, valid], probas[:, valid], average="macro"))
    except Exception:
        return 0.0


def _save_checkpoint(model, cfg, threshold, val_auroc, output_path):
    ckpt = {
        "model_config":    dataclasses.asdict(cfg),
        "model_state":     {k: v.cpu() for k, v in model.state_dict().items()},
        "threshold":       threshold,
        "val_macro_auroc": val_auroc,
    }
    with open(output_path, "wb") as f:
        pickle.dump(ckpt, f)


def main():
    args = parse_args()
    cfg = ModelConfig.from_yaml(args.config)
    print(f"[train.py] Config loaded: {cfg.name}")
    print(f"           seq_encoder={cfg.seq_encoder.type}  vis_encoder={cfg.vis_encoder.type}")
    has_rna_enc = hasattr(cfg, "rna_encoder") and cfg.rna_encoder is not None and cfg.rna_encoder.type != "none"
    print(f"           rna_encoder={cfg.rna_encoder.type if has_rna_enc else 'none'}  inpainting={cfg.heads.inpainting}")

    # Inject SubCell ViT checkpoint path into config if needed
    if cfg.vis_encoder.type == "subcell_vit":
        if args.subcell_vit_ckpt is None:
            raise ValueError("--subcell_vit_ckpt is required for subcell_vit vis_encoder")
        cfg.vis_encoder.subcell_vit_ckpt = args.subcell_vit_ckpt

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional wandb
    wandb_run = None
    if args.wandb_run_name:
        try:
            import wandb
            wandb_run = wandb.init(
                project="ProteinMap",
                name=args.wandb_run_name,
                config=dataclasses.asdict(cfg),
            )
        except ImportError:
            print("[train.py] wandb not installed — skipping.")

    # Sequence loader (skip if seq_encoder is none)
    seq_loader = None
    if cfg.seq_encoder.type == "esm2_pooled":
        if args.pooled_cache is None:
            raise ValueError("--pooled_cache is required for esm2_pooled mode")
        seq_loader = SequenceLoader("esm2_pooled", pooled_cache_path=args.pooled_cache)
    elif cfg.seq_encoder.type == "one_hot":
        if args.tsv_path is None:
            raise ValueError("--tsv_path is required for one_hot mode")
        seq_loader = SequenceLoader("one_hot", tsv_path=args.tsv_path)
    elif cfg.seq_encoder.type == "none":
        # No sequence encoder — create a dummy loader returning zeros
        seq_loader = None
    else:
        raise ValueError(f"Unsupported seq_encoder.type: {cfg.seq_encoder.type!r}")

    # RNA loader (seq_rna_baseline)
    rna_loader = None
    if has_rna_enc:
        if args.rna_tsv is None:
            raise ValueError("--rna_tsv is required when rna_encoder is configured")
        from proteinmap.data.rna_loader import RNAExpressionLoader
        rna_loader = RNAExpressionLoader(args.rna_tsv)
        # Inject n_genes into config before building the model
        cfg.rna_encoder.n_genes = rna_loader.n_genes
        print(f"[train.py] RNA loader: {rna_loader.n_genes} genes")

    # Datasets
    return_antibody = cfg.heads.inpainting

    # Skip image loading for sequence-only models (saves huge disk IO)
    skip_image = (cfg.vis_encoder.type == "none")
    if skip_image:
        print(f"[train.py] vis_encoder=none: skipping image loading (faster IO)")

    # Use pairs_csv (split_v2) if provided; else fall back to legacy antibody-list mode
    if args.train_pairs_csv is not None:
        train_ds = HPATrainDataset(
            split=None, data_root=args.data_root, seq_loader=seq_loader,
            return_antibody=return_antibody, pairs_csv=args.train_pairs_csv,
            rna_loader=rna_loader, input_size=args.input_size, skip_image=skip_image,
        )
    else:
        train_ds = HPATrainDataset(
            DatasetType.TRAIN, args.data_root, seq_loader, return_antibody,
            rna_loader=rna_loader, input_size=args.input_size, skip_image=skip_image,
        )

    if args.val_pairs_csv is not None:
        val_ds = HPATrainDataset(
            split=None, data_root=args.data_root, seq_loader=seq_loader,
            return_antibody=return_antibody, pairs_csv=args.val_pairs_csv,
            rna_loader=rna_loader, input_size=args.input_size, skip_image=skip_image,
        )
    else:
        val_ds = HPATrainDataset(
            DatasetType.EVAL, args.data_root, seq_loader, return_antibody,
            rna_loader=rna_loader, input_size=args.input_size, skip_image=skip_image,
        )

    print(f"[train.py] Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    if args.train_frac < 1.0:
        n = max(1, int(len(train_ds) * args.train_frac))
        indices = np.random.permutation(len(train_ds))[:n].tolist()
        train_ds = Subset(train_ds, indices)
        print(f"           Using {n} training samples ({args.train_frac:.0%} of full set)")

    collate_fn = _make_collate(return_antibody, has_rna=has_rna_enc)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # Model + optimiser + schedule
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalizationModel.from_config(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    best_val_auroc = -1.0
    patience_count = 0
    global_step = 0
    stopped_early = False

    print(f"[train.py] Training for {args.epochs} epochs on {device}")

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            if batch is None:
                continue

            if return_antibody and has_rna_enc:
                landmarks, seq_inputs, labels, antibodies, rna_vecs = batch
                antibodies = antibodies.to(device)
                rna_vecs   = rna_vecs.to(device)
            elif return_antibody:
                landmarks, seq_inputs, labels, antibodies = batch
                antibodies = antibodies.to(device)
                rna_vecs   = None
            elif has_rna_enc:
                landmarks, seq_inputs, labels, rna_vecs = batch
                rna_vecs = rna_vecs.to(device)
                antibodies = None
            else:
                landmarks, seq_inputs, labels = batch
                rna_vecs   = None
                antibodies = None

            seq_in = seq_inputs.to(device) if cfg.seq_encoder.type != "none" else None
            vis_in = landmarks.to(device)  if cfg.vis_encoder.type != "none" else None
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(seq_input=seq_in, vis_input=vis_in, rna_input=rna_vecs)
            loss = args.cls_weight * bce_loss(out["cls_logits"], labels)
            if return_antibody and out["inp_pred"] is not None and antibodies is not None:
                loss = loss + args.inp_weight * mse_loss(out["inp_pred"], antibodies)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if wandb_run:
                wandb_run.log({"train/loss": loss.item(), "step": global_step})

            # Step-level validation
            if global_step % args.val_every == 0:
                val_auroc = _validate(model, val_loader, cfg, device, return_antibody, has_rna=has_rna_enc)
                model.train()
                print(f"  step={global_step:>7d}  val_macro_auroc={val_auroc:.4f}")
                if wandb_run:
                    wandb_run.log({"val/macro_auroc": val_auroc, "step": global_step})

                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    patience_count = 0
                    _save_checkpoint(model, cfg, args.threshold, best_val_auroc, output_path)
                    print(f"  [*] New best  val_auroc={best_val_auroc:.4f}  saved → {output_path}")
                else:
                    patience_count += 1
                    if patience_count >= args.patience:
                        print(f"  [!] Early stopping at step {global_step}")
                        stopped_early = True
                        break

        if stopped_early:
            break

        # Epoch-level validation
        val_auroc = _validate(model, val_loader, cfg, device, return_antibody, has_rna=has_rna_enc)
        model.train()
        print(f"Epoch {epoch+1}/{args.epochs}  val_macro_auroc={val_auroc:.4f}")
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_count = 0
            _save_checkpoint(model, cfg, args.threshold, best_val_auroc, output_path)

    if wandb_run:
        wandb_run.finish()
    print(f"\n[train.py] Done.  best val_macro_auroc={best_val_auroc:.4f}")
    print(f"           Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
