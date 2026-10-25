"""
Sanity check 2 — verify LocalizationModel forward pass shapes for all 4 configs.

Checks:
1. Load all 4 config YAMLs and instantiate LocalizationModel.from_config()
2. Create random batch (B=4) with correct tensor shapes for each config
3. Assert forward pass runs without error; assert output shapes
4. Assert supports_inpainting is True only for inpainting config
5. Round-trip: save checkpoint → LocalizationPredictor.from_checkpoint() → weights identical
6. Assert loss.backward() completes without NaN gradients

Usage:
    python scripts/sanity_model_forward.py
"""

import pathlib
import pickle
import sys
import tempfile
import dataclasses

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import torch

from proteinmap.models.components.localization_model import LocalizationModel, ModelConfig
from proteinmap.models.localization_predictor import LocalizationPredictor
from proteinmap.data.sequence_loader import MAX_SEQ_LEN, ONEHOT_DIM


CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs" / "models"
B = 4   # batch size for checks


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--configs_dir", default=str(CONFIGS_DIR))
    return p.parse_args()


def make_batch(cfg: ModelConfig, B: int, device: str):
    """Build a random batch matching the config's expected input shapes."""
    seq_type = cfg.seq_encoder.type
    vis_type = cfg.vis_encoder.type

    if seq_type == "esm2_pooled":
        seq_in = torch.randn(B, cfg.seq_encoder.esm2_dim)
    elif seq_type == "one_hot":
        seq_in = torch.zeros(B, MAX_SEQ_LEN, ONEHOT_DIM)
        # Fill first 100 positions with valid one-hot
        idx = torch.randint(0, ONEHOT_DIM, (B, 100))
        for b in range(B):
            seq_in[b, :100, :].scatter_(1, idx[b].unsqueeze(1), 1.0)
    else:
        seq_in = None

    vis_in = torch.randn(B, 3, 128, 128) if vis_type != "none" else None

    seq_in = seq_in.to(device) if seq_in is not None else None
    vis_in = vis_in.to(device) if vis_in is not None else None
    return seq_in, vis_in


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs_dir = pathlib.Path(args.configs_dir)
    passed = 0
    failed = 0

    def check(condition: bool, msg: str):
        nonlocal passed, failed
        if condition:
            print(f"  [OK]  {msg}")
            passed += 1
        else:
            print(f"  [FAIL] {msg}")
            failed += 1

    yaml_files = sorted(configs_dir.glob("*.yaml"))
    print("=" * 60)
    print(f"Sanity 2: LocalizationModel forward pass ({len(yaml_files)} configs)")
    print("=" * 60)

    check(len(yaml_files) >= 4, f"Found ≥ 4 YAML configs in {configs_dir} ({len(yaml_files)} found)")

    for yaml_path in yaml_files:
        print(f"\n--- Config: {yaml_path.name} ---")
        try:
            cfg = ModelConfig.from_yaml(str(yaml_path))
        except Exception as e:
            check(False, f"ModelConfig.from_yaml({yaml_path.name}): {e}")
            continue

        check(True, f"ModelConfig.from_yaml({yaml_path.name}) succeeded")

        # ------------------------------------------------------------------
        # Instantiate model
        # ------------------------------------------------------------------
        try:
            model = LocalizationModel.from_config(cfg).to(device)
            check(True, "LocalizationModel.from_config() succeeded")
        except Exception as e:
            check(False, f"LocalizationModel.from_config() raised: {e}")
            continue

        # ------------------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------------------
        seq_in, vis_in = make_batch(cfg, B, device)
        try:
            model.eval()
            with torch.no_grad():
                out = model(seq_input=seq_in, vis_input=vis_in)
            check(True, "model.forward() completed without error")
        except Exception as e:
            check(False, f"model.forward() raised: {e}")
            continue

        # Output shapes
        expected_cls = (B, cfg.heads.n_classes)
        cls_shape = tuple(out["cls_logits"].shape)
        check(cls_shape == expected_cls,
              f"cls_logits shape {cls_shape} == {expected_cls}")

        if cfg.heads.inpainting:
            check(out["inp_pred"] is not None,
                  "inp_pred is not None for inpainting config")
            if out["inp_pred"] is not None:
                expected_inp = (B, 1, 128, 128)
                inp_shape = tuple(out["inp_pred"].shape)
                check(inp_shape == expected_inp,
                      f"inp_pred shape {inp_shape} == {expected_inp}")
        else:
            check(out["inp_pred"] is None,
                  "inp_pred is None for non-inpainting config")

        # supports_inpainting property
        check(
            model.supports_inpainting == cfg.heads.inpainting,
            f"supports_inpainting={model.supports_inpainting} matches config ({cfg.heads.inpainting})"
        )

        # ------------------------------------------------------------------
        # Backward pass (gradient check)
        # ------------------------------------------------------------------
        try:
            model.train()
            seq_in2, vis_in2 = make_batch(cfg, B, device)
            if seq_in2 is not None:
                seq_in2.requires_grad_(False)
            out2 = model(seq_input=seq_in2, vis_input=vis_in2)
            labels = torch.zeros(B, cfg.heads.n_classes, device=device)
            import torch.nn as nn
            loss = nn.BCEWithLogitsLoss()(out2["cls_logits"], labels)
            if out2["inp_pred"] is not None:
                tgt = torch.zeros_like(out2["inp_pred"])
                loss = loss + nn.MSELoss()(out2["inp_pred"], tgt)
            loss.backward()

            has_nan = any(
                p.grad is not None and p.grad.isnan().any()
                for p in model.parameters()
            )
            check(not has_nan, "No NaN gradients after loss.backward()")
        except Exception as e:
            check(False, f"loss.backward() raised: {e}")

        # ------------------------------------------------------------------
        # Checkpoint round-trip
        # ------------------------------------------------------------------
        try:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp_path = tmp.name
            ckpt = {
                "model_config":    dataclasses.asdict(cfg),
                "model_state":     {k: v.cpu() for k, v in model.state_dict().items()},
                "threshold":       0.5,
                "val_macro_auroc": 0.0,
            }
            with open(tmp_path, "wb") as f:
                pickle.dump(ckpt, f)

            predictor = LocalizationPredictor.from_checkpoint(tmp_path, device=device)

            # Compare state dicts
            orig_sd = model.state_dict()
            loaded_sd = predictor.model.state_dict()
            all_close = all(
                torch.allclose(orig_sd[k].cpu(), loaded_sd[k].cpu())
                for k in orig_sd
            )
            check(all_close, "Round-trip checkpoint → weights identical")
            pathlib.Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            check(False, f"Checkpoint round-trip raised: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
