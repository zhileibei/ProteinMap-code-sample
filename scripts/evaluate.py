"""
Unified evaluation script — replaces scripts 03, 08, 11, 12, 14.

Usage:
    python scripts/evaluate.py \\
        --checkpoint results/image_seq_v2/checkpoint.pkl \\
        --output_dir results/image_seq_v2/ \\
        --split hpa_proteins \\
        --data_root data/czi_224 \\
        [--pooled_cache .../pooled_esm2.pkl] \\
        [--per_protein] [--bootstrap 1000] \\
        [--batch_size 64] [--num_workers 8] [--limit_batches N]

    # split_v2 splits (val_v2 / protein_test_v2 / cell_line_test_v2):
        --split val_v2 --split_v2_dir .../split_v2 --data_root .../czi

    # PUPS / ProtGPS predictors:
        --predictor_type pups    --pups_ckpt   results/pups_v2/checkpoint.ckpt
        --predictor_type protgps --protgps_ckpt .../protgps/checkpoints/protgps_hpa/best.ckpt

    # Old-format checkpoints (no embedded model_config):
        --legacy_model image_seq_classifier   (or image_seq_inpainting)

Supported splits:
    hpa_proteins        — SubCellDatasetCZI TEST split (old benchmark)
    hpa_cell_lines      — SubCellDatasetCZICellLine TEST split
    misloc              — MislocDataset WT+MT
    val_v2              — split_v2 validation set
    protein_test_v2     — split_v2 protein-holdout test
    cell_line_test_v2   — split_v2 cell-line-holdout test
"""

import argparse
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from proteinmap.data.constants import DatasetType
from proteinmap.data.collate import _CZIWithMeta, _CZIWithMetaV2, _collate_with_meta
from proteinmap.models.localization_predictor import LocalizationPredictor
from proteinmap.eval.unified_evaluator import UnifiedEvaluator

_V2_SPLIT_FILES = {
    "val_v2":            "val_pairs.csv",
    "protein_test_v2":   "protein_test_pairs.csv",
    "cell_line_test_v2": "cell_line_test_pairs.csv",
}


def parse_args():
    p = argparse.ArgumentParser("ProteinMap unified evaluation")
    p.add_argument("--checkpoint",      default=None,   help="Path to checkpoint .pkl")
    p.add_argument("--output_dir",      required=True,  help="Directory for results/metrics")
    p.add_argument("--split",           default="hpa_proteins",
                   choices=["hpa_proteins", "hpa_cell_lines", "misloc",
                             "val_v2", "protein_test_v2", "cell_line_test_v2"],
                   help="Which evaluation split to run")
    p.add_argument("--data_root", default="data/czi_224")
    p.add_argument("--pooled_cache",    default=None,
                   help="Pre-pooled ESM2 pkl or joblib; avoids loading per-protein pkls at inference")
    p.add_argument("--tsv_path",        default=None,
                   help="SwissProt TSV for one_hot mode evaluation")
    p.add_argument("--rna_tsv",         default=None,
                   help="rna_celline.tsv for seq_rna_baseline evaluation")
    # split_v2 args
    p.add_argument("--split_v2_dir",    default=None,
                   help="Path to split_v2 directory (enables v2 splits)")
    p.add_argument("--input_size",      type=int, default=128,
                   help="Image resize resolution passed to DataLoader (default: 128)")
    # Predictor type
    p.add_argument("--predictor_type",  default="localization",
                   choices=["localization", "pups", "protgps"],
                   help="Which predictor to instantiate")
    p.add_argument("--pups_ckpt",       default=None,
                   help="Path to PUPS Lightning .ckpt")
    p.add_argument("--protgps_ckpt",    default=None,
                   help="Path to ProtGPS Lightning .ckpt")
    # Evaluation options
    p.add_argument("--per_protein",     action="store_true",
                   help="Also compute per-protein aggregated metrics + CI")
    p.add_argument("--bootstrap",       type=int, default=1000,
                   help="Bootstrap samples for CI (0 = skip CI)")
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--num_workers",     type=int, default=8)
    p.add_argument("--limit_batches",   type=int, default=None,
                   help="Cap number of batches (smoke-test mode)")
    p.add_argument("--device",          default=None)
    p.add_argument("--esm2_cache_dir",  default=None,
                   help="ESM2 per-protein pkl cache dir")
    p.add_argument("--legacy_model",    default=None,
                   choices=["image_seq_classifier", "image_seq_inpainting"],
                   help="For old checkpoints without embedded model_config")
    return p.parse_args()


def build_predictor(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.predictor_type == "pups":
        from proteinmap.models.pups_adapter import PUPSPredictor
        return PUPSPredictor(
            checkpoint_path=args.pups_ckpt,
            device=device,
            esm2_cache_dir=args.esm2_cache_dir,
        )

    if args.predictor_type == "protgps":
        from proteinmap.models.protgps_adapter import ProtGPSPredictor
        return ProtGPSPredictor(
            checkpoint_path=args.protgps_ckpt,
            device=device,
        )

    # localization predictor (LocalizationModel checkpoint)
    if args.legacy_model is not None:
        return _build_legacy_predictor(args, device)

    pooled_cache_path = args.pooled_cache

    # RNA loader (seq_rna_baseline)
    rna_loader = None
    if args.rna_tsv is not None:
        from proteinmap.data.rna_loader import RNAExpressionLoader
        rna_loader = RNAExpressionLoader(args.rna_tsv)

    return LocalizationPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        pooled_cache_path=pooled_cache_path,
        tsv_path=args.tsv_path,
        rna_loader=rna_loader,
    )


def _build_legacy_predictor(args, device) -> LocalizationPredictor:
    """
    Backward-compat path for checkpoints that pre-date the embedded model_config.
    """
    if args.legacy_model == "image_seq_classifier":
        from proteinmap.models.image_seq_adapter import ImageSeqPredictor
        pooled = args.pooled_cache or f"{args.data_root}/esm2_cache/pooled_esm2.pkl"
        return ImageSeqPredictor(
            checkpoint_path=args.checkpoint,
            device=device,
            pooled_cache_path=pooled,
        )
    else:
        from proteinmap.models.image_seq_adapter import ImageSeqInpaintingPredictor
        pooled = args.pooled_cache or f"{args.data_root}/esm2_cache/pooled_esm2.pkl"
        return ImageSeqInpaintingPredictor(
            checkpoint_path=args.checkpoint,
            device=device,
            pooled_cache_path=pooled,
        )


def build_dataloader(args) -> tuple:
    """Returns (dataloader, split_name)."""
    esm2_cache_dir = args.esm2_cache_dir or f"{args.data_root}/esm2_cache"

    # --- split_v2 splits -----------------------------------------------
    if args.split in _V2_SPLIT_FILES:
        if args.split_v2_dir is None:
            raise ValueError(f"--split_v2_dir is required for split '{args.split}'")
        pairs_csv = str(pathlib.Path(args.split_v2_dir) / _V2_SPLIT_FILES[args.split])
        # Auto-detect skip_image from checkpoint for sequence-only models (saves huge NFS IO)
        skip_image = False
        if args.checkpoint and args.predictor_type == "localization":
            try:
                import pickle
                with open(args.checkpoint, "rb") as _f:
                    _ckpt = pickle.load(_f)
                _vis_type = _ckpt.get("model_config", {}).get("vis_encoder", {}).get("type", "")
                skip_image = (_vis_type == "none")
                if skip_image:
                    print(f"[evaluate.py] vis_encoder=none detected: skipping image loading")
            except Exception:
                pass
        dataset = _CZIWithMetaV2(
            pairs_csv=pairs_csv,
            data_root=args.data_root,
            esm2_cache_dir=esm2_cache_dir,
            input_size=args.input_size,
            skip_image=skip_image,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate_with_meta,
        )
        return loader, args.split

    # --- legacy splits --------------------------------------------------
    if args.split == "hpa_proteins":
        dataset = _CZIWithMeta(
            split=DatasetType.TEST,
            data_root=args.data_root,
            esm2_cache_dir=esm2_cache_dir,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate_with_meta,
        )
        return loader, "hpa_proteins"

    elif args.split == "hpa_cell_lines":
        try:
            from proteinmap.data.hpa_cell_line_split import SubCellDatasetCZICellLine
            dataset = SubCellDatasetCZICellLine(
                split=DatasetType.TEST,
                data_root=args.data_root,
                esm2_cache_dir=esm2_cache_dir,
            )
        except (ImportError, AttributeError):
            raise RuntimeError(
                "hpa_cell_lines split requires SubCellDatasetCZICellLine. "
                "Check proteinmap/data/hpa_cell_line_split.py."
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate_with_meta,
        )
        return loader, "hpa_cell_lines"

    elif args.split == "misloc":
        try:
            from proteinmap.data.misloc_adapter import MislocDataset
            wt_ds = MislocDataset(use_mutant=False, data_root=args.data_root,
                                  esm2_cache_dir=esm2_cache_dir)
            mt_ds = MislocDataset(use_mutant=True,  data_root=args.data_root,
                                  esm2_cache_dir=esm2_cache_dir)
        except (ImportError, AttributeError):
            raise RuntimeError(
                "misloc split requires MislocDataset. "
                "Check proteinmap/data/misloc_adapter.py."
            )
        wt_loader = DataLoader(wt_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=_collate_with_meta)
        mt_loader = DataLoader(mt_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=_collate_with_meta)
        return (wt_loader, mt_loader), "misloc"

    else:
        raise ValueError(f"Unknown split: {args.split!r}")


def main():
    args = parse_args()
    print(f"[evaluate.py] checkpoint:    {args.checkpoint}")
    print(f"              split:         {args.split}")
    print(f"              predictor:     {args.predictor_type}")
    print(f"              per_protein:   {args.per_protein}  bootstrap:{args.bootstrap}")

    predictor = build_predictor(args)
    loader_or_tuple, split_name = build_dataloader(args)

    evaluator = UnifiedEvaluator(
        predictor=predictor,
        output_dir=args.output_dir,
        compute_protein_metrics=args.per_protein,
        n_bootstrap=args.bootstrap,
        limit_batches=args.limit_batches,
    )

    if split_name == "misloc":
        wt_loader, mt_loader = loader_or_tuple
        evaluator.evaluate(wt_loader, split_name="misloc_wt")
        evaluator.evaluate(mt_loader, split_name="misloc_mt")
        # Shift-detection metrics
        try:
            import pandas as pd, pickle, json
            from proteinmap.metrics.compartment_metrics import compute_mutation_shift_detection
            from proteinmap.data.constants import CLASSES as _C
            wt_df = pd.read_pickle(
                str(pathlib.Path(args.output_dir) / "misloc_wt_results.pkl")
            )
            mt_df = pd.read_pickle(
                str(pathlib.Path(args.output_dir) / "misloc_mt_results.pkl")
            )
            loc_csv = pathlib.Path(args.data_root) / "misloc" / "location_pairs.csv"
            if loc_csv.exists():
                loc_pairs = pd.read_csv(loc_csv)
                shift = compute_mutation_shift_detection(wt_df, mt_df, loc_pairs, _C[0])
                out = pathlib.Path(args.output_dir) / "misloc_shift_metrics.json"
                with open(out, "w") as f:
                    json.dump(shift, f, indent=2)
                print(f"  Shift detection metrics saved to {out.name}")
        except Exception as e:
            print(f"  [!] Shift detection skipped: {e}")
    else:
        evaluator.evaluate(loader_or_tuple, split_name=split_name)


if __name__ == "__main__":
    main()
