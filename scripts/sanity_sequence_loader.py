"""
Sanity check 1 — verify TSV join and one-hot encoding.

Checks:
1. Load TSV → {ensembl_id: sequence}; print row count
2. Load HPA metadata.csv, extract unique primary Ensembl IDs from test split
3. Print coverage: N_found / N_total; list first 10 missing IDs
4. For 5 random proteins: assert one-hot shape (MAX_SEQ_LEN, 20) and non-zero rows
5. Assert ESM2 pooled cache keys ≥80% covered by TSV Ensembl IDs
6. Assert SequenceLoader(esm2_pooled) and SequenceLoader(one_hot) agree on serveable proteins

Usage:
    python scripts/sanity_sequence_loader.py \\
        --tsv_path data/sequences.tsv \\
        --data_root data/czi_224 \\
        --pooled_cache data/esm2_cache/pooled_esm2.pkl
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from proteinmap.data.sequence_loader import (
    SequenceLoader,
    load_tsv_sequence_map,
    one_hot_encode,
    MAX_SEQ_LEN,
    ONEHOT_DIM,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv_path",    required=True)
    p.add_argument("--data_root",   required=True)
    p.add_argument("--pooled_cache", required=True)
    return p.parse_args()


def main():
    args = parse_args()
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

    print("=" * 60)
    print("Sanity 1: SequenceLoader")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load TSV
    # ------------------------------------------------------------------
    print("\n[1] Loading TSV sequence map ...")
    seq_map = load_tsv_sequence_map(args.tsv_path)
    print(f"    {len(seq_map)} unique Ensembl IDs in TSV")
    check(len(seq_map) > 0, f"TSV loaded non-empty ({len(seq_map)} IDs)")

    # ------------------------------------------------------------------
    # Step 2: Load HPA metadata, extract test split primary IDs
    # ------------------------------------------------------------------
    print("\n[2] Loading HPA metadata (test split) ...")
    data_root = args.data_root
    test_abs_file = f"{data_root}/annotations/test_antibodies.txt"
    test_antibodies = set(open(test_abs_file).read().splitlines())
    df = pd.read_csv(f"{data_root}/cell_crops/metadata.csv")
    test_df = df[df["antibody"].isin(test_antibodies)]
    primary_ids = (
        test_df["ensembl_ids"].str.strip().str.split(",").str[0].str.strip().unique().tolist()
    )
    print(f"    {len(primary_ids)} unique primary Ensembl IDs in test split")
    check(len(primary_ids) > 0, f"Test split has protein IDs ({len(primary_ids)})")

    # ------------------------------------------------------------------
    # Step 3: Coverage
    # ------------------------------------------------------------------
    print("\n[3] Coverage: test proteins in TSV ...")
    found = [eid for eid in primary_ids if eid in seq_map]
    missing = [eid for eid in primary_ids if eid not in seq_map]
    coverage = len(found) / len(primary_ids)
    print(f"    Found: {len(found)} / {len(primary_ids)} = {coverage:.1%}")
    print(f"    First 10 missing: {missing[:10]}")
    check(coverage >= 0.5, f"TSV coverage ≥ 50% ({coverage:.1%})")   # lenient threshold

    # ------------------------------------------------------------------
    # Step 4: One-hot encoding shape and content
    # ------------------------------------------------------------------
    print("\n[4] One-hot encoding checks (5 random proteins) ...")
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(found, size=min(5, len(found)), replace=False)
    for eid in sample_ids:
        seq = seq_map[eid]
        oh = one_hot_encode(seq)
        check(oh.shape == (MAX_SEQ_LEN, ONEHOT_DIM),
              f"{eid}: one-hot shape {oh.shape} == ({MAX_SEQ_LEN}, {ONEHOT_DIM})")
        nonzero_rows = int((oh.sum(axis=1) > 0).sum())
        expected = min(len(seq), MAX_SEQ_LEN)
        check(
            nonzero_rows == expected,
            f"{eid}: non-zero rows {nonzero_rows} == sequence length {expected}"
        )

    # ------------------------------------------------------------------
    # Step 5: ESM2 cache cross-validation
    # ------------------------------------------------------------------
    print("\n[5] Cross-validating ESM2 cache keys against TSV ...")
    import pickle
    with open(args.pooled_cache, "rb") as f:
        pooled = pickle.load(f)
    esm2_ids = set(pooled.keys())
    tsv_ids = set(seq_map.keys())
    overlap = esm2_ids & tsv_ids
    overlap_frac = len(overlap) / len(esm2_ids)
    print(f"    ESM2 cache: {len(esm2_ids)} IDs")
    print(f"    TSV:        {len(tsv_ids)} IDs")
    print(f"    Overlap:    {len(overlap)} ({overlap_frac:.1%} of ESM2 keys in TSV)")
    check(overlap_frac >= 0.80,
          f"ESM2 cache coverage in TSV ≥ 80% ({overlap_frac:.1%})")

    # ------------------------------------------------------------------
    # Step 6: SequenceLoader agreement
    # ------------------------------------------------------------------
    print("\n[6] SequenceLoader(esm2_pooled) vs SequenceLoader(one_hot) agreement ...")
    esm2_loader = SequenceLoader("esm2_pooled", pooled_cache_path=args.pooled_cache)
    oh_loader   = SequenceLoader("one_hot",     tsv_path=args.tsv_path)

    common_ids = list(esm2_ids & tsv_ids)[:20]
    esm2_serveable = sum(1 for eid in common_ids if esm2_loader[eid].esm2_pooled is not None)
    oh_serveable   = sum(1 for eid in common_ids if oh_loader[eid].one_hot is not None)
    check(esm2_serveable == len(common_ids),
          f"ESM2 loader serves all {len(common_ids)} common IDs")
    check(oh_serveable == len(common_ids),
          f"One-hot loader serves all {len(common_ids)} common IDs")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
