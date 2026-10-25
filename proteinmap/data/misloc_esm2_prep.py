"""
Utilities for generating and caching ESM-2 embeddings for misloc sequences.

Reuses the same ESM-2 model and embedding format as the PUPS pipeline so that
the same cache files are compatible.

Cache format (pickle):
    {"embedding": np.ndarray (2000, 1280) float32, "length": int}

Called by scripts/01_prep_misloc_esm2.py.
"""

import pathlib
import pickle
import re
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def apply_point_mutation(sequence: str, mutation_str: str) -> Optional[str]:
    """
    Apply a single amino-acid point mutation to a sequence.

    mutation_str format: {wt_aa}{position}{mut_aa}  e.g. "V600E"

    Returns the mutated sequence, or None if the mutation is invalid
    (position out of range, WT AA mismatch, or unrecognised format).
    """
    pattern = re.compile(r"^([A-Z])(\d+)([A-Z])$")
    m = pattern.match(mutation_str.strip())
    if not m:
        return None
    wt_aa, pos_str, mut_aa = m.group(1), m.group(2), m.group(3)
    pos = int(pos_str) - 1  # 0-indexed
    if pos < 0 or pos >= len(sequence):
        return None
    if sequence[pos] != wt_aa:
        # WT mismatch — log but still apply the substitution
        pass
    mutated = sequence[:pos] + mut_aa + sequence[pos + 1:]
    return mutated


def load_swissprot_sequences(tsv_path: str) -> dict:
    """
    Load WT protein sequences from swissprot TSV.

    Expected columns: Gene_names (space-separated), Sequence
    Returns: {gene_upper: sequence_str}
    """
    df = pd.read_csv(tsv_path, sep="\t")
    seq_map = {}
    gene_col = None
    for candidate in ["Gene names", "Gene_names", "gene_names", "Gene names  (primary )", "Gene_name", "gene_name"]:
        if candidate in df.columns:
            gene_col = candidate
            break
    if gene_col is None:
        raise ValueError(f"Could not find gene-name column in {tsv_path}. Columns: {list(df.columns)}")
    seq_col = None
    for candidate in ["Sequence", "sequence"]:
        if candidate in df.columns:
            seq_col = candidate
            break
    if seq_col is None:
        raise ValueError(f"Could not find sequence column in {tsv_path}. Columns: {list(df.columns)}")

    for _, row in df.iterrows():
        gene_names_raw = str(row[gene_col]) if pd.notna(row[gene_col]) else ""
        seq = str(row[seq_col]).strip().replace(" ", "")
        for g in gene_names_raw.split():
            seq_map[g.upper()] = seq
    return seq_map


# ---------------------------------------------------------------------------
# ESM-2 embedding
# ---------------------------------------------------------------------------

def embed_sequences_esm2(
    sequences: dict,
    cache_dir: str,
    device: str = None,
    max_length: int = 2000,
    batch_size: int = 1,
    overwrite: bool = False,
) -> None:
    """
    Embed a dict of {cache_key: sequence} using ESM-2 and save to cache_dir.

    The output format matches the PUPS ESM2 cache:
        {"embedding": (2000, 1280) float32 zero-padded, "length": int}

    Args:
        sequences: {cache_key: amino_acid_sequence}
        cache_dir: directory to write .pkl files
        device: "cuda" or "cpu"
        max_length: pad/truncate to this length (must match PUPS training)
        overwrite: re-embed even if cache file exists
    """
    import torch
    import esm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Filter to sequences that need embedding
    to_embed = {}
    for key, seq in sequences.items():
        pkl_path = cache_dir / f"{key}.pkl"
        if overwrite or not pkl_path.exists():
            to_embed[key] = seq

    if not to_embed:
        print("All sequences already cached — nothing to embed.")
        return

    print(f"Embedding {len(to_embed)} sequences with ESM-2 ...")

    # Load ESM-2 model (650M parameter version used by PUPS)
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    keys = list(to_embed.keys())
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i: i + batch_size]
        batch_data = [(k, to_embed[k][:max_length]) for k in batch_keys]
        _labels, _strs, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[33], return_contacts=False)
        token_repr = results["representations"][33]  # (B, L+2, 1280) incl. BOS/EOS

        for j, key in enumerate(batch_keys):
            seq_len = len(to_embed[key])
            # Remove BOS/EOS; take up to max_length positions
            emb = token_repr[j, 1: min(seq_len + 1, max_length + 1), :].cpu().numpy()
            actual_len = emb.shape[0]
            # Zero-pad to max_length
            padded = np.zeros((max_length, 1280), dtype=np.float32)
            padded[:actual_len] = emb.astype(np.float32)

            pkl_path = cache_dir / f"{key}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump({"embedding": padded, "length": actual_len}, f)
            print(f"  Cached {pkl_path.name} (len={actual_len})")

    print("ESM-2 embedding complete.")
