"""
RNA expression encoders for LocalizationModel.

RNAExpressionEncoder: MLP over gene expression vector → cell embedding.
Designed to be extensible: a scGPT wrapper can be swapped in as encoder_type='scgpt'.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RNAExpressionEncoder(nn.Module):
    """
    Gene expression vector → cell embedding.

    type='mlp':
        Linear(n_genes, hidden_dim) → LayerNorm → ReLU → Linear(hidden_dim, out_dim)

    type='scgpt':
        scGPT foundation model wrapper (future extension; raises NotImplementedError).

    Parameters
    ----------
    n_genes     : int   — input dimension (number of genes in the expression vector)
    out_dim     : int   — output embedding dimension
    hidden_dim  : int   — hidden layer width (MLP only)
    encoder_type: str   — 'mlp' or 'scgpt'
    """

    def __init__(
        self,
        n_genes: int,
        out_dim: int = 256,
        hidden_dim: int = 512,
        encoder_type: str = "mlp",
    ):
        super().__init__()
        self._out_dim = out_dim
        if encoder_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            raise NotImplementedError(f"encoder_type={encoder_type!r} is not implemented")

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, n_genes) → (B, out_dim)"""
        return self.net(x)
