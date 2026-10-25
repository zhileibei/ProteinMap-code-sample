"""
FusionModule — generalises the concat in ImageSeqClassifier.forward().

mode='concat'     : cat([seq, vis]) → Linear(seq_dim+vis_dim, out_dim)
mode='single_seq' : pass seq features through unchanged (or project if out_dim differs)
mode='single_vis' : pass vis features through unchanged (or project if out_dim differs)
gate=True         : apply sigmoid-weighted gating before projection (experimental)
"""

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    Fuses sequence and/or vision embeddings into a single fixed-size vector.

    Args:
        seq_dim: output dim of sequence encoder (0 if no seq encoder)
        vis_dim: output dim of vision encoder   (0 if no vis encoder)
        out_dim: output dim of fusion; defaults to in_dim (no projection)
        mode:    'concat' | 'single_seq' | 'single_vis'
        gate:    sigmoid gating before projection (only in concat mode)
    """

    def __init__(
        self,
        seq_dim: int,
        vis_dim: int = 0,
        out_dim: int = None,
        mode: str = "concat",
        gate: bool = False,
    ):
        super().__init__()
        if mode not in ("concat", "single_seq", "single_vis"):
            raise ValueError(
                f"mode must be 'concat', 'single_seq', or 'single_vis'; got {mode!r}"
            )
        self.mode = mode
        self.gate = gate

        if mode == "concat":
            in_dim = seq_dim + vis_dim
        elif mode == "single_seq":
            in_dim = seq_dim
        else:
            in_dim = vis_dim

        self._out_dim = out_dim if out_dim is not None else in_dim
        self.proj = nn.Linear(in_dim, self._out_dim) if in_dim != self._out_dim else None

        self.gate_layer = (
            nn.Linear(in_dim, in_dim) if (gate and mode == "concat") else None
        )

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        seq_feat: torch.Tensor = None,
        vis_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.mode == "concat":
            assert seq_feat is not None and vis_feat is not None, \
                "concat mode requires both seq_feat and vis_feat"
            x = torch.cat([seq_feat, vis_feat], dim=-1)
        elif self.mode == "single_seq":
            assert seq_feat is not None, "single_seq mode requires seq_feat"
            x = seq_feat
        else:
            assert vis_feat is not None, "single_vis mode requires vis_feat"
            x = vis_feat

        if self.gate_layer is not None:
            x = torch.sigmoid(self.gate_layer(x)) * x

        if self.proj is not None:
            x = self.proj(x)
        return x
