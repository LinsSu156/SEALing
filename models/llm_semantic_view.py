"""Semantic graph construction and encoding from LLM evidence."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from math import log


class SemanticGraphConv(nn.Module):
    """Aggregate weighted neighbors with degree normalization and a residual path."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        self.res_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = self.W(x)
        with g.local_scope():
            g.ndata["h"] = h
            g.update_all(fn.u_mul_e("h", "ew", "m"), fn.sum("m", "h_agg"))
            h_agg = g.ndata["h_agg"]
            deg = g.in_degrees().float().clamp(min=1).unsqueeze(-1).to(x.device)
            h_agg = h_agg / deg
            return F.gelu(self.norm(h_agg + self.res_proj(x)))


class SemanticViewEncoder(nn.Module):
    """Encode node features through stacked semantic graph convolutions."""

    def __init__(
        self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                SemanticGraphConv(in_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

    def forward(self, sem_g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                h = self.dropout(h)
            h = layer(sem_g, h)
        return h


class SemanticEnhancedPredictor(nn.Module):
    """Unused legacy head retained to preserve seeded fusion initialization."""

    def __init__(self, hidden_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, hidden_dim),
        )


class LLMSemanticViewModule(nn.Module):
    """Construct and encode semantic graphs and pruning scores."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_sem_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph_builder = nn.Module()
        self.graph_builder.log_tau = nn.Parameter(torch.tensor(log(0.1)))
        self.sem_encoder = SemanticViewEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_sem_layers,
            dropout=dropout,
        )
        # Its construction consumes RNG before fusion initialization in the reference.
        # Retain it for seeded reproduction and historical checkpoint compatibility.
        self.predictor = SemanticEnhancedPredictor(hidden_dim, proj_dim=256)

    def build_pruning_scores(
        self, soft_labels: torch.Tensor, anchor_confidence: torch.Tensor
    ) -> torch.Tensor:
        pruning_scores = 0.5 * (soft_labels + soft_labels.T)
        node_conf = anchor_confidence.clamp(min=1e-8)
        pruning_scores = pruning_scores * torch.sqrt(
            node_conf.unsqueeze(1) * node_conf.unsqueeze(0)
        )
        pruning_scores = pruning_scores.clamp(min=1e-8)
        pruning_scores.fill_diagonal_(1.0)
        return pruning_scores

    @staticmethod
    def score_evidence(evidence: torch.Tensor):
        """Derive biological similarity and anchor confidence from [N, 10] evidence."""
        if evidence.ndim != 2 or evidence.size(1) != 10:
            raise ValueError("The mainline requires ten-column semantic evidence.")
        x = evidence.float()
        bio = x[:, :7]
        bio = (bio - bio.mean(dim=0, keepdim=True)) / bio.std(
            dim=0, keepdim=True
        ).clamp_min(1e-6)
        z = F.normalize(bio, p=2, dim=-1)
        cosine = torch.matmul(z, z.T).clamp(-1.0, 1.0)
        scores = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)
        scores.fill_diagonal_(1.0)
        uncertainty = x[:, 7:9].mean(dim=1).clamp(0.0, 1.0)
        confidence = x[:, 9].clamp(0.0, 1.0)
        anchor_confidence = (confidence * (1.0 - 0.5 * uncertainty)).clamp(0.0, 1.0)
        return scores, anchor_confidence
