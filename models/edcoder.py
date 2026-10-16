from typing import Optional
import torch
import torch.nn as nn
from functools import partial
import dgl

from .gat import GAT
from .llm_semantic_view import LLMSemanticViewModule

from .loss_func import sce_loss


def setup_module(
    enc_dec,
    in_dim,
    num_hidden,
    out_dim,
    num_layers,
    dropout,
    activation,
    residual,
    norm,
    nhead,
    nhead_out,
    attn_drop,
    negative_slope=0.2,
    concat_out=True,
) -> nn.Module:
    return GAT(
        in_dim=in_dim,
        num_hidden=num_hidden,
        out_dim=out_dim,
        num_layers=num_layers,
        nhead=nhead,
        nhead_out=nhead_out,
        concat_out=concat_out,
        activation=activation,
        feat_drop=dropout,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        norm=norm,
        encoding=(enc_dec == "encoding"),
    )


class PreModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        num_layers: int,
        num_dec_layers: int,
        nhead: int,
        nhead_out: int,
        activation: str,
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        residual: bool,
        norm: Optional[str],
        alpha_l: float = 3,
        zero_init: bool = False,
        sem_num_layers: int = 2,
        sem_weak_edge_prune_rate: float = 0.035,
        sem_weak_edge_temperature: float = 1.0,
        sem_struct_shared_neighbor_weight: float = 1.0,
        sem_struct_density_weight: float = 1.0,
        sem_struct_mutual_bonus: float = 1.0,
        sem_struct_keep_threshold: float = 0.5,
        sem_anchor_confidence_temperature: float = 1.0,
        sem_anchor_confidence_min: float = 0.30,
        align_loss_weight: float = 0.02,
        contrast_loss_weight: float = 0.007,
        contrast_temperature: float = 0.2,
        contrast_negative_top_k: int = 32,
        contrast_semantic_negative_quantile: float = 0.10,
        contrast_positive_min_score: float = 0.60,
        contrast_unknown_weight: float = 0.10,
        contrast_warmup_epochs: int = 50,
        contrast_decay_start: int = 300,
        contrast_final_scale: float = 0.50,
    ):
        super(PreModel, self).__init__()
        for name, value in {
            "align_loss_weight": align_loss_weight,
            "contrast_loss_weight": contrast_loss_weight,
            "sem_weak_edge_prune_rate": sem_weak_edge_prune_rate,
            "contrast_temperature": contrast_temperature,
            "contrast_negative_top_k": contrast_negative_top_k,
            "contrast_unknown_weight": contrast_unknown_weight,
            "contrast_warmup_epochs": contrast_warmup_epochs,
            "contrast_final_scale": contrast_final_scale,
            "sem_anchor_confidence_min": sem_anchor_confidence_min,
        }.items():
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be positive for the complete mainline.")
        self._sem_weak_edge_prune_rate = sem_weak_edge_prune_rate
        self._sem_weak_edge_temperature = sem_weak_edge_temperature
        self._sem_struct_shared_neighbor_weight = sem_struct_shared_neighbor_weight
        self._sem_struct_density_weight = sem_struct_density_weight
        self._sem_struct_mutual_bonus = sem_struct_mutual_bonus
        self._sem_struct_keep_threshold = sem_struct_keep_threshold
        self._sem_anchor_confidence_temperature = sem_anchor_confidence_temperature
        self._sem_anchor_confidence_min = sem_anchor_confidence_min
        self._align_loss_weight = align_loss_weight
        self._contrast_loss_weight = contrast_loss_weight
        self._contrast_temperature = contrast_temperature
        self._contrast_positive_min_score = float(contrast_positive_min_score)
        self._soft_labels = None
        self._anchor_confidence = None
        self._contrast_negative_top_k = int(contrast_negative_top_k)
        self._contrast_semantic_negative_quantile = float(
            contrast_semantic_negative_quantile
        )
        self._contrast_unknown_weight = float(contrast_unknown_weight)
        self._contrast_warmup_epochs = int(contrast_warmup_epochs)
        self._contrast_decay_start = int(contrast_decay_start)
        self._contrast_final_scale = float(contrast_final_scale)
        self._view_fusion_gate_stats = {
            "view_fusion_gate": 0.5,
            "view_fusion_gate_min": 0.5,
            "view_fusion_gate_max": 0.5,
        }
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        enc_num_hidden = num_hidden // nhead
        enc_nhead = nhead

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead

        self.struct_encoder = setup_module(
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.decoder = setup_module(
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            nhead_out=nhead_out,
            num_layers=num_dec_layers,
            nhead=nhead,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        if not zero_init:
            nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

        self.criterion = partial(sce_loss, alpha=alpha_l)
        self.semantic_view = LLMSemanticViewModule(
            in_dim=in_dim,
            hidden_dim=num_hidden,
            num_sem_layers=sem_num_layers,
            dropout=feat_drop,
        )
        self.view_fusion = nn.Sequential(
            nn.Linear(num_hidden * 2, num_hidden),
            nn.LayerNorm(num_hidden),
            nn.GELU(),
        )
        self.view_fusion_gate = nn.Parameter(torch.tensor(0.0))
        self._pruning_scores = None

        self._cached_pruned_graph_key = None
        self._cached_pruned_graph = None
        self._cached_prune_stats = None
        self._cached_semantic_view_graph_key = None
        self._cached_semantic_view_graph = None

    def forward(self, g, x, epoch=0, return_details=False):
        return self.semantic_guided_prediction(
            g, x, epoch=epoch, return_details=return_details
        )

    def _invalidate_static_graph_cache(self):
        self._cached_pruned_graph_key = None
        self._cached_pruned_graph = None
        self._cached_prune_stats = None
        self._cached_semantic_view_graph_key = None
        self._cached_semantic_view_graph = None

    def _pruned_graph_cache_key(self, g):
        score_ptr = (
            int(self._pruning_scores.data_ptr())
            if self._pruning_scores is not None
            and hasattr(self._pruning_scores, "data_ptr")
            else id(self._pruning_scores)
        )
        edge_data_ptrs = tuple(
            (name, int(g.edata[name].data_ptr()))
            for name in ("weight", "w", "ew")
            if name in g.edata and hasattr(g.edata[name], "data_ptr")
        )
        return (
            id(g),
            int(g.num_nodes()),
            int(g.num_edges()),
            str(g.device),
            score_ptr,
            float(self._sem_weak_edge_prune_rate),
            float(self._sem_weak_edge_temperature),
            float(self._sem_struct_keep_threshold),
            edge_data_ptrs,
        )

    def _get_pruned_graph(self, g):
        if self._pruning_scores is None:
            raise RuntimeError(
                "Semantic pruning graph is not ready. Please call update_semantic_graph(llm_emb) before training or inference."
            )
        cache_key = self._pruned_graph_cache_key(g)
        if (
            self._cached_pruned_graph_key == cache_key
            and self._cached_pruned_graph is not None
        ):
            return self._cached_pruned_graph, dict(self._cached_prune_stats)
        pruned_g, prune_stats = self.build_llm_pruned_graph(g, self._pruning_scores)
        self._cached_pruned_graph_key = cache_key
        self._cached_pruned_graph = pruned_g
        self._cached_prune_stats = dict(prune_stats)
        self._cached_semantic_view_graph_key = None
        self._cached_semantic_view_graph = None
        return pruned_g, dict(prune_stats)

    def encode_views(self, g, x):
        pruned_g, prune_stats = self._get_pruned_graph(g)
        h_struct = self.struct_encoder(pruned_g, x)
        sem_view_g = self._get_semantic_view_graph(pruned_g, x.device)
        prune_stats["semantic_view_edges"] = int(sem_view_g.num_edges())
        h_sem = self.semantic_view.sem_encoder(sem_view_g, x)
        return h_struct, h_sem, pruned_g, prune_stats

    def fuse_views(self, h_struct, h_sem):
        h_joint = self.view_fusion(torch.cat([h_struct, h_sem], dim=-1))
        gate_tensor = torch.sigmoid(self.view_fusion_gate)
        fused = (1.0 - gate_tensor) * h_struct + gate_tensor * h_joint
        gate = float(gate_tensor.detach().item())
        self._view_fusion_gate_stats = {
            "view_fusion_gate": gate,
            "view_fusion_gate_min": gate,
            "view_fusion_gate_max": gate,
        }
        return fused

    def compute_alignment_loss(self, h_struct, h_sem):
        align = (
            1.0 - torch.nn.functional.cosine_similarity(h_struct, h_sem, dim=-1).mean()
        )
        return align

    def _safe_quantile(self, values, quantile, max_values: int = 1_000_000):
        values = values.reshape(-1)
        if values.numel() == 0:
            return values.new_tensor(0.0)

        q = min(1.0, max(0.0, float(quantile)))
        if values.numel() > max_values:
            step = max(1, int(values.numel() // max_values))
            values = values[::step]
            if values.numel() > max_values:
                values = values[:max_values]
        return torch.quantile(values, q)

    def compute_similarity_logits(self, h_struct, h_sem):
        z_struct = torch.nn.functional.normalize(h_struct, p=2, dim=-1)
        z_sem = torch.nn.functional.normalize(h_sem, p=2, dim=-1)
        return (
            torch.matmul(z_struct, z_sem.transpose(0, 1)) / self._contrast_temperature
        )

    def _build_graph_edge_mask(self, graph, num_nodes, device):
        src, dst = graph.edges()
        graph_mask = torch.zeros(
            (num_nodes, num_nodes), device=device, dtype=torch.bool
        )
        graph_mask[src, dst] = True
        graph_mask[dst, src] = True
        graph_mask.fill_diagonal_(True)
        return graph_mask

    def _build_semantic_local_graph(self, graph):
        src, dst = graph.edges()
        sem_g = dgl.graph((src, dst), num_nodes=graph.num_nodes(), device=src.device)
        edge_weight = self._soft_labels[src, dst].to(
            device=src.device, dtype=torch.float32
        )

        confidence = self._anchor_confidence.to(device=src.device, dtype=torch.float32)
        edge_weight = edge_weight * confidence[src] * confidence[dst]
        sem_g.edata["ew"] = edge_weight.clamp(min=1e-4).unsqueeze(-1)
        return sem_g

    def _get_semantic_view_graph(self, pruned_g, device):
        soft_label_ptr = (
            int(self._soft_labels.data_ptr())
            if self._soft_labels is not None and hasattr(self._soft_labels, "data_ptr")
            else id(self._soft_labels)
        )
        anchor_ptr = (
            int(self._anchor_confidence.data_ptr())
            if self._anchor_confidence is not None
            and hasattr(self._anchor_confidence, "data_ptr")
            else id(self._anchor_confidence)
        )
        cache_key = (
            id(pruned_g),
            int(pruned_g.num_nodes()),
            int(pruned_g.num_edges()),
            str(pruned_g.device),
            str(device),
            soft_label_ptr,
            anchor_ptr,
        )
        if (
            self._cached_semantic_view_graph_key == cache_key
            and self._cached_semantic_view_graph is not None
        ):
            return self._cached_semantic_view_graph
        sem_g = self._build_semantic_local_graph(pruned_g)
        self._cached_semantic_view_graph_key = cache_key
        self._cached_semantic_view_graph = sem_g
        return sem_g

    def _build_filtered_semantic_neighbor_mask(self, num_nodes, device, pruned_g=None):
        if pruned_g is None:
            raise RuntimeError(
                "Pruned graph is required for local semantic-neighbor contrastive."
            )
        if self._soft_labels is None:
            raise RuntimeError(
                "Semantic scores are required for semantic-neighbor contrastive."
            )

        src, dst = pruned_g.edges()
        edge_weight = self._soft_labels[src, dst].to(device=device, dtype=torch.float32)

        confidence = self._anchor_confidence.to(device=device, dtype=torch.float32)
        edge_weight = edge_weight * confidence[src] * confidence[dst]
        edge_weight = edge_weight.clamp(min=1e-6)

        not_self = src != dst
        keep = not_self & (
            self._soft_labels[src, dst].to(device=device)
            >= self._contrast_positive_min_score
        )

        candidate_src = src[keep]
        candidate_dst = dst[keep]
        candidate_weight = edge_weight[keep]
        pos_mask = torch.eye(num_nodes, device=device, dtype=torch.float32)

        if candidate_src.numel() > 0:
            for node_idx in range(num_nodes):
                node_edge_mask = candidate_src == node_idx
                if not bool(node_edge_mask.any()):
                    continue
                node_dst = candidate_dst[node_edge_mask]
                node_weight = candidate_weight[node_edge_mask]
                if node_dst.numel() > 1:
                    selected = torch.topk(node_weight, k=1).indices
                    node_dst = node_dst[selected]
                    node_weight = node_weight[selected]
                pos_mask[node_idx, node_dst] = node_weight

        stats = {
            "contrast_candidate_pairs": int(not_self.sum().item()),
            "contrast_filtered_pairs": int((pos_mask > 0).sum().item()),
        }
        return pos_mask, stats

    def _build_k_hop_graph_mask(self, graph, num_nodes, device, hops):
        local_mask = self._build_graph_edge_mask(graph, num_nodes, device)
        local_mask.fill_diagonal_(True)
        reach = local_mask
        frontier = local_mask
        for _ in range(1, max(1, int(hops))):
            frontier = (
                torch.matmul(
                    frontier.to(dtype=torch.float32), local_mask.to(dtype=torch.float32)
                )
                > 0
            )
            reach = reach | frontier
        reach.fill_diagonal_(True)
        return reach

    def build_contrastive_negative_mask(
        self, num_nodes, device, pruned_g, pos_mask, base_g=None
    ):
        if pruned_g is None:
            raise RuntimeError(
                "Pruned graph is required for structural negative contrastive sampling."
            )
        if self._soft_labels is None:
            raise RuntimeError(
                "Semantic scores are required for semantic-structural negative sampling."
            )

        local_mask = self._build_k_hop_graph_mask(pruned_g, num_nodes, device, hops=2)
        if base_g is not None:
            local_mask = local_mask | self._build_k_hop_graph_mask(
                base_g, num_nodes, device, hops=2
            )
        semantic_scores = self._soft_labels.to(device=device).detach().clamp(0.0, 1.0)
        reliability = self._anchor_confidence.to(device=device).detach().clamp(0.0, 1.0)
        reliability_weight = torch.sqrt(
            reliability.unsqueeze(1) * reliability.unsqueeze(0)
        )

        finite_sem = semantic_scores[
            ~torch.eye(num_nodes, device=device, dtype=torch.bool)
        ]
        if finite_sem.numel() == 0:
            threshold = semantic_scores.new_tensor(0.0)
        else:
            threshold = self._safe_quantile(
                finite_sem,
                float(self._contrast_semantic_negative_quantile),
            )

        semantic_low = semantic_scores <= threshold
        reliable_node = reliability >= self._sem_anchor_confidence_min
        reliable_pair = reliable_node.unsqueeze(1) & reliable_node.unsqueeze(0)
        candidate_mask = (~local_mask) & semantic_low & reliable_pair & (pos_mask <= 0)
        candidate_mask.fill_diagonal_(False)

        neg_weight = (1.0 - semantic_scores) * reliability_weight
        score = neg_weight.masked_fill(~candidate_mask, float("-inf"))
        k = min(self._contrast_negative_top_k, num_nodes)
        neg_idx = torch.topk(score, k=k, dim=1, largest=True).indices
        neg_mask = torch.zeros_like(candidate_mask)
        neg_mask.scatter_(1, neg_idx, True)
        neg_mask = neg_mask & candidate_mask

        candidate_scores = semantic_scores[candidate_mask]
        selected_scores = semantic_scores[neg_mask]
        selected_weight = neg_weight[neg_mask]
        if candidate_scores.numel() == 0:
            candidate_scores = semantic_scores.new_zeros(1)
        if selected_scores.numel() == 0:
            selected_scores = semantic_scores.new_zeros(1)
        if selected_weight.numel() == 0:
            selected_weight = semantic_scores.new_zeros(1)

        neg_weight = neg_weight * neg_mask.to(dtype=torch.float32)
        negative_pairs_per_node = (neg_weight > 0).sum(dim=1).to(dtype=torch.float32)
        stats = {
            "contrast_negative_pairs": int((neg_weight > 0).sum().item()),
            "contrast_negative_pairs_avg": float(negative_pairs_per_node.mean().item()),
            "contrast_global_negative_pairs": int((neg_mask > 0).sum().item()),
            "contrast_candidate_sem_mean": float(candidate_scores.mean().item()),
            "contrast_candidate_sem_min": float(candidate_scores.min().item()),
            "contrast_candidate_sem_max": float(candidate_scores.max().item()),
            "contrast_negative_sem_mean": float(selected_scores.mean().item()),
            "contrast_negative_sem_min": float(selected_scores.min().item()),
            "contrast_negative_sem_max": float(selected_scores.max().item()),
            "contrast_negative_weight_mean": float(selected_weight.mean().item()),
            "contrast_negative_weight_min": float(selected_weight.min().item()),
            "contrast_negative_weight_max": float(selected_weight.max().item()),
            "contrast_semantic_negative_threshold": float(threshold.item()),
        }
        return neg_weight, stats

    def build_contrastive_positive_mask(self, num_nodes, device, pruned_g=None):
        return self._build_filtered_semantic_neighbor_mask(
            num_nodes, device, pruned_g=pruned_g
        )

    def build_contrastive_masks(self, num_nodes, device, base_g=None, pruned_g=None):
        pos_mask, filter_stats = self.build_contrastive_positive_mask(
            num_nodes, device, pruned_g=pruned_g
        )
        neg_mask, negative_stats = self.build_contrastive_negative_mask(
            num_nodes, device, pruned_g=pruned_g, pos_mask=pos_mask, base_g=base_g
        )
        return pos_mask, neg_mask, filter_stats, negative_stats

    def _masked_weighted_nce_loss(self, logits, pos_weight, neg_weight):
        pos_mask = pos_weight > 0
        neg_mask = neg_weight > 0
        unknown_mask = (~pos_mask) & (~neg_mask)
        den_weight = pos_mask.to(dtype=logits.dtype) + neg_weight
        den_weight = (
            den_weight
            + unknown_mask.to(dtype=logits.dtype) * self._contrast_unknown_weight
        )
        valid_mask = den_weight > 0
        weighted_logits = logits + den_weight.clamp_min(1e-8).log()
        masked_logits = weighted_logits.masked_fill(~valid_mask, float("-inf"))
        log_denominator = torch.logsumexp(masked_logits, dim=1, keepdim=True)
        log_prob = logits - log_denominator
        pos_mass = pos_weight.sum(dim=1).clamp_min(1.0)
        return -(pos_weight * log_prob).sum(dim=1) / pos_mass

    def _bidirectional_nce_loss(self, query_rep, key_rep, pos_weight, neg_weight):
        logits = self.compute_similarity_logits(query_rep, key_rep)
        loss_row = self._masked_weighted_nce_loss(logits, pos_weight, neg_weight)
        loss_col = self._masked_weighted_nce_loss(
            logits.transpose(0, 1),
            pos_weight.transpose(0, 1),
            neg_weight.transpose(0, 1),
        )
        return 0.5 * (loss_row.mean() + loss_col.mean())

    def current_contrast_weight(self, epoch):
        scale = min(1.0, float(epoch + 1) / float(self._contrast_warmup_epochs))
        if epoch >= self._contrast_decay_start:
            scale = min(scale, self._contrast_final_scale)
        return float(self._contrast_loss_weight) * scale

    def compute_contrastive_loss(self, h_struct, h_sem, base_g=None, pruned_g=None):
        pos_mask, neg_mask, filter_stats, negative_stats = self.build_contrastive_masks(
            h_struct.size(0), h_struct.device, base_g=base_g, pruned_g=pruned_g
        )
        loss = self._bidirectional_nce_loss(h_struct, h_sem, pos_mask, neg_mask)
        stats = {
            **filter_stats,
            **negative_stats,
            "contrast_positive_pairs": int((pos_mask > 0).sum().item()),
            "contrast_positive_pairs_avg": float(pos_mask.sum(dim=1).mean().item()),
            "contrast_loss_nce": float(loss.detach().item()),
        }
        return loss, stats

    def semantic_guided_prediction(self, g, x, epoch=0, return_details=False):
        """Encode both views and optimize reconstruction, alignment, and contrast."""
        h_struct, h_sem, pruned_g, prune_stats = self.encode_views(g, x)
        fused_rep = self.fuse_views(h_struct, h_sem)
        loss_rec = self.compute_reconstruction_loss(fused_rep, x, pruned_g)
        loss_align = self.compute_alignment_loss(h_struct, h_sem)
        loss_contrast, contrast_stats = self.compute_contrastive_loss(
            h_struct, h_sem, base_g=g, pruned_g=pruned_g
        )
        contrast_weight = self.current_contrast_weight(epoch)
        loss = (
            loss_rec
            + self._align_loss_weight * loss_align
            + contrast_weight * loss_contrast
        )
        if return_details:
            return {
                **prune_stats,
                **contrast_stats,
                **self._view_fusion_gate_stats,
                "loss": loss,
                "loss_rec": loss_rec.detach(),
                "loss_align": loss_align.detach(),
                "loss_contrast": loss_contrast.detach(),
                "align_loss_weight": float(self._align_loss_weight),
                "contrast_loss_weight": contrast_weight,
            }
        return loss

    def compute_reconstruction_loss(self, enc_rep, x, graph):
        """Decode fused representations and compare reconstructed node features."""
        origin_rep = self.encoder_to_decoder(enc_rep)
        recon = self.decoder(graph, origin_rep)
        loss_rec = self.criterion(x, recon)
        return loss_rec

    @torch.no_grad()
    def compute_structural_confidence(self, g):
        src, dst = g.edges()
        num_edges = int(src.numel())
        if num_edges == 0:
            empty = torch.zeros(
                0, device=g.device if hasattr(g, "device") else src.device
            )
            return empty, empty.bool()

        num_nodes = int(g.num_nodes())
        src_cpu = src.detach().cpu().tolist()
        dst_cpu = dst.detach().cpu().tolist()

        out_neighbors = [set() for _ in range(num_nodes)]
        in_neighbors = [set() for _ in range(num_nodes)]
        for u, v in zip(src_cpu, dst_cpu):
            out_neighbors[u].add(v)
            in_neighbors[v].add(u)

        undirected_neighbors = [
            out_neighbors[i] | in_neighbors[i] for i in range(num_nodes)
        ]
        local_density = torch.tensor(
            [float(len(undirected_neighbors[i])) for i in range(num_nodes)],
            device=src.device,
            dtype=torch.float32,
        )
        density_norm = local_density / local_density.max().clamp(min=1.0)

        confidence = torch.zeros(num_edges, device=src.device, dtype=torch.float32)
        mutual_mask = torch.zeros(num_edges, device=src.device, dtype=torch.bool)
        for idx, (u, v) in enumerate(zip(src_cpu, dst_cpu)):
            neighbors_u = undirected_neighbors[u]
            neighbors_v = undirected_neighbors[v]
            union_neighbors = neighbors_u | neighbors_v
            if union_neighbors:
                shared_ratio = len(neighbors_u & neighbors_v) / len(union_neighbors)
            else:
                shared_ratio = 0.0

            mutual = 1.0 if (u in out_neighbors[v] and v in out_neighbors[u]) else 0.0
            mutual_mask[idx] = bool(mutual)
            density_score = float((density_norm[u] + density_norm[v]).item() * 0.5)

            score = (
                self._sem_struct_shared_neighbor_weight * shared_ratio
                + self._sem_struct_density_weight * density_score
                + self._sem_struct_mutual_bonus * mutual
            )
            confidence[idx] = score

        max_score = confidence.max().clamp(min=1e-8)
        confidence = confidence / max_score
        return confidence, mutual_mask

    @torch.no_grad()
    def build_llm_pruned_graph(self, g, soft_labels):
        prune_rate = float(self._sem_weak_edge_prune_rate)
        default_stats = {
            "pruned_edges": 0,
            "total_edges": int(g.num_edges()),
            "prune_ratio": 0.0,
            "semantic_strength_mean": 0.0,
            "semantic_strength_min": 0.0,
            "semantic_strength_max": 0.0,
            "structural_confidence_mean": 0.0,
            "structural_confidence_min": 0.0,
            "structural_confidence_max": 0.0,
            "eligible_prune_edges": 0,
            "prune_rank_score_mean": 0.0,
            "prune_rank_score_min": 0.0,
            "prune_rank_score_max": 0.0,
        }
        if g.num_edges() == 0:
            return g, default_stats

        src, dst = g.edges()

        semantic_strength = soft_labels[src, dst]
        if self._sem_weak_edge_temperature != 1.0:
            semantic_strength = semantic_strength.pow(self._sem_weak_edge_temperature)

        semantic_strength = semantic_strength.clamp(min=1e-8)
        structural_confidence, mutual_mask = self.compute_structural_confidence(g)
        structural_confidence = structural_confidence.clamp(min=0.0, max=1.0)
        total_edges = int(src.numel())
        num_prune = int(total_edges * prune_rate)

        stats = {
            **default_stats,
            "total_edges": total_edges,
            "semantic_strength_mean": float(semantic_strength.mean().item()),
            "semantic_strength_min": float(semantic_strength.min().item()),
            "semantic_strength_max": float(semantic_strength.max().item()),
            "structural_confidence_mean": float(structural_confidence.mean().item()),
            "structural_confidence_min": float(structural_confidence.min().item()),
            "structural_confidence_max": float(structural_confidence.max().item()),
        }
        eligible_mask = (
            structural_confidence <= self._sem_struct_keep_threshold
        ) & (~mutual_mask)
        eligible_idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        stats["eligible_prune_edges"] = int(eligible_idx.numel())
        if num_prune <= 0 or eligible_idx.numel() == 0:
            return g, stats

        max_prunable = max(0, min(total_edges - 1, int(eligible_idx.numel())))
        num_prune = min(num_prune, max_prunable)
        if num_prune <= 0:
            return g, stats

        prune_rank_score = semantic_strength
        eligible_score = prune_rank_score[eligible_idx]
        local_rank = torch.argsort(eligible_score)[:num_prune]
        prune_edge_idx = eligible_idx[local_rank]
        selected_score = prune_rank_score[prune_edge_idx]
        stats.update(
            {
                "prune_rank_score_mean": float(selected_score.mean().item()),
                "prune_rank_score_min": float(selected_score.min().item()),
                "prune_rank_score_max": float(selected_score.max().item()),
            }
        )
        keep_mask = torch.ones(total_edges, device=src.device, dtype=torch.bool)
        keep_mask[prune_edge_idx] = False

        if keep_mask.sum() == 0:
            strongest_idx = torch.argmax(semantic_strength)
            keep_mask[strongest_idx] = True

        new_src = src[keep_mask]
        new_dst = dst[keep_mask]
        pruned_g = dgl.graph(
            (new_src, new_dst), num_nodes=g.num_nodes(), device=src.device
        )
        pruned_g = dgl.remove_self_loop(pruned_g)
        pruned_g = dgl.add_self_loop(pruned_g)

        if "weight" in g.edata:
            kept_weight = g.edata["weight"][keep_mask]
            self_loop_weight = torch.ones(
                pruned_g.num_nodes(), device=kept_weight.device, dtype=kept_weight.dtype
            )
            pruned_g.edata["weight"] = torch.cat([kept_weight, self_loop_weight], dim=0)
        if "w" in g.edata:
            kept_w = g.edata["w"][keep_mask]
            self_loop_w = torch.ones(
                pruned_g.num_nodes(), device=kept_w.device, dtype=kept_w.dtype
            )
            pruned_g.edata["w"] = torch.cat([kept_w, self_loop_w], dim=0)
        if "ew" in g.edata:
            kept_ew = g.edata["ew"][keep_mask]
            self_loop_ew = torch.ones(
                pruned_g.num_nodes(), device=kept_ew.device, dtype=kept_ew.dtype
            )
            pruned_g.edata["ew"] = torch.cat([kept_ew, self_loop_ew], dim=0)

        pruned_edges = int((~keep_mask).sum().item())
        stats["pruned_edges"] = pruned_edges
        stats["prune_ratio"] = float(pruned_edges / max(1, total_edges))
        return pruned_g, stats

    @torch.no_grad()
    def update_semantic_graph(self, llm_emb: torch.Tensor):
        """Refresh similarity, confidence, and pruning scores from ten-column evidence."""
        self._invalidate_static_graph_cache()
        soft_labels, anchor_confidence = self.semantic_view.score_evidence(llm_emb)
        if self._sem_anchor_confidence_temperature != 1.0:
            anchor_confidence = anchor_confidence.clamp(min=1e-8).pow(
                float(self._sem_anchor_confidence_temperature)
            )
        anchor_confidence = anchor_confidence.clamp(0.0, 1.0)
        pruning_scores = self.semantic_view.build_pruning_scores(
            soft_labels, anchor_confidence
        )
        self._soft_labels = soft_labels
        self._anchor_confidence = anchor_confidence
        self._pruning_scores = pruning_scores

    @torch.no_grad()
    def initialize_static_training_graphs(self, g, device=None):
        if self._pruning_scores is None:
            raise RuntimeError(
                "Semantic evidence is not ready. Call update_semantic_graph(llm_emb) before static graph initialization."
            )

        graph_device = g.device if device is None else device
        pruned_g, prune_stats = self._get_pruned_graph(g)
        semantic_view_g = self._get_semantic_view_graph(pruned_g, graph_device)
        init_stats = dict(prune_stats)
        init_stats["semantic_view_edges"] = int(semantic_view_g.num_edges())
        init_stats["initialized_before_training"] = True
        return dict(init_stats)

    def embed_views(self, g, x):
        h_struct, h_sem, _, _ = self.encode_views(g, x)
        h_fused = self.fuse_views(h_struct, h_sem)
        return {
            "struct": h_struct,
            "sem": h_sem,
            "fused": h_fused,
        }
