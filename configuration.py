"""Shared CLI and YAML configuration, independent of the training runtime."""

import argparse
import logging
import os
from pathlib import Path

import yaml


def _add_mainline_args(parser):
    parser.add_argument("--seeds", type=int, nargs="+", default=[46])
    parser.add_argument("--dataset", type=str, default="DLPFC")
    parser.add_argument("--data-root", dest="data_root", type=str, default=None)
    parser.add_argument(
        "--output-root", dest="output_root", type=str, default="results"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=900)
    parser.add_argument(
        "--scheduler-horizon", dest="scheduler_horizon", type=int, default=900
    )
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--monitor_interval", type=int, default=100)
    parser.add_argument("--eval_verbose", action="store_true", default=False)
    parser.add_argument("--refinement_radius", type=int, default=50)
    parser.add_argument("--preprocess_force", action="store_true", default=False)
    parser.add_argument("--preprocess_gene_num", type=int, default=0)
    parser.add_argument("--preprocess_hvg_flavor", type=str, default="seurat_v3")
    parser.add_argument("--preprocess_gene_min_cells", type=int, default=0)
    parser.add_argument("--preprocess_target_sum", type=float, default=1e4)
    parser.add_argument("--preprocess_scale_max_value", type=float, default=10.0)

    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_out_heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--num_hidden", type=int, default=1024)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--in_drop", type=float, default=0.1)
    parser.add_argument("--attn_drop", type=float, default=0.1)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--alpha_l", type=float, default=3)
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")

    parser.add_argument("--sample", type=str, nargs="+", default=["151673"])
    parser.add_argument("--n_clusters", type=int, default=0)
    parser.add_argument("--self_loop", action="store_true", default=False)

    parser.add_argument(
        "--evidence-path",
        dest="evidence_path",
        default=None,
        help="Local NPY evidence; accepts {dataset} and {sample} placeholders.",
    )
    parser.add_argument(
        "--evidence-root", dest="evidence_root", default="cache/semantic_evidence"
    )
    parser.add_argument("--llm_model_name", type=str, default="gpt-5.5")
    parser.add_argument("--sem_weak_edge_prune_rate", type=float, default=0.035)
    parser.add_argument("--sem_anchor_confidence_min", type=float, default=0.30)

    parser.add_argument("--align_loss_weight", type=float, default=0.02)
    parser.add_argument("--contrast_loss_weight", type=float, default=0.007)
    parser.add_argument("--contrast_temperature", type=float, default=0.2)
    parser.add_argument("--contrast_negative_top_k", type=int, default=32)
    parser.add_argument(
        "--contrast_semantic_negative_quantile", type=float, default=0.10
    )
    parser.add_argument("--contrast_positive_min_score", type=float, default=0.60)
    parser.add_argument("--contrast_unknown_weight", type=float, default=0.10)
    parser.add_argument("--contrast_warmup_epochs", type=int, default=50)
    parser.add_argument("--contrast_decay_start", type=int, default=300)
    parser.add_argument("--contrast_final_scale", type=float, default=0.50)
    parser.add_argument(
        "--cluster_method",
        choices=["mclust", "leiden", "louvain"],
        default="mclust",
    )
    parser.add_argument("--cluster_fused_weight", type=float, default=1.0)
    parser.add_argument("--cluster_struct_weight", type=float, default=1.0)
    parser.add_argument("--cluster_sem_weight", type=float, default=0.5)
    parser.add_argument("--cluster_fused_pca_dim", type=int, default=27)
    parser.add_argument("--cluster_struct_pca_dim", type=int, default=6)
    parser.add_argument("--cluster_sem_pca_dim", type=int, default=3)
    return parser


def build_args(argv=None):
    parser = _add_mainline_args(argparse.ArgumentParser(description="SEALing"))
    requested, _ = parser.parse_known_args(argv)
    if requested.config or requested.use_cfg:
        configured = load_best_configs(requested)
        parser.set_defaults(**vars(configured))
    return parser.parse_args(argv)


def build_args_known(the_args):
    parser = _add_mainline_args(argparse.ArgumentParser(description="SEALing"))
    args, unknown = parser.parse_known_args(the_args)
    return args


def load_best_configs(args):
    dataset_name = args.dataset
    config_path = getattr(args, "config", None) or os.path.join(
        "configs", f"{dataset_name}.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f) or {}
    if getattr(args, "config", None):
        args.dataset = configs.get("dataset", Path(config_path).stem)

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    logging.info(f"----- Using best configs from {config_path} -----")

    return args
