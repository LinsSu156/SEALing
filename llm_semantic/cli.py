"""Command-line entry point for dataset-agnostic semantic evidence generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None) -> None:
    from .profiles import (
        MAINLINE_MODEL,
        MAINLINE_TOP_K_GENES,
        get_dataset_profile,
        resolve_n_prototypes,
        resolve_spots_per_cluster,
    )

    parser = argparse.ArgumentParser(
        description="Generate semantic evidence caches for one or more samples."
    )
    parser.add_argument("--dataset", default="DLPFC")
    parser.add_argument("--root", default=None)
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--sample", default=None, help="Alias for a single sample.")
    parser.add_argument(
        "--model-name",
        "--model_name",
        default=os.environ.get("LLM_MODEL", MAINLINE_MODEL),
    )
    parser.add_argument("--api-model-id", "--api_model_id", default=None)
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai-compatible", "anthropic", "gemini", "deepseek", "openai"],
    )
    parser.add_argument("--base-url", "--base_url", default=None)
    parser.add_argument("--api-key", "--api_key", default=None)
    parser.add_argument("--n-prototypes", "--n_prototypes", default="auto")
    parser.add_argument("--n-clusters", "--n_clusters", type=int, default=None)
    parser.add_argument(
        "--spots-per-cluster", "--spots_per_cluster", type=int, default=None
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", "--max_retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if args.sample and args.samples:
        parser.error("Use either --sample or --samples")
    from .generation import (
        _load_dataset_adata,
        _print_generation_plan,
        _read_dataset_spot_count,
        generate_llm_embeddings_for_adata,
    )

    profile = get_dataset_profile(args.dataset, sample=args.sample)
    root = Path(
        args.root
        or os.environ.get(f"{args.dataset.upper()}_ROOT", profile.default_root)
    )
    samples = args.samples or ([args.sample] if args.sample else list(profile.samples))
    for sample in samples:
        if args.dry_run:
            adata = None
            n_obs = _read_dataset_spot_count(args.dataset, root, sample)
        else:
            adata = _load_dataset_adata(args.dataset, root, sample)
            n_obs = int(adata.n_obs)
        multiplier = resolve_spots_per_cluster(
            args.dataset, sample, args.spots_per_cluster
        )
        n_prototypes = resolve_n_prototypes(
            n_obs, args.n_prototypes, args.n_clusters, multiplier
        )
        _print_generation_plan(
            args.dataset, root, sample, args, n_obs, n_prototypes, multiplier
        )
        if args.dry_run:
            continue
        evidence = generate_llm_embeddings_for_adata(
            adata,
            dataset=args.dataset,
            sample=sample,
            model_name=args.model_name,
            top_k_genes=MAINLINE_TOP_K_GENES,
            n_prototypes=args.n_prototypes,
            n_clusters=args.n_clusters,
            spots_per_cluster=multiplier,
            limit=args.limit,
            api_model_id=args.api_model_id,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
        print(f"generated {args.dataset}/{sample}: shape={tuple(evidence.shape)}")


if __name__ == "__main__":
    main()
