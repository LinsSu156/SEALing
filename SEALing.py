"""Unified SEALing training entry point."""

from pathlib import Path

from configuration import build_args, load_best_configs


def SEALing(args=None):
    """Train configured samples and save one AnnData result per seed."""
    if args is None:
        args = build_args()
    elif getattr(args, "use_cfg", False) or getattr(args, "config", None):
        args = load_best_configs(args)
    import torch
    from utils import TBLogger, train_single_sample

    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    if args.max_epoch <= 0:
        raise ValueError("max_epoch must be positive.")
    results = []
    for seed in args.seeds:
        for sample in args.sample:
            logger = (
                TBLogger(name=f"{args.dataset}_{sample}_{seed}")
                if args.logging
                else None
            )
            try:
                score, adata, _ = train_single_sample(
                    args,
                    sample=sample,
                    seed=seed,
                    dataset=args.dataset,
                    device=device,
                    logger=logger,
                )
            finally:
                if logger is not None:
                    logger.finish()
            output_dir = Path(args.output_root) / args.dataset / sample
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"seed_{seed}.h5ad"
            adata.write_h5ad(output_path, compression="gzip")
            results.append(
                {
                    "dataset": args.dataset,
                    "sample": sample,
                    "seed": seed,
                    "score": score,
                    "path": output_path,
                }
            )
    return results


if __name__ == "__main__":
    SEALing()
