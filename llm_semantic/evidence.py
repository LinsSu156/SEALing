"""Load local spot evidence without provider configuration or network access."""

from pathlib import Path
import re

import numpy as np

from .profiles import get_dataset_profile


DEFAULT_EVIDENCE_ROOT = (
    Path(__file__).resolve().parent.parent / "cache" / "semantic_evidence"
)
EVIDENCE_KEY = "llm_semantic_evidence"


def load_semantic_evidence(
    adata,
    dataset,
    sample,
    evidence_path=None,
    evidence_root=DEFAULT_EVIDENCE_ROOT,
    model_name="gpt-5.5",
):
    """Use an explicit file, AnnData evidence, or a unique matching cache.

    File rows must already match the observations in order. Legacy NPY caches
    have no observation identifiers, so the caller must preserve that order.
    """
    if evidence_path:
        path = Path(str(evidence_path).format(dataset=dataset, sample=sample))
        array = np.load(path, allow_pickle=False)
        source = str(path)
    elif EVIDENCE_KEY in adata.obsm:
        array = np.asarray(adata.obsm[EVIDENCE_KEY])
        source = f"adata.obsm['{EVIDENCE_KEY}']"
    else:
        profile = get_dataset_profile(dataset, sample=sample)
        cache_tag = f"{profile.cache_tag}_" if profile.cache_tag else ""
        pattern = re.compile(
            re.escape(f"{dataset}_{sample}_{model_name}_{cache_tag}")
            + r"proto\d+_k10_evidence\.npy"
        )
        root = Path(evidence_root)
        candidates = sorted(
            p for p in root.glob("*_evidence.npy") if pattern.fullmatch(p.name)
        )
        if not candidates:
            raise FileNotFoundError(
                f"No local spot evidence for {dataset}/{sample} ({model_name}) in {root}. "
                "Set --evidence-path or adata.obsm['llm_semantic_evidence']. "
                "Generate missing evidence separately with llm_semantic_generator.py; "
                "training never calls an API."
            )
        if len(candidates) > 1:
            names = "\n".join(f"  {p}" for p in candidates)
            raise ValueError(
                f"Multiple evidence files match {dataset}/{sample}; select --evidence-path:\n{names}"
            )
        path = candidates[0]
        array = np.load(path, allow_pickle=False)
        source = str(path)

    expected_shape = (adata.n_obs, 10)
    if array.shape != expected_shape:
        raise ValueError(
            f"{source}: expected evidence shape {expected_shape}, got {array.shape}."
        )
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{source}: evidence must contain finite real numbers.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{source}: evidence exceeds float32 range.")
    print(f"[SemanticView] loaded local evidence: {source} | shape={array.shape}")
    return array
