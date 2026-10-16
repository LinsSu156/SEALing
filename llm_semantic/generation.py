"""Prototype construction, evidence generation, and resumable caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .profiles import (
    DatasetSemanticProfile,
    DISC_EVIDENCE_FIELDS,
    MAINLINE_DATASET,
    MAINLINE_MODEL,
    MAINLINE_TOP_K_GENES,
    MAINLINE_VERSION,
    REFERENCE_SAMPLE,
    _make_system_prompt,
    get_dataset_profile,
    resolve_n_prototypes,
    resolve_spots_per_cluster,
)
from .providers import (
    _call_llm_json_api,
    _resolve_provider_api_settings,
    get_llm_provider_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_CACHE_DIR = PROJECT_ROOT / "cache" / "llm_embeddings"


def get_llm_cache_paths(
    dataset: str,
    sample: str,
    n_prototypes: int,
    model_name: str = MAINLINE_MODEL,
    version: str = MAINLINE_VERSION,
    top_k_genes: int = MAINLINE_TOP_K_GENES,
    spots_per_cluster: int = None,
) -> Dict[str, Path]:
    profile = get_dataset_profile(dataset, sample=sample)
    spots_per_cluster = resolve_spots_per_cluster(
        dataset,
        sample=sample,
        spots_per_cluster=spots_per_cluster,
    )
    cache_version = f"{version}_{profile.cache_tag}" if profile.cache_tag else version
    prefix = (
        f"{dataset}_{sample}_{model_name}_{spots_per_cluster}\u00d7_"
        f"{cache_version}_proto{int(n_prototypes)}_k{int(top_k_genes)}"
    )
    return {
        "spot_evidence": LLM_CACHE_DIR / f"{prefix}_evidence.npy",
        "prototype_evidence": LLM_CACHE_DIR / f"{prefix}_prototype_evidence.npy",
        "prototype_prompts": LLM_CACHE_DIR / f"{prefix}_prototype_prompts.jsonl",
        "prototype_summary": LLM_CACHE_DIR / f"{prefix}_prototype_summary.jsonl",
    }


def _to_dense_array(adata) -> np.ndarray:
    return adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def _clip01(value, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _evidence_to_vector(
    summary: dict, evidence_fields: Tuple[str, ...] = None
) -> np.ndarray:
    fields = tuple(evidence_fields or DISC_EVIDENCE_FIELDS)
    return np.array(
        [_clip01(summary.get(field, 0.0)) for field in fields], dtype=np.float32
    )


def _adjust_summary_by_prototype_size(summary: dict, member_count: int) -> dict:
    adjusted = dict(summary)
    confidence = _clip01(adjusted.get("confidence", 0.0))
    ambiguity = _clip01(adjusted.get("ambiguity", 0.0))
    boundary = _clip01(adjusted.get("boundary_or_mixed", 0.0))

    if member_count <= 1:
        confidence = min(confidence, 0.35)
        ambiguity = max(ambiguity, 0.65)
        boundary = max(boundary, 0.45)
    elif member_count < 5:
        confidence = min(confidence, 0.55)
        ambiguity = max(ambiguity, 0.45)
        boundary = max(boundary, 0.35)
    elif member_count < 10:
        confidence = min(confidence, 0.75)
        ambiguity = max(ambiguity, 0.30)

    adjusted["confidence"] = confidence
    adjusted["ambiguity"] = ambiguity
    adjusted["boundary_or_mixed"] = boundary
    return adjusted


def _load_jsonl_by_key(path: Path, key: str) -> Dict[int, dict]:
    records: Dict[int, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key in record and "summary" in record:
                records[int(record[key])] = record
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _validate_evidence_array(
    array: np.ndarray,
    name: str,
    n_obs: int | None = None,
    evidence_fields: Tuple[str, ...] = None,
) -> np.ndarray:
    fields = tuple(evidence_fields or DISC_EVIDENCE_FIELDS)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(fields):
        raise ValueError(
            f"{name} must have shape [N, {len(fields)}], got {array.shape}"
        )
    if n_obs is not None and array.shape[0] != int(n_obs):
        raise ValueError(
            f"{name} row count {array.shape[0]} does not match adata.n_obs={n_obs}"
        )
    return array


def _load_spot_evidence_cache(
    path: Path,
    adata=None,
    evidence_fields: Tuple[str, ...] = None,
) -> torch.Tensor | None:
    if not path.exists():
        return None
    n_obs = getattr(adata, "n_obs", None) if adata is not None else None
    evidence = _validate_evidence_array(
        np.load(path), path.name, n_obs=n_obs, evidence_fields=evidence_fields
    )
    print(f"[LLM-Prototype] load fixed spot evidence cache: {path}")
    return torch.tensor(evidence, dtype=torch.float32)


def _format_gene_items(
    gene_names: np.ndarray,
    values: np.ndarray,
    indices: Iterable[int],
    value_name: str,
    max_items: int,
) -> str:
    items = []
    for idx in list(indices)[:max_items]:
        items.append(f"- {gene_names[idx]}: {value_name}={float(values[idx]):.4f}")
    return "\n".join(items) if items else "- none"


def _format_gene_list(
    gene_names: np.ndarray, indices: Iterable[int], max_items: int
) -> str:
    items = [f"- {gene_names[idx]}" for idx in list(indices)[:max_items]]
    return "\n".join(items) if items else "- none"


def _preprocess_expr_for_semantic_profiles(
    adata,
    highly_genes: int = 3000,
    target_sum: float = 10000.0,
    max_value: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    import scanpy as sc

    work = adata.copy()
    if hasattr(work, "var_names_make_unique"):
        work.var_names_make_unique()
    sc.pp.filter_genes(work, min_cells=100)
    if work.n_vars > highly_genes:
        try:
            sc.pp.highly_variable_genes(
                work, flavor="seurat_v3", n_top_genes=highly_genes
            )
        except Exception:
            sc.pp.highly_variable_genes(work, flavor="seurat", n_top_genes=highly_genes)
        work = work[:, work.var["highly_variable"]].copy()

    expr = _to_dense_array(work).astype(np.float32)
    row_sum = expr.sum(axis=1, keepdims=True)
    expr = expr / np.maximum(row_sum, 1e-8) * target_sum
    mean = expr.mean(axis=0, keepdims=True)
    std = expr.std(axis=0, keepdims=True) + 1e-6
    expr = np.clip((expr - mean) / std, a_min=None, a_max=max_value)
    expr = np.nan_to_num(expr, nan=0.0, posinf=max_value, neginf=0.0)
    return expr.astype(np.float32), work.var_names.to_numpy()


def _semantic_feature_matrix(
    adata,
    semantic_preprocess: bool = True,
    highly_genes: int = 3000,
) -> Tuple[np.ndarray, np.ndarray]:
    if semantic_preprocess:
        return _preprocess_expr_for_semantic_profiles(adata, highly_genes=highly_genes)
    expr = _safe_log1p(_to_dense_array(adata).astype(np.float32))
    return expr, adata.var_names.to_numpy()


def _build_prototype_assignments(
    expr_profile: np.ndarray,
    adata,
    n_prototypes: int = 160,
    spatial_weight: float = 0.25,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler, normalize

    n_prototypes = min(int(n_prototypes), expr_profile.shape[0])
    expr_z = StandardScaler(with_mean=True, with_std=True).fit_transform(expr_profile)
    n_components = min(50, expr_z.shape[0] - 1, expr_z.shape[1] - 1)
    if n_components >= 2:
        expr_feat = TruncatedSVD(
            n_components=n_components, random_state=random_state
        ).fit_transform(expr_z)
    else:
        expr_feat = expr_z
    expr_feat = normalize(expr_feat)

    if "spatial" in adata.obsm and spatial_weight > 0:
        coords = StandardScaler().fit_transform(
            np.asarray(adata.obsm["spatial"], dtype=np.float32)
        )
        features = np.hstack([expr_feat, coords * float(spatial_weight)])
    else:
        features = expr_feat

    kmeans = MiniBatchKMeans(
        n_clusters=n_prototypes,
        random_state=random_state,
        n_init=20,
        batch_size=min(1024, max(128, expr_profile.shape[0])),
    )
    assignments = kmeans.fit_predict(features)
    return assignments.astype(np.int64), features.astype(np.float32)


def _build_prototype_neighbors(
    assignments: np.ndarray,
    features: np.ndarray,
    max_neighbors: int = 6,
) -> list[np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    n_proto = int(assignments.max()) + 1
    centers = np.zeros((n_proto, features.shape[1]), dtype=np.float32)
    for proto_id in range(n_proto):
        members = np.where(assignments == proto_id)[0]
        if len(members) > 0:
            centers[proto_id] = features[members].mean(axis=0)

    k = min(max_neighbors + 1, n_proto)
    if k <= 1:
        return [np.array([], dtype=np.int64) for _ in range(n_proto)]

    indices = (
        NearestNeighbors(n_neighbors=k)
        .fit(centers)
        .kneighbors(centers, return_distance=False)
    )
    return [
        row[row != proto_id][:max_neighbors] for proto_id, row in enumerate(indices)
    ]


def _build_prototype_prompt(
    prototype_id: int,
    member_indices: np.ndarray,
    assignments: np.ndarray,
    prototype_neighbors: np.ndarray,
    gene_names: np.ndarray,
    expr_profile: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    top_k_genes: int,
    profile: DatasetSemanticProfile,
) -> str:
    proto_mean = expr_profile[member_indices].mean(axis=0)
    z = (proto_mean - global_mean) / global_std
    top_expr_idx = np.argsort(proto_mean)[::-1][:top_k_genes]
    top_global_enriched_idx = np.argsort(z)[::-1][:top_k_genes]

    neighbor_members = []
    for neighbor_proto in prototype_neighbors:
        neighbor_members.extend(
            np.where(assignments == int(neighbor_proto))[0].tolist()
        )

    if neighbor_members:
        neigh_mean = expr_profile[np.array(neighbor_members, dtype=np.int64)].mean(
            axis=0
        )
        local_delta = proto_mean - neigh_mean
        top_neighbor_idx = np.argsort(neigh_mean)[::-1][:top_k_genes]
        top_local_enriched_idx = np.argsort(local_delta)[::-1][:top_k_genes]
        top_local_depleted_idx = np.argsort(local_delta)[:top_k_genes]
        shared = [idx for idx in top_expr_idx if idx in set(top_neighbor_idx)]
    else:
        neigh_mean = np.zeros_like(proto_mean)
        local_delta = proto_mean.copy()
        top_neighbor_idx = np.array([], dtype=np.int64)
        top_local_enriched_idx = top_global_enriched_idx
        top_local_depleted_idx = np.argsort(local_delta)[:top_k_genes]
        shared = []

    member_profiles = expr_profile[member_indices]
    proto_norm = np.linalg.norm(proto_mean) + 1e-8
    member_norms = np.linalg.norm(member_profiles, axis=1) + 1e-8
    member_cos = (member_profiles @ proto_mean) / (member_norms * proto_norm)
    within_stability = float(np.mean(member_cos))
    within_heterogeneity = float(np.std(member_cos))
    shared_fraction = float(len(shared) / max(1, len(top_expr_idx)))
    rules = "\n".join(f"- {rule}" for rule in profile.interpretation_rules)
    evidence_axes = "\n".join(f"- {field}" for field in profile.evidence_fields)

    return f"""Dataset context:
{profile.context}
The semantic evidence vector will be used by downstream multi-view learning: semantic-view graph construction, semantic pruning, and structure-semantic dual-confidence contrastive learning.

Task objective:
{profile.objective}

Evidence axes to score:
{evidence_axes}

Interpretation rules:
{rules}

Prototype id: {prototype_id}
Member spot count: {len(member_indices)}
Neighbor prototype ids: {list(map(int, prototype_neighbors))}

Prototype top expressed genes after semantic preprocessing:
{_format_gene_items(gene_names, proto_mean, top_expr_idx, value_name="proto_expr", max_items=top_k_genes)}

Prototype enriched genes versus dataset background:
{_format_gene_items(gene_names, z, top_global_enriched_idx, value_name="z", max_items=top_k_genes)}

Neighbor prototype consensus top genes:
{_format_gene_items(gene_names, neigh_mean, top_neighbor_idx, value_name="neighbor_expr", max_items=top_k_genes)}

Prototype-enriched genes compared with neighbor prototypes:
{_format_gene_items(gene_names, local_delta, top_local_enriched_idx, value_name="delta_expr", max_items=top_k_genes)}

Prototype-depleted genes compared with neighbor prototypes:
{_format_gene_items(gene_names, -local_delta, top_local_depleted_idx, value_name="depletion", max_items=top_k_genes)}

Genes shared by prototype and neighbor top lists:
{_format_gene_list(gene_names, shared, max_items=top_k_genes)}

Prototype stability:
- within_prototype_cosine_to_center: {within_stability:.4f}
- within_prototype_heterogeneity: {within_heterogeneity:.4f}
- shared_top_gene_fraction_with_neighbor_prototypes: {shared_fraction:.4f}

Return strict JSON only. Calibrate all numeric evidence scores to [0.0, 1.0]. Do not output a hard label."""


def _prepare_assignments_and_profiles(
    adata,
    n_prototypes: int,
    semantic_preprocess: bool,
    highly_genes: int,
    spatial_weight: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expr_profile, gene_names = _semantic_feature_matrix(
        adata,
        semantic_preprocess=semantic_preprocess,
        highly_genes=highly_genes,
    )
    assignments, features = _build_prototype_assignments(
        expr_profile=expr_profile,
        adata=adata,
        n_prototypes=n_prototypes,
        spatial_weight=spatial_weight,
        random_state=random_state,
    )
    return expr_profile, gene_names, assignments, features


def _write_spot_evidence_from_prototypes(
    prototype_evidence: np.ndarray,
    assignments: np.ndarray,
    path: Path,
    evidence_fields: Tuple[str, ...] = None,
) -> np.ndarray:
    prototype_evidence = _validate_evidence_array(
        prototype_evidence,
        "prototype_evidence",
        evidence_fields=evidence_fields,
    )
    if assignments.max(initial=0) >= prototype_evidence.shape[0]:
        raise ValueError(
            "Prototype assignment id exceeds prototype_evidence rows. "
            f"max_assignment={int(assignments.max())}, rows={prototype_evidence.shape[0]}"
        )
    spot_evidence = prototype_evidence[assignments]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, spot_evidence.astype(np.float32))
    print(
        f"[LLM-Prototype] fixed spot evidence saved: {path} | shape={spot_evidence.shape}"
    )
    return spot_evidence.astype(np.float32)


def generate_prototype_domain_discriminative_evidence_for_adata(
    adata,
    dataset: str = MAINLINE_DATASET,
    sample: str = REFERENCE_SAMPLE,
    model_name: str = MAINLINE_MODEL,
    n_prototypes=None,
    n_clusters: int = None,
    spots_per_cluster: int = None,
    top_k_genes: int = MAINLINE_TOP_K_GENES,
    max_neighbors: int = 6,
    spatial_weight: float = 0.25,
    semantic_preprocess: bool = True,
    highly_genes: int = 3000,
    version: str = MAINLINE_VERSION,
    api_key: str = None,
    api_model_id: str = None,
    provider: str = None,
    base_url: str = None,
    max_retries: int = 3,
    timeout: int = 120,
    random_state: int = 0,
    limit: int = None,
) -> torch.Tensor:
    """Generate or load the fixed LLM prototype evidence used by training."""
    if top_k_genes != MAINLINE_TOP_K_GENES or version != MAINLINE_VERSION:
        raise ValueError(
            "This generator requires top_k_genes=10 and its current cache version"
        )
    provider_config = get_llm_provider_config(
        model_name=model_name,
        api_model_id=api_model_id,
        provider=provider,
        base_url=base_url,
    )
    profile = get_dataset_profile(dataset, sample=sample)
    spots_per_cluster = resolve_spots_per_cluster(
        dataset,
        sample=sample,
        spots_per_cluster=spots_per_cluster,
    )
    evidence_fields = profile.evidence_fields
    system_prompt = _make_system_prompt(evidence_fields)
    if n_prototypes is None or str(n_prototypes).lower() in {"", "auto", "none"}:
        if adata is None or not hasattr(adata, "n_obs"):
            raise ValueError("adata.n_obs is required when n_prototypes is auto.")
        resolved_n_prototypes = resolve_n_prototypes(
            adata.n_obs,
            n_prototypes=None,
            n_clusters=n_clusters,
            spots_per_cluster=spots_per_cluster,
        )
    else:
        n_obs = getattr(adata, "n_obs", int(n_prototypes))
        resolved_n_prototypes = resolve_n_prototypes(n_obs, n_prototypes=n_prototypes)

    paths = get_llm_cache_paths(
        dataset=dataset,
        sample=sample,
        model_name=model_name,
        version=version,
        n_prototypes=resolved_n_prototypes,
        top_k_genes=top_k_genes,
        spots_per_cluster=spots_per_cluster,
    )

    cached = _load_spot_evidence_cache(
        paths["spot_evidence"], adata=adata, evidence_fields=evidence_fields
    )
    if cached is not None:
        return cached

    if adata is None:
        raise ValueError(
            "adata is required when the fixed spot evidence cache is missing."
        )

    expr_profile, gene_names, assignments, features = _prepare_assignments_and_profiles(
        adata=adata,
        n_prototypes=resolved_n_prototypes,
        semantic_preprocess=semantic_preprocess,
        highly_genes=highly_genes,
        spatial_weight=spatial_weight,
        random_state=random_state,
    )

    if paths["prototype_evidence"].exists():
        prototype_evidence = _validate_evidence_array(
            np.load(paths["prototype_evidence"]),
            paths["prototype_evidence"].name,
            evidence_fields=evidence_fields,
        )
        spot_evidence = _write_spot_evidence_from_prototypes(
            prototype_evidence=prototype_evidence,
            assignments=assignments,
            path=paths["spot_evidence"],
            evidence_fields=evidence_fields,
        )
        return torch.tensor(spot_evidence, dtype=torch.float32)

    actual_n_prototypes = int(assignments.max()) + 1
    prototype_neighbors = _build_prototype_neighbors(
        assignments=assignments,
        features=features,
        max_neighbors=max_neighbors,
    )
    global_mean = expr_profile.mean(axis=0)
    global_std = expr_profile.std(axis=0) + 1e-6

    existing = _load_jsonl_by_key(paths["prototype_summary"], key="prototype_id")
    prototype_evidence = np.full(
        (actual_n_prototypes, len(evidence_fields)),
        np.nan,
        dtype=np.float32,
    )
    for prototype_id, record in existing.items():
        if 0 <= prototype_id < actual_n_prototypes:
            member_count = int(
                record.get("member_count", np.sum(assignments == prototype_id))
            )
            summary = _adjust_summary_by_prototype_size(record["summary"], member_count)
            prototype_evidence[prototype_id] = _evidence_to_vector(
                summary, evidence_fields=evidence_fields
            )

    missing_before_api = np.isnan(prototype_evidence).any(axis=1)
    if missing_before_api.any():
        api_key, resolved_base_url = _resolve_provider_api_settings(
            provider_config,
            api_key=api_key,
            base_url=base_url,
        )
        if not api_key:
            raise RuntimeError(
                f"API key is required to generate missing {model_name} prototype evidence. "
                "Set the provider API key via the corresponding environment variable or --api-key."
            )

        print(
            "[LLM-Prototype] generating fixed prototype evidence | "
            f"dataset={dataset} | model={model_name} | provider={provider_config.provider} | "
            f"api_model={provider_config.api_model_id} | prototypes={actual_n_prototypes} | spots={adata.n_obs}"
        )
        print(f"[LLM-Prototype] prompts cache: {paths['prototype_prompts']}")
        print(f"[LLM-Prototype] summary cache: {paths['prototype_summary']}")

        n_to_process = (
            actual_n_prototypes
            if limit is None
            else min(actual_n_prototypes, int(limit))
        )
        for prototype_id in tqdm(
            range(n_to_process), desc=f"{model_name} prototype evidence", ascii=True
        ):
            if not np.isnan(prototype_evidence[prototype_id]).any():
                continue

            member_indices = np.where(assignments == prototype_id)[0]
            if len(member_indices) == 0:
                prototype_evidence[prototype_id] = np.zeros(
                    len(evidence_fields), dtype=np.float32
                )
                continue

            prompt = _build_prototype_prompt(
                prototype_id=prototype_id,
                member_indices=member_indices,
                assignments=assignments,
                prototype_neighbors=prototype_neighbors[prototype_id],
                gene_names=gene_names,
                expr_profile=expr_profile,
                global_mean=global_mean,
                global_std=global_std,
                top_k_genes=top_k_genes,
                profile=profile,
            )
            summary, raw_content = _call_llm_json_api(
                user_prompt=prompt,
                api_key=api_key,
                base_url=resolved_base_url,
                model_name=model_name,
                api_model_id=provider_config.api_model_id,
                provider=provider_config.provider,
                max_retries=max_retries,
                timeout=timeout,
                system_prompt=system_prompt,
            )
            summary = _adjust_summary_by_prototype_size(
                summary, int(len(member_indices))
            )
            prototype_evidence[prototype_id] = _evidence_to_vector(
                summary, evidence_fields=evidence_fields
            )
            _append_jsonl(
                paths["prototype_prompts"],
                {"prototype_id": prototype_id, "prompt": prompt},
            )
            _append_jsonl(
                paths["prototype_summary"],
                {
                    "prototype_id": prototype_id,
                    "member_count": int(len(member_indices)),
                    "summary": summary,
                    "raw_content": raw_content,
                },
            )

    missing = np.isnan(prototype_evidence).any(axis=1)
    if missing.any():
        missing_count = int(missing.sum())
        if limit is not None:
            print(
                "[LLM-Prototype] partial run finished: "
                f"{actual_n_prototypes - missing_count}/{actual_n_prototypes} prototypes cached. "
                "Re-run without limit to write the final npy caches."
            )
            partial = np.nan_to_num(prototype_evidence[assignments], nan=0.0)
            return torch.tensor(partial, dtype=torch.float32)
        raise RuntimeError(
            f"Prototype evidence generation incomplete: {missing_count} prototypes missing. "
            f"Re-run to resume from {paths['prototype_summary']}."
        )

    paths["prototype_evidence"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["prototype_evidence"], prototype_evidence.astype(np.float32))
    print(
        f"[LLM-Prototype] fixed prototype evidence saved: {paths['prototype_evidence']} "
        f"| shape={prototype_evidence.shape}"
    )

    spot_evidence = _write_spot_evidence_from_prototypes(
        prototype_evidence=prototype_evidence,
        assignments=assignments,
        path=paths["spot_evidence"],
        evidence_fields=evidence_fields,
    )
    return torch.tensor(spot_evidence, dtype=torch.float32)


generate_llm_embeddings_for_adata = (
    generate_prototype_domain_discriminative_evidence_for_adata
)


def _load_dataset_adata(dataset: str, root: Path, sample: str):
    import scanpy as sc

    if dataset == "DLPFC":
        adata = sc.read_visium(
            str(root / sample),
            count_file="filtered_feature_bc_matrix.h5",
            load_images=False,
        )
    elif dataset in {"HBC", "MBA", "MBP"}:
        adata = sc.read_visium(
            str(root), count_file="filtered_feature_bc_matrix.h5", load_images=False
        )
    elif dataset in {"MBM_s1", "MBM_s2"}:
        adata = sc.read_h5ad(str(root / "filtered_feature_bc_matrix.h5ad"))
    elif dataset == "Simulated_STARmap":
        adata = sc.read_h5ad(str(root / f"{sample}.h5ad"))
    else:
        h5ad_path = root / f"{sample}.h5ad"
        if h5ad_path.exists():
            adata = sc.read_h5ad(str(h5ad_path))
        else:
            adata = sc.read_visium(
                str(root / sample),
                count_file="filtered_feature_bc_matrix.h5",
                load_images=False,
            )
    adata.var_names_make_unique()
    return adata


def _read_h5ad_obs_count(path: Path) -> int:
    import h5py

    with h5py.File(path, "r") as f:
        obs = f["obs"]
        index_key = obs.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        index_key = str(index_key)
        if index_key not in obs:
            raise KeyError(
                f"AnnData observation index {index_key!r} is missing from {path}"
            )
        return int(obs[index_key].shape[0])


def _read_dataset_spot_count(dataset: str, root: Path, sample: str) -> int:
    import h5py

    if dataset == "DLPFC":
        h5_path = root / sample / "filtered_feature_bc_matrix.h5"
        with h5py.File(h5_path, "r") as f:
            return int(f["matrix"]["shape"][:][1])
    if dataset in {"HBC", "MBA", "MBP"}:
        with h5py.File(root / "filtered_feature_bc_matrix.h5", "r") as f:
            return int(f["matrix"]["shape"][:][1])
    if dataset in {"MBM_s1", "MBM_s2"}:
        return _read_h5ad_obs_count(root / "filtered_feature_bc_matrix.h5ad")
    if dataset == "Simulated_STARmap":
        return _read_h5ad_obs_count(root / f"{sample}.h5ad")

    h5ad_path = root / f"{sample}.h5ad"
    if h5ad_path.exists():
        return _read_h5ad_obs_count(h5ad_path)
    with h5py.File(root / sample / "filtered_feature_bc_matrix.h5", "r") as f:
        return int(f["matrix"]["shape"][:][1])


def _print_generation_plan(
    dataset: str,
    root: Path,
    sample: str,
    args,
    n_obs: int,
    n_prototypes: int,
    spots_per_cluster: int,
) -> None:
    provider_config = get_llm_provider_config(
        args.model_name,
        api_model_id=args.api_model_id,
        provider=args.provider,
        base_url=args.base_url,
    )
    paths = get_llm_cache_paths(
        dataset=dataset,
        sample=sample,
        model_name=args.model_name,
        n_prototypes=n_prototypes,
        top_k_genes=MAINLINE_TOP_K_GENES,
        spots_per_cluster=spots_per_cluster,
    )
    print(
        "[LLM-Prototype] dataset={} sample={} spots={} clusters={} spots_per_cluster={} prototypes={} model={} provider={} api_model={}".format(
            dataset,
            sample,
            n_obs,
            args.n_clusters,
            spots_per_cluster,
            n_prototypes,
            args.model_name,
            provider_config.provider,
            provider_config.api_model_id,
        )
    )
    print(f"[LLM-Prototype] root: {root}")
    print(f"[LLM-Prototype] spot evidence: {paths['spot_evidence']}")
    print(f"[LLM-Prototype] prototype evidence: {paths['prototype_evidence']}")
    print(f"[LLM-Prototype] dry_run={args.dry_run}")
