"""Public semantic-evidence API and dataset CLI."""

from llm_semantic.profiles import (
    API_SYSTEM_PROMPT as API_SYSTEM_PROMPT,
    DATASET_PROFILES as DATASET_PROFILES,
    DISC_EVIDENCE_FIELDS as DISC_EVIDENCE_FIELDS,
    DLPFC_SAMPLES as DLPFC_SAMPLES,
    DatasetSemanticProfile as DatasetSemanticProfile,
    MAINLINE_DATASET as MAINLINE_DATASET,
    MAINLINE_MODEL as MAINLINE_MODEL,
    MAINLINE_TOP_K_GENES as MAINLINE_TOP_K_GENES,
    MAINLINE_VERSION as MAINLINE_VERSION,
    REFERENCE_SAMPLE as REFERENCE_SAMPLE,
    get_dataset_profile as get_dataset_profile,
    get_dataset_spots_per_cluster as get_dataset_spots_per_cluster,
    resolve_n_prototypes as resolve_n_prototypes,
    resolve_spots_per_cluster as resolve_spots_per_cluster,
)
from llm_semantic.providers import (
    LLMProviderConfig as LLMProviderConfig,
    LLMProviderResponseError as LLMProviderResponseError,
    LLMQuotaExhaustedError as LLMQuotaExhaustedError,
    get_llm_provider_config as get_llm_provider_config,
)
from llm_semantic.generation import (
    LLM_CACHE_DIR as LLM_CACHE_DIR,
    generate_llm_embeddings_for_adata as generate_llm_embeddings_for_adata,
    generate_prototype_domain_discriminative_evidence_for_adata as generate_prototype_domain_discriminative_evidence_for_adata,
    get_llm_cache_paths as get_llm_cache_paths,
)

__all__ = [
    "API_SYSTEM_PROMPT",
    "DATASET_PROFILES",
    "DISC_EVIDENCE_FIELDS",
    "DLPFC_SAMPLES",
    "DatasetSemanticProfile",
    "MAINLINE_DATASET",
    "MAINLINE_MODEL",
    "MAINLINE_TOP_K_GENES",
    "MAINLINE_VERSION",
    "REFERENCE_SAMPLE",
    "get_dataset_profile",
    "get_dataset_spots_per_cluster",
    "resolve_n_prototypes",
    "resolve_spots_per_cluster",
    "LLMProviderConfig",
    "LLMProviderResponseError",
    "LLMQuotaExhaustedError",
    "get_llm_provider_config",
    "LLM_CACHE_DIR",
    "generate_llm_embeddings_for_adata",
    "generate_prototype_domain_discriminative_evidence_for_adata",
    "get_llm_cache_paths",
]

if __name__ == "__main__":
    from llm_semantic.cli import main

    main()
