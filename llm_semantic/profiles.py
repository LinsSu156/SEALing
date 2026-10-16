"""Biological evidence schemas and prototype budgets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

MAINLINE_DATASET = "DLPFC"


MAINLINE_MODEL = "gpt-5.5"


MAINLINE_VERSION = "proto_disc_v2_soft_codex"


MAINLINE_TOP_K_GENES = 10


REFERENCE_SAMPLE = "151672"
REFERENCE_SPOTS = 3639
REFERENCE_PROTOTYPES = 160
MIN_PROTOTYPES = 96
MAX_PROTOTYPES = 240


DLPFC_SAMPLES = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]


@dataclass(frozen=True)
class DatasetSemanticProfile:
    dataset: str
    default_sample: str
    default_root: str
    samples: Tuple[str, ...]
    evidence_fields: Tuple[str, ...]
    context: str
    interpretation_rules: Tuple[str, ...]
    spots_per_cluster: int = 10
    cache_tag: str = ""
    objective: str = (
        "Extract continuous biological evidence axes, not a hard domain label. "
        "The output should help decide whether two prototypes/spots are semantically similar, "
        "whether a semantic edge should be trusted, and whether a prototype is a reliable "
        "contrastive anchor."
    )


def _make_system_prompt(evidence_fields: Tuple[str, ...]) -> str:
    score_lines = ",\n".join(f'  "{field}": 0.0' for field in evidence_fields)
    return f"""You are a spatial transcriptomics semantic-evidence annotator.
Your output is used as a semantic view for graph pruning and structure-semantic dual-confidence contrastive learning, not as a classifier.
Do not predict a hard spatial-domain label. Do not force a layer assignment. Extract continuous, calibrated biological evidence useful for semantic similarity, semantic graph construction, anchor selection, and uncertainty-aware contrastive learning.
Use only the provided gene expression evidence.
Return scores from 0.0 to 1.0. High confidence means the evidence vector is reliable for semantic graph edges and contrastive anchors. High ambiguity means the prototype/spot should be down-weighted or treated as boundary/mixed in semantic pruning.
Return strict JSON only, with exactly these keys:
{{
  "semantic_programs": ["short continuous evidence descriptors, not labels"],
{score_lines},
  "key_positive_markers": ["GENE"],
  "key_negative_markers": ["GENE"],
  "short_signature": "one concise sentence describing continuous semantic evidence, not a label"
}}"""


DATASET_PROFILES: Dict[str, DatasetSemanticProfile] = {
    "DLPFC": DatasetSemanticProfile(
        dataset="DLPFC",
        default_sample=REFERENCE_SAMPLE,
        default_root="data/DLPFC",
        samples=tuple(DLPFC_SAMPLES),
        evidence_fields=(
            "neuronal_synaptic",
            "oligodendrocyte_myelin",
            "upper_layer_cortical",
            "middle_layer_cortical",
            "deep_layer_cortical",
            "white_matter",
            "glial_vascular",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Human DLPFC spatial transcriptomics. This profile summarizes one "
            "expression-spatial prototype, not an individual spot."
        ),
        interpretation_rules=(
            "Do not classify this prototype into Layer1-Layer6 or white matter.",
            "Do not output a single dominant hard label.",
            "Separate white-matter/myelin evidence from cortical neuronal evidence, but keep both as continuous scores.",
            "For cortical prototypes, assign graded upper/middle/deep cortical evidence when layer-informative markers are present; do not mark all cortical prototypes as mixed just because exact layer identity is uncertain.",
            "High ambiguity should mean genuinely mixed, boundary-like, low-specificity, or internally heterogeneous evidence.",
            "High confidence should mean the continuous evidence vector is suitable for semantic graph edges and contrastive anchors.",
            "Boundary/mixed and ambiguity scores are useful uncertainty signals, not failures.",
        ),
        spots_per_cluster=20,
    ),
    "HBC": DatasetSemanticProfile(
        dataset="HBC",
        default_sample="HBC",
        default_root="data/HBC",
        samples=("HBC",),
        evidence_fields=(
            "epithelial_tumor",
            "stromal_fibroblast",
            "immune_lymphoid",
            "myeloid_inflammatory",
            "vascular_endothelial",
            "proliferation_stress",
            "adipose_normal_mammary",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Human breast cancer spatial transcriptomics. This profile summarizes one "
            "expression-spatial prototype from tumor tissue, not an individual spot."
        ),
        interpretation_rules=(
            "Do not output a hard tumor region label or a single cancer subtype.",
            "Represent tumor epithelial, stromal, immune, myeloid, vascular, proliferative/stress, and adipose or normal mammary programs as continuous evidence axes.",
            "Treat mixed tumor-stroma, tumor-immune, necrotic/stress-like, or low-specificity prototypes as higher boundary_or_mixed and ambiguity.",
            "High confidence should mean the marker evidence is internally coherent and suitable for semantic graph edges and contrastive anchors.",
            "Use marker evidence only; do not infer clinical attributes that are not present in the expression profile.",
        ),
        spots_per_cluster=20,
    ),
    "MBA": DatasetSemanticProfile(
        dataset="MBA",
        default_sample="MBA",
        default_root="data/MBA",
        samples=("MBA",),
        evidence_fields=(
            "neuronal_synaptic",
            "excitatory_cortical",
            "inhibitory_interneuron",
            "oligodendrocyte_myelin",
            "astrocyte_glial",
            "vascular_meningeal",
            "ependymal_choroid_plexus",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Mouse anterior brain spatial transcriptomics. This profile summarizes one "
            "expression-spatial prototype from brain tissue, not an individual spot."
        ),
        interpretation_rules=(
            "Do not assign a hard anatomical region label.",
            "Represent neuronal, excitatory, inhibitory, oligodendrocyte/myelin, astrocytic/glial, vascular/meningeal, and ependymal/choroid-like programs as continuous evidence axes.",
            "Separate cell-state evidence from anatomical certainty; uncertain or transition-like prototypes should receive higher ambiguity.",
            "High confidence should mean the expression program is coherent enough for semantic graph edges and contrastive anchors.",
            "Use the provided genes and neighborhood context only; avoid unsupported anatomical claims.",
        ),
        spots_per_cluster=5,
    ),
    "MBP": DatasetSemanticProfile(
        dataset="MBP",
        default_sample="MBP",
        default_root="data/MBP",
        samples=("MBP",),
        evidence_fields=(
            "neuronal_synaptic",
            "excitatory_cortical_hippocampal",
            "inhibitory_interneuron",
            "posterior_subcortical_midbrain",
            "oligodendrocyte_myelin_white_matter",
            "astrocyte_glial",
            "immune_vascular_meningeal_choroid",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Mouse brain serial section 1 sagittal-posterior Visium spatial transcriptomics. "
            "This profile summarizes one expression-spatial prototype from posterior brain tissue, "
            "not an individual spot."
        ),
        interpretation_rules=(
            "No manual ground-truth labels are present in the local AnnData; optimize evidence for unsupervised semantic graph reliability and SC/DB-guided clustering rather than ARI/NMI label recovery.",
            "Do not assign a hard anatomical atlas region, sagittal layer, or cluster label.",
            "Represent posterior-brain neuronal, excitatory cortical/hippocampal, inhibitory, subcortical/midbrain, myelin/white-matter, astrocytic, immune, vascular, meningeal, and choroid-like programs as continuous evidence axes.",
            "Separate coherent cell-state evidence from anatomical certainty; marker-sparse or transition-like prototypes should receive higher boundary_or_mixed and ambiguity.",
            "High confidence should mean internally coherent marker evidence suitable for semantic edges, compact unsupervised clusters, and reliable contrastive anchors.",
            "Use only the provided genes and neighborhood context; avoid unsupported posterior-region claims when marker evidence is weak.",
        ),
        objective=(
            "Extract continuous biological evidence axes for unsupervised posterior-brain structure discovery, "
            "not a hard domain label. The output should help build semantically coherent neighborhoods, "
            "down-weight mixed or uncertain prototypes, improve cluster compactness/separation for SC/DB evaluation, "
            "and select reliable contrastive anchors without relying on ground-truth labels."
        ),
    ),
    "MBM_s1": DatasetSemanticProfile(
        dataset="MBM_s1",
        default_sample="MBM_s1",
        default_root="data/MBM_s1",
        samples=("MBM_s1",),
        evidence_fields=(
            "neuronal_synaptic",
            "cortical_excitatory_laminar",
            "inhibitory_interneuron",
            "hippocampal_striatal_thalamic_hypothalamic",
            "posterior_midbrain_hindbrain_cerebellar",
            "oligodendrocyte_myelin_white_matter",
            "astrocyte_immune_vascular_choroid",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Aligned and jointly analyzed 10x Visium mouse brain serial Section 1 sagittal-anterior "
            "and sagittal-posterior spatial transcriptomics. The local aligned AnnData contains "
            "6,050 spots: 2,695 anterior-section spots and 3,355 posterior-section spots from "
            "fresh-frozen brain of a male C57BL/6 mouse older than 8 weeks. The source sections "
            "were 10 micrometers thick and processed with Space Ranger 1.1.0. This profile "
            "summarizes one expression-spatial prototype, not an individual spot."
        ),
        interpretation_rules=(
            "No manual ground-truth domain labels are present in the local AnnData; optimize continuous evidence for unsupervised semantic-graph reliability and SC/DB-guided clustering.",
            "The anterior and posterior tissues were aligned and jointly analyzed, but do not predict an anterior/posterior assignment.",
            "Do not assign a named anatomical atlas region, a hard spatial-domain label, or any of the 26 cluster identifiers.",
            "Increase an anatomical-program score only when the supplied marker genes and neighbor-prototype evidence support that program.",
            "Keep neuronal/synaptic, cortical excitatory/laminar, inhibitory, hippocampal-striatal-thalamic-hypothalamic, posterior midbrain-hindbrain-cerebellar, myelin/white-matter, and astrocyte-immune-vascular-choroid evidence as continuous axes.",
            "Conflicting neuronal and glial programs, spatial transitions, marker-sparse prototypes, or internally heterogeneous evidence should increase boundary_or_mixed and ambiguity.",
            "High confidence requires coherent marker and neighborhood evidence suitable for semantic graph edges and contrastive anchors.",
            "Use only the supplied expression and neighbor-prototype evidence; anatomical expectations are context, not labels or evidence by themselves.",
        ),
        cache_tag="mbm_joint_section1_v1",
        objective=(
            "Extract ten continuous biological evidence axes for unsupervised discovery in aligned "
            "mouse brain Section 1 anterior-posterior tissue, not a hard anatomical or cluster label. "
            "The evidence should strengthen biologically coherent semantic neighborhoods, down-weight "
            "mixed or uncertain prototypes, and support compact SC/DB-selected clusters."
        ),
    ),
    "MBM_s2": DatasetSemanticProfile(
        dataset="MBM_s2",
        default_sample="MBM_s2",
        default_root="data/MBM_s2",
        samples=("MBM_s2",),
        evidence_fields=(
            "neuronal_synaptic",
            "cortical_excitatory_laminar",
            "inhibitory_interneuron",
            "hippocampal_striatal_thalamic_hypothalamic",
            "posterior_midbrain_hindbrain_cerebellar",
            "oligodendrocyte_myelin_white_matter",
            "astrocyte_immune_vascular_choroid",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Aligned and jointly analyzed 10x Visium mouse brain serial Section 2 sagittal-anterior "
            "and sagittal-posterior spatial transcriptomics. The public aggregate reports 6,112 "
            "under-tissue spots (2,823 anterior and 3,289 posterior), while the local preprocessed "
            "and aligned AnnData intentionally preserves 6,114 observations. The fresh-frozen source "
            "brain was from a male C57BL/6 mouse older than 8 weeks; sections were 10 micrometers "
            "thick and processed with Space Ranger 2.0.0. This profile summarizes one "
            "expression-spatial prototype, not an individual spot."
        ),
        interpretation_rules=(
            "No manual ground-truth domain labels are present in the local AnnData; optimize continuous evidence for unsupervised semantic-graph reliability and SC/DB-guided clustering.",
            "The anterior and posterior tissues were aligned and jointly analyzed, but do not predict an anterior/posterior assignment.",
            "Do not assign a named anatomical atlas region, a hard spatial-domain label, or any of the 26 cluster identifiers.",
            "Increase an anatomical-program score only when the supplied marker genes and neighbor-prototype evidence support that program.",
            "Keep neuronal/synaptic, cortical excitatory/laminar, inhibitory, hippocampal-striatal-thalamic-hypothalamic, posterior midbrain-hindbrain-cerebellar, myelin/white-matter, and astrocyte-immune-vascular-choroid evidence as continuous axes.",
            "Conflicting neuronal and glial programs, spatial transitions, marker-sparse prototypes, or internally heterogeneous evidence should increase boundary_or_mixed and ambiguity.",
            "High confidence requires coherent marker and neighborhood evidence suitable for semantic graph edges and contrastive anchors.",
            "Use only the supplied expression and neighbor-prototype evidence; anatomical expectations are context, not labels or evidence by themselves.",
            "Treat the two additional local observations as part of the local preprocessing/alignment version; do not infer special biology from their presence.",
        ),
        cache_tag="mbm_joint_section2_v1",
        objective=(
            "Extract ten continuous biological evidence axes for unsupervised discovery in aligned "
            "mouse brain Section 2 anterior-posterior tissue, not a hard anatomical or cluster label. "
            "The evidence should strengthen biologically coherent semantic neighborhoods, down-weight "
            "mixed or uncertain prototypes, and support compact SC/DB-selected clusters."
        ),
    ),
    "Simulated_STARmap": DatasetSemanticProfile(
        dataset="Simulated_STARmap",
        default_sample="STARmap_20180505_BY3_1k",
        default_root="data/Simulated_STARmap",
        samples=("STARmap_20180505_BY3_1k",),
        evidence_fields=(
            "neuronal_synaptic",
            "excitatory_neuron",
            "inhibitory_neuron",
            "astrocyte_glial",
            "oligodendrocyte_myelin",
            "microglia_immune",
            "vascular_endothelial",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Simulated STARmap spatial transcriptomics. This profile summarizes one "
            "expression-spatial prototype from spatially resolved brain-like data, "
            "not an individual cell or spot."
        ),
        interpretation_rules=(
            "Do not output a hard simulated cluster label.",
            "Represent neuronal, excitatory, inhibitory, astrocytic/glial, oligodendrocyte/myelin, microglial/immune, and vascular programs as continuous evidence axes.",
            "Treat mixtures, spatial transitions, and marker-sparse prototypes as higher boundary_or_mixed and ambiguity.",
            "High confidence should mean the prototype has coherent expression evidence for semantic graph edges and contrastive anchors.",
            "Use only the supplied expression evidence and local neighborhood summaries.",
        ),
    ),
    "Slide": DatasetSemanticProfile(
        dataset="Slide",
        default_sample="Slide_seqV2",
        default_root="data/Slide",
        samples=("Slide_seqV2",),
        evidence_fields=(
            "olfactory_sensory_neuron",
            "mitral_tufted_excitatory",
            "inhibitory_interneuron",
            "neuronal_synaptic",
            "oligodendrocyte_myelin",
            "astrocyte_ensheathing_glial",
            "immune_vascular_meningeal",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Mouse olfactory Slide-seqV2 spatial transcriptomics. "
            "This profile summarizes one high-density expression-spatial prototype, not an individual bead."
        ),
        interpretation_rules=(
            "No manual ground-truth labels are present in the local AnnData; optimize evidence for unsupervised semantic graph reliability and SC/DB-guided clustering rather than ARI/NMI label recovery.",
            "Do not assign a hard anatomical layer, olfactory-bulb region, or cluster label.",
            "Represent olfactory sensory, mitral/tufted excitatory, inhibitory interneuron, broad neuronal, myelin, astrocyte/ensheathing glial, immune, vascular, and meningeal programs as continuous evidence axes.",
            "Treat high-density boundary beads, mixed neuronal-glial evidence, and marker-sparse prototypes as higher boundary_or_mixed and ambiguity.",
            "High confidence should mean coherent marker evidence suitable for semantic edges, compact unsupervised clusters, and reliable contrastive anchors.",
        ),
        objective=(
            "Extract continuous biological evidence axes for unsupervised mouse olfactory Slide-seqV2 structure discovery, "
            "not a hard domain label. The output should improve semantically coherent neighborhoods, "
            "down-weight mixed prototypes, and support SC/DB-guided model selection."
        ),
    ),
    "Stereo": DatasetSemanticProfile(
        dataset="Stereo",
        default_sample="Stereo_seq",
        default_root="data/Stereo",
        samples=("Stereo_seq",),
        evidence_fields=(
            "olfactory_sensory_neuron",
            "mitral_tufted_excitatory",
            "inhibitory_interneuron",
            "neuronal_synaptic",
            "oligodendrocyte_myelin",
            "astrocyte_ensheathing_glial",
            "immune_vascular_meningeal",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Mouse olfactory Stereo-seq spatial transcriptomics. "
            "This profile summarizes one high-density expression-spatial prototype, not an individual bin."
        ),
        interpretation_rules=(
            "No manual ground-truth labels are present in the local AnnData; optimize evidence for unsupervised semantic graph reliability and SC/DB-guided clustering rather than ARI/NMI label recovery.",
            "Do not assign a hard anatomical layer, olfactory-bulb region, or cluster label.",
            "Represent olfactory sensory, mitral/tufted excitatory, inhibitory interneuron, broad neuronal, myelin, astrocyte/ensheathing glial, immune, vascular, and meningeal programs as continuous evidence axes.",
            "Treat high-density boundary bins, mixed neuronal-glial evidence, and marker-sparse prototypes as higher boundary_or_mixed and ambiguity.",
            "High confidence should mean coherent marker evidence suitable for semantic edges, compact unsupervised clusters, and reliable contrastive anchors.",
        ),
        objective=(
            "Extract continuous biological evidence axes for unsupervised mouse olfactory Stereo-seq structure discovery, "
            "not a hard domain label. The output should improve semantically coherent neighborhoods, "
            "down-weight mixed prototypes, and support SC/DB-guided model selection."
        ),
    ),
    "MOSTA_E9.5": DatasetSemanticProfile(
        dataset="MOSTA_E9.5",
        default_sample="E9.5_E1S1.MOSTA",
        default_root="data/MOSTA_E9.5",
        samples=(
            "E9.5_E1S1.MOSTA",
        ),
        evidence_fields=(
            "brain_neuroectoderm",
            "neural_crest_branchial_mesenchyme",
            "somitic_dermomyotome_sclerotome",
            "notochord_axial_mesoderm",
            "cardiac_muscle_heart",
            "vascular_hematopoietic_agm_cavity",
            "hepatic_endoderm_liver",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            "Mouse Organogenesis Spatiotemporal Transcriptomic Atlas (MOSTA) Stereo-seq "
            "at embryonic day 9.5, an early-organogenesis stage. The E9.5 files are "
            "sagittal sections from C57BL/6 mouse embryos; names such as E9.5_E1S1 "
            "identify the developmental stage, embryo/biological replicate, and section. "
            "The standard MOSTA section files are processed bin-50 expression matrices, "
            "so each observation aggregates a local spatial area rather than representing "
            "a guaranteed single cell. This profile summarizes one multi-observation "
            "expression-spatial prototype. The local E9.5_E1S1 reference contains 5,913 "
            "bins and 25,568 genes. Its annotation metadata describes 12 anatomical tissue "
            "domains: AGM, Brain, Branchial arch, Cavity, Connective tissue, Dermomyotome, "
            "Heart, Liver, Mesenchyme, Neural crest, Notochord, and Sclerotome. These names "
            "provide developmental context and evaluation metadata, not labels to predict."
        ),
        interpretation_rules=(
            "Do not output any of the 12 annotation names as a hard domain label and do not infer a domain from spatial position alone.",
            "Treat the annotations as tissue domains that can contain multiple cell states; score only marker-supported developmental programs from the supplied genes and neighbor-prototype evidence.",
            "Use brain_neuroectoderm for coherent neural-tube or early-brain progenitor programs, not for every prototype with generic developmental genes.",
            "Use neural_crest_branchial_mesenchyme for migratory neural-crest, pharyngeal/branchial-arch, mesenchymal, or extracellular-matrix-rich connective programs; preserve uncertainty when their origins cannot be separated.",
            "Use somitic_dermomyotome_sclerotome for paraxial-mesoderm and somite-compartment programs, including dermomyotome/myogenic or sclerotome/chondrogenic evidence, without forcing a compartment assignment.",
            "Use notochord_axial_mesoderm only for coherent axial midline or notochord signaling evidence; shared neural developmental genes alone are insufficient.",
            "Keep cardiac contractile evidence separate from vascular/hematopoietic evidence: cardiac_muscle_heart reflects cardiomyocyte and heart-muscle programs, whereas vascular_hematopoietic_agm_cavity reflects endothelial, aortic/AGM, erythroid, or blood-rich cavity programs.",
            "Use hepatic_endoderm_liver for coherent early hepatic or endodermal programs; do not call liver from generic epithelial or proliferation signals alone.",
            "Because bin-50 observations and prototypes can span adjacent embryonic structures, conflicting lineage programs, transition zones, blood-filled cavities, or marker-sparse profiles should increase boundary_or_mixed and ambiguity.",
            "High confidence requires a coherent developmental program supported by both enriched markers and neighborhood context; stage expectations are context, not evidence by themselves.",
        ),
        cache_tag="mosta_e95_bin50_v1",
        objective=(
            "Extract ten continuous developmental-program evidence axes for unsupervised "
            "structure discovery in MOSTA E9.5 mouse-embryo Stereo-seq, not a hard organ, "
            "tissue-domain, or cell-type label. The evidence should preserve early "
            "organogenesis lineage relationships, distinguish coherent tissue programs "
            "from mixed bin-50 boundaries, improve semantic neighborhoods, and select "
            "reliable contrastive anchors without using annotation labels as input."
        ),
    ),
}


DISC_EVIDENCE_FIELDS = list(DATASET_PROFILES[MAINLINE_DATASET].evidence_fields)


API_SYSTEM_PROMPT = _make_system_prompt(tuple(DISC_EVIDENCE_FIELDS))


def get_dataset_profile(dataset: str, sample: str = None) -> DatasetSemanticProfile:
    canonical = str(dataset)
    sample_name = Path(str(sample)).name.upper() if sample is not None else ""
    if sample_name.endswith(".H5AD"):
        sample_name = sample_name[:-5]
    if canonical in {"E9.5", "MOSTA_E9.5"}:
        return DATASET_PROFILES["MOSTA_E9.5"]
    if (
        canonical in {"Stereo", "MOSTA"}
        and sample_name.startswith("E9.5_")
        and "MOSTA" in sample_name
    ):
        return DATASET_PROFILES["MOSTA_E9.5"]
    if canonical in DATASET_PROFILES:
        return DATASET_PROFILES[canonical]
    return DatasetSemanticProfile(
        dataset=canonical,
        default_sample=canonical,
        default_root=f"data/{canonical}",
        samples=(canonical,),
        evidence_fields=(
            "cell_type_program_1",
            "cell_type_program_2",
            "cell_type_program_3",
            "lineage_or_state_program_1",
            "lineage_or_state_program_2",
            "microenvironment_program",
            "technical_or_low_specificity_program",
            "boundary_or_mixed",
            "ambiguity",
            "confidence",
        ),
        context=(
            f"{canonical} spatial transcriptomics. This profile summarizes one "
            "expression-spatial prototype, not an individual spot or cell."
        ),
        interpretation_rules=(
            "Do not output a hard domain or cluster label.",
            "Map the strongest marker-supported biological programs onto the first seven continuous evidence axes.",
            "Use boundary_or_mixed and ambiguity for mixed, transition-like, marker-sparse, or internally heterogeneous prototypes.",
            "High confidence should mean the evidence vector is reliable for semantic graph edges and contrastive anchors.",
            "Use only the supplied expression evidence and local neighborhood summaries.",
        ),
    )


def get_dataset_spots_per_cluster(dataset: str, sample: str = None) -> int:
    """Return the dataset-specific prototype multiplier A in A × n_clusters."""
    spots_per_cluster = int(get_dataset_profile(dataset, sample).spots_per_cluster)
    if spots_per_cluster < 1:
        raise ValueError(f"spots_per_cluster must be positive, got {spots_per_cluster}")
    return spots_per_cluster


def resolve_spots_per_cluster(
    dataset: str,
    sample: str = None,
    spots_per_cluster: int = None,
) -> int:
    """Use an explicit multiplier for a comparison sweep, or the dataset default."""
    if spots_per_cluster is None:
        return get_dataset_spots_per_cluster(dataset, sample=sample)
    spots_per_cluster = int(spots_per_cluster)
    if spots_per_cluster < 1:
        raise ValueError(f"spots_per_cluster must be positive, got {spots_per_cluster}")
    return spots_per_cluster


def resolve_n_prototypes(
    n_obs: int,
    n_prototypes=None,
    n_clusters: int = None,
    spots_per_cluster: int = None,
) -> int:
    """Resolve the prototype budget from A × the current task's cluster count.

    A numeric ``n_prototypes`` remains an explicit reproducibility override.
    Known-count tasks use the clustering target; unlabeled tasks use a bounded
    sample-size budget without inferring or inventing ground-truth categories.
    """
    n_obs = int(n_obs)
    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}")

    if n_prototypes is not None and str(n_prototypes).lower() not in {
        "",
        "auto",
        "none",
    }:
        resolved = int(n_prototypes)
    else:
        if n_clusters is None or int(n_clusters) == 0:
            # Unlabeled tasks derive a bounded prototype budget from sample size.
            scaled = round(REFERENCE_PROTOTYPES * n_obs / REFERENCE_SPOTS)
            return min(n_obs, max(MIN_PROTOTYPES, min(MAX_PROTOTYPES, scaled)))
        if int(n_clusters) < 0:
            raise ValueError("n_clusters must be nonnegative.")
        if spots_per_cluster is None or int(spots_per_cluster) < 1:
            raise ValueError(
                "spots_per_cluster must be a positive integer when n_prototypes is auto."
            )
        resolved = int(n_clusters) * int(spots_per_cluster)

    return max(1, min(n_obs, resolved))
