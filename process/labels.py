"""Ground-truth metadata resolution shared by training and evaluation."""

import pandas as pd


LABEL_COLUMNS = ("ground_truth", "cluster", "annotation", "celltype", "label")
MISSING_LABELS = {"", "not_annotated", "unknown", "nan", "na", "none", "<na>"}


def resolve_labels(adata, dataset="", sample=""):
    """Return category count and normalized labels, or zero for unlabeled data.

    Only the explicit metadata allowlist is considered. Predicted domains and
    arbitrary observation columns must never become evaluation ground truth.
    """
    preferred = {
        "Mouse_HT": "cluster",
        "Slide": "cluster",
        "Stereo": "annotation",
        "Simulated_STARmap": "label",
    }.get(dataset)
    columns = ([preferred] if preferred else []) + list(LABEL_COLUMNS)
    source = next((column for column in columns if column in adata.obs), None)
    labels = (
        adata.obs[source].astype("string").str.strip()
        if source
        else pd.Series(pd.NA, index=adata.obs.index, dtype="string")
    )
    labels = labels.mask(labels.str.lower().isin(MISSING_LABELS))
    count = int(labels.nunique())
    labels = (
        labels.astype(object).where(labels.notna(), float("nan")).astype("category")
    )
    adata.uns["has_ground_truth"] = count > 0
    adata.obs["ground_truth"] = labels
    return count, labels
