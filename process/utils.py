import numpy as np
import scanpy as sc
import ot
from sklearn.decomposition import PCA


def mclust_R(
    adata,
    num_cluster,
    modelNames="EEE",
    used_obsm="emb_pca",
    random_seed=2020,
    verbose=True,
):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects

    robjects.r("suppressPackageStartupMessages(library(mclust))")

    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)
    rmclust = robjects.r["Mclust"]

    r_embedding = None
    res = None
    try:
        r_embedding = rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm])
        res = rmclust(r_embedding, num_cluster, modelNames, verbose=verbose)
        mclust_res = np.array(res[-2])
    finally:
        del res, r_embedding
        robjects.r("gc(verbose=FALSE)")

    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int")
    adata.obs["mclust"] = adata.obs["mclust"].astype("category")
    return adata


def clustering(
    adata,
    n_clusters=None,
    radius=50,
    key="emb",
    method="mclust",
    start=0.1,
    end=3.0,
    increment=0.01,
    refinement=False,
    verbose=True,
    use_precomputed_pca=False,
    pca_components=27,
):
    """Cluster embeddings; an unspecified count uses unconstrained Leiden."""
    if n_clusters is not None and n_clusters < 0:
        raise ValueError("n_clusters must be nonnegative.")
    if method not in {"mclust", "leiden", "louvain"}:
        raise ValueError(f"Unsupported clustering method: {method}")
    if adata.n_obs < 2:
        raise ValueError("Clustering requires at least two observations.")
    if not use_precomputed_pca:
        components = min(int(pca_components), adata.obsm[key].shape[1], adata.n_obs - 1)
        adata.obsm["emb_pca"] = PCA(
            n_components=components, random_state=42
        ).fit_transform(adata.obsm[key])
    if not n_clusters:
        method = "leiden"
    if method == "mclust":
        mclust_R(adata, num_cluster=n_clusters, used_obsm="emb_pca", verbose=verbose)
    else:
        if n_clusters:
            resolution = search_res(
                adata, n_clusters, method, "emb_pca", start, end, increment
            )
        else:
            sc.pp.neighbors(
                adata, n_neighbors=min(50, adata.n_obs - 1), use_rep="emb_pca"
            )
            resolution = 1.0
        getattr(sc.tl, method)(adata, random_state=0, resolution=resolution)
    adata.obs["domain"] = adata.obs[method].astype(str)
    if refinement:
        adata.obs["domain"] = refine_label(adata, radius, key="domain")
    return adata


def refine_label(adata, radius=50, key="label"):
    """Preserve reference distance sorting and first-neighbor tie breaking."""
    labels = adata.obs[key].values
    k = min(int(radius), adata.n_obs - 1)
    if k <= 0:
        return [str(label) for label in labels]
    positions = adata.obsm["spatial"]
    distance = ot.dist(positions, positions, metric="euclidean")
    refined = []
    for row in distance:
        neighbors = [labels[index] for index in row.argsort()[1:k + 1]]
        refined.append(str(max(neighbors, key=neighbors.count)))
    return refined


def search_res(
    adata,
    n_clusters,
    method="leiden",
    use_rep="emb",
    start=0.1,
    end=3.0,
    increment=0.01,
):
    """Search descending resolutions for a requested category count."""
    sc.pp.neighbors(adata, n_neighbors=min(50, adata.n_obs - 1), use_rep=use_rep)
    for resolution in np.arange(start, end, increment)[::-1]:
        getattr(sc.tl, method)(adata, random_state=0, resolution=float(resolution))
        if adata.obs[method].nunique() == n_clusters:
            return float(resolution)
    raise ValueError("Requested category count was not found in the resolution range.")
