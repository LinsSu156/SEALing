import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors


def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm["spatial"]

    distance_matrix = ot.dist(position, position, metric="euclidean")
    n_spot = distance_matrix.shape[0]

    adata.obsm["distance_matrix"] = distance_matrix

    interaction = np.zeros([n_spot, n_spot])

    u = []
    v = []
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
            u.append(y)
            v.append(i)

    adata.obsm["graph_neigh"] = interaction

    adj = interaction
    adj = np.where(adj > 1, 1, adj)

    adata.obsm["adj"] = adj

    return u, v


def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm["spatial"]
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm["graph_neigh"] = interaction

    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm["adj"] = adj
    print("Graph constructed!")
    u = y.tolist()
    v = x.tolist()
    return u, v


def construct_interaction_edge_list(adata, n_neighbors=3):
    """Construct k-NN spatial edges without materializing an n_spot x n_spot matrix."""
    position = np.asarray(adata.obsm["spatial"])
    n_spot = int(position.shape[0])
    if n_spot <= n_neighbors:
        raise ValueError(
            f"Need more than {n_neighbors} spots to construct the spatial graph; got {n_spot}."
        )

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    destinations = np.repeat(np.arange(n_spot, dtype=np.int64), n_neighbors)
    sources = indices[:, 1:].reshape(-1).astype(np.int64, copy=False)
    print(
        f"Edge-list graph constructed: spots={n_spot}, "
        f"directed_edges={sources.size}, neighbors={n_neighbors}"
    )
    return sources.tolist(), destinations.tolist()


def preprocess(
    adata,
    sample=None,
    gene_num=None,
    hvg_flavor="seurat_v3",
    gene_min_cells=0,
    target_sum=1e4,
    scale_max_value=10,
):
    if gene_num is None or int(gene_num) <= 0:
        gene_num = (
            2000
            if sample in ["151507", "151510", "151670", "151672", "151674", "151675"]
            else 3000
        )
    gene_num = int(gene_num)
    gene_min_cells = int(gene_min_cells or 0)
    target_sum = float(target_sum)
    scale_max_value = (
        None
        if scale_max_value is None or float(scale_max_value) <= 0
        else float(scale_max_value)
    )

    if gene_min_cells > 0:
        before_genes = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=gene_min_cells)
        print(
            f"[Preprocess] filter genes min_cells={gene_min_cells}: {before_genes}->{adata.n_vars}"
        )

    sc.pp.highly_variable_genes(adata, flavor=str(hvg_flavor), n_top_genes=gene_num)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=scale_max_value)


def get_feature(adata, deconvolution=False):
    if deconvolution:
        feature_matrix = adata.X
    else:
        highly_variable = np.asarray(adata.var["highly_variable"], dtype=bool)
        if highly_variable.all():
            feature_matrix = adata.X
        else:
            feature_matrix = adata[:, highly_variable].X

    if isinstance(feature_matrix, csc_matrix) or isinstance(feature_matrix, csr_matrix):
        feat = feature_matrix.toarray().astype(np.float32, copy=False)
    else:
        feat = np.asarray(feature_matrix, dtype=np.float32)

    adata.obsm["feat"] = feat


def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
