"""Loading package"""

import os
from pathlib import Path

os.environ["R_ENCODING"] = "UTF-8"
os.environ["LC_ALL"] = "C"
import scanpy as sc
from sklearn import metrics
from .preprocess import (
    preprocess,
    construct_interaction,
    construct_interaction_KNN,
    construct_interaction_edge_list,
    get_feature,
)
import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


class Function:
    def __init__(
        self,
        sample: str = "151673",
        datatype: str = "10x",
        device: int = 0,
        n_clusters: int = 0,
        data_root=None,
        preprocess_config=None,
    ):
        super(Function, self).__init__()
        self._sample = sample
        self._datatype = datatype
        self._device = device
        self.n_clusters = n_clusters
        self.data_root = Path(
            data_root or os.environ.get("SEALING_DATA_ROOT", "data")
        ).expanduser()
        self.preprocess_config = dict(preprocess_config or {})
        self.file_fold = self._resolve_data_path()

    def _resolve_data_path(self):
        """Resolve a dataset folder below the user-provided data root."""
        dataset_root = self.data_root / self._datatype
        return (
            dataset_root / str(self._sample)
            if self._datatype == "DLPFC"
            else dataset_root
        )

    def loadData(self):
        """Load one sample from the user-provided data root."""
        if self._datatype in ["10x", "DLPFC", "MBA", "MBP", "HBC"]:
            adata = sc.read_visium(
                str(self.file_fold),
                count_file="filtered_feature_bc_matrix.h5",
                load_images=True,
            )
        elif self._datatype in ["MBC_s1", "MBC_s2", "MBM_s1", "MBM_s2", "Mouse_Olf"]:
            adata = sc.read_h5ad(
                str(self.file_fold / "filtered_feature_bc_matrix.h5ad")
            )
        elif self._datatype == "Mouse_HT":
            adata = sc.read_h5ad(
                str(self.file_fold / "filtered_feature_bc_matrix_200115_08.h5ad")
            )
        elif self._datatype in ["Stereo", "Slide"]:
            adata = sc.read_h5ad(str(self.file_fold / f"{self._sample}.h5ad"))
        elif self._datatype == "Simulated_STARmap":
            adata = sc.read_h5ad(str(self.file_fold / "STARmap_20180505_BY3_1k.h5ad"))
        else:
            adata = sc.read_h5ad(self.file_fold / f"{self._sample}.h5ad")

        if self._datatype in ["DLPFC", "MBA", "HBC"]:
            metadata_path = os.path.join(self.file_fold, "metadata.tsv")
            if os.path.exists(metadata_path):
                df_meta = pd.read_csv(metadata_path, sep="\t", index_col=0)
                label_col = (
                    "layer_guess" if self._datatype == "DLPFC" else "ground_truth"
                )
                if label_col in df_meta.columns:
                    adata_prefix = [
                        bc.rsplit("-", 1)[0] for bc in adata.obs.index.tolist()
                    ]
                    meta_prefix = [
                        bc.rsplit("-", 1)[0] for bc in df_meta.index.tolist()
                    ]
                    prefix_to_label = dict(zip(meta_prefix, df_meta[label_col]))
                    adata.obs["ground_truth"] = [
                        prefix_to_label.get(prefix, "Unknown")
                        for prefix in adata_prefix
                    ]

        if self.n_clusters == 0:
            self.resolve_cluster_count(adata)

        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        return adata

    def resolve_cluster_count(self, adata):
        """Use configured labels when available; otherwise select unsupervised mode."""
        if self.n_clusters > 0:
            return self.n_clusters
        count, _ = self.get_true_labels(adata, set_n_clusters=True)
        self.n_clusters = count
        if count == 0:
            print(
                f"[INFO] {self._datatype}/{self._sample}: using unsupervised clustering"
            )
        return self.n_clusters

    def process(self, adata):
        adata = adata.copy()
        preprocess_force = bool(self.preprocess_config.get("force", False))
        if preprocess_force or "highly_variable" not in adata.var.keys():
            preprocess_kwargs = {
                key: value
                for key, value in self.preprocess_config.items()
                if key != "force"
            }
            preprocess(adata, sample=self._sample, **preprocess_kwargs)

        if self._datatype in ["MBM_s1", "MBM_s2"]:
            print(f"Building memory-safe spatial edge list for {self._datatype}")
            u, v = construct_interaction_edge_list(adata)
        else:
            if "adj" not in adata.obsm:
                print(f"Building spatial graph for {self._datatype}")
                if self._datatype in ["Stereo", "Slide"]:
                    u, v = construct_interaction_KNN(adata)
                else:
                    u, v = construct_interaction(adata)
            else:
                adjacency = adata.obsm["adj"]
                if hasattr(adjacency, "tocoo"):
                    adjacency = adjacency.tocoo()
                    u, v = adjacency.col, adjacency.row
                else:
                    v, u = np.nonzero(np.asarray(adjacency))

        if "feat" not in adata.obsm.keys():
            get_feature(adata)

        return adata, u, v

    def get_true_labels(self, adata, set_n_clusters=False, verbose=True):
        """Resolve labels from the supported metadata columns in one place."""
        from .labels import resolve_labels

        count, labels = resolve_labels(adata, self._datatype, self._sample)
        if verbose:
            print(
                f"[Labels] {self._datatype}/{self._sample}: {count} annotated classes"
            )
        return (count, labels) if set_n_clusters else labels

    def evaluate_domain_clustering(self, adata, quiet=False):
        true_labels = self.get_true_labels(adata, verbose=not quiet)
        adata.obs["true_label"] = true_labels

        if not bool(adata.uns.get("has_ground_truth", True)):
            adata.uns["ARI"] = float("nan")
            adata.uns["NMI"] = float("nan")
            adata.uns["ACC"] = float("nan")
            adata.uns["F1"] = float("nan")
            adata.uns["completeness"] = float("nan")
            adata.uns["homogeneity"] = float("nan")
            adata.uns["v_measure"] = float("nan")

            labels = adata.obs["domain"].astype(str).values
            n_unique = len(np.unique(labels))
            adata.uns["n_clusters_requested"] = int(self.n_clusters)
            adata.uns["n_clusters_observed"] = int(n_unique)
            if 1 < n_unique < adata.n_obs:
                adata.uns["SC"] = metrics.silhouette_score(
                    adata.obsm["emb_pca"], labels
                )
                adata.uns["DB"] = metrics.davies_bouldin_score(
                    adata.obsm["emb_pca"], labels
                )
            else:
                adata.uns["SC"] = float("nan")
                adata.uns["DB"] = float("nan")

            if not quiet:
                print(f"Sample: {self._sample}")
                print("Mode: Unsupervised clustering (no ground truth labels)")
                print(f"  Total spots: {len(adata)}")
                print(f'  Number of clusters: {adata.obs["domain"].nunique()}')
                print(f'  SC: {adata.uns["SC"]:.4f}')
                print(f'  DB: {adata.uns["DB"]:.4f}')
            return adata

        valid_mask = adata.obs["true_label"].notna()
        adata_valid = adata[valid_mask].copy()

        pred_labels = adata_valid.obs["domain"]
        true_labels_valid = adata_valid.obs["true_label"]

        contingency = pd.crosstab(
            pd.Series(true_labels_valid.astype(str), dtype="string"),
            pd.Series(pred_labels.astype(str), dtype="string"),
        )
        rows, columns = linear_sum_assignment(-contingency.to_numpy())
        label_mapping = {
            contingency.columns[column]: contingency.index[row]
            for row, column in zip(rows, columns)
        }
        aligned_predictions = [
            label_mapping.get(str(label), "__unmatched_cluster__")
            for label in pred_labels
        ]

        adata.uns["ARI"] = metrics.adjusted_rand_score(true_labels_valid, pred_labels)
        adata.uns["NMI"] = metrics.normalized_mutual_info_score(
            true_labels_valid, pred_labels
        )
        adata.uns["ACC"] = float(
            np.mean(
                true_labels_valid.astype(str).to_numpy()
                == np.asarray(aligned_predictions)
            )
        )
        adata.uns["F1"] = metrics.f1_score(
            true_labels_valid.astype(str),
            aligned_predictions,
            average="macro",
            zero_division=0,
        )
        adata.uns["completeness"] = metrics.completeness_score(
            true_labels_valid, pred_labels
        )
        adata.uns["homogeneity"] = metrics.homogeneity_score(
            true_labels_valid, pred_labels
        )
        adata.uns["v_measure"] = metrics.v_measure_score(true_labels_valid, pred_labels)
        adata.uns["SC"] = float("nan")
        adata.uns["DB"] = float("nan")

        if not quiet:
            print(f"Sample: {self._sample}")
            print(
                f"Evaluation on {len(adata_valid)} valid spots (excluding {len(adata) - len(adata_valid)} NA/excluded labels):"
            )
            print(f'  ARI: {adata.uns["ARI"]:.4f}')
            print(f'  NMI: {adata.uns["NMI"]:.4f}')
            print(f'  ACC: {adata.uns["ACC"]:.4f}')
            print(f'  F1: {adata.uns["F1"]:.4f}')
            print(f'  Completeness: {adata.uns["completeness"]:.4f}')
            print(f'  Homogeneity: {adata.uns["homogeneity"]:.4f}')
            print(f'  V-measure: {adata.uns["v_measure"]:.4f}')

        return adata

    def _guard_refinement_cluster_count(self, adata, refinement_requested, quiet=False):
        """Keep the requested cluster count when spatial refinement collapses labels."""
        requested = int(self.n_clusters)
        if "domain" not in adata.obs:
            raise RuntimeError("Clustering did not produce obs['domain'].")
        raw_key = next(
            (key for key in ("mclust", "leiden", "louvain", "domain")
             if key in adata.obs),
        )
        raw_labels = adata.obs[raw_key].astype(str).to_numpy()
        final_labels = adata.obs["domain"].astype(str).to_numpy()
        raw_count = int(pd.unique(raw_labels).size)
        refined_count = int(pd.unique(final_labels).size)
        adata.uns.update(
            n_clusters_requested=requested,
            n_clusters_raw=raw_count,
            n_clusters_refined=refined_count,
            refinement_requested=bool(refinement_requested),
        )
        if not refinement_requested:
            adata.obs["domain"] = raw_labels
            adata.uns["refinement_status"] = "not_requested"
            adata.uns["refinement_applied"] = False
            return adata
        adata.obs["domain_refined_candidate"] = final_labels
        if refined_count == requested:
            adata.uns["refinement_status"] = "accepted"
            adata.uns["refinement_applied"] = True
            return adata
        adata.obs["domain"] = raw_labels
        adata.uns["refinement_status"] = "rejected_cluster_collapse"
        adata.uns["refinement_applied"] = False
        return adata

    def evaluate_clustering(self, adata, quiet=False):
        """Backward-compatible alias for older training/notebook calls."""
        return self.evaluate_domain_clustering(adata, quiet=quiet)

    def clusting(
        self,
        adata,
        model,
        graph,
        x,
        device,
        refinement=False,
        spatial_coords=None,
        cluster_fused_weight=1.0,
        cluster_struct_weight=1.0,
        cluster_sem_weight=0.5,
        cluster_fused_pca_dim=27,
        cluster_struct_pca_dim=6,
        cluster_sem_pca_dim=3,
        refinement_radius=50,
    ):
        """Cluster one embedding and return the active evaluation score."""
        result = self.clusting_for_plot(
            adata,
            model,
            graph,
            x,
            device,
            refinement=refinement,
            cluster_fused_weight=cluster_fused_weight,
            cluster_struct_weight=cluster_struct_weight,
            cluster_sem_weight=cluster_sem_weight,
            cluster_fused_pca_dim=cluster_fused_pca_dim,
            cluster_struct_pca_dim=cluster_struct_pca_dim,
            cluster_sem_pca_dim=cluster_sem_pca_dim,
            refinement_radius=refinement_radius,
        )
        return (
            result.uns["SC"]
            if not result.uns.get("has_ground_truth", True)
            else result.uns["ARI"]
        )

    def clustingAndPlot(
        self,
        adata,
        model,
        graph,
        x,
        device,
        sample,
        path,
        refinement=False,
        **kwargs,
    ):
        """Cluster and save the result; plotting is handled by the public plot helper."""
        result = self.clusting_for_plot(
            adata, model, graph, x, device, refinement=refinement, **kwargs
        )
        output_dir = os.path.join(path, sample)
        os.makedirs(output_dir, exist_ok=True)
        self.create_visium_style_plots(result, output_dir)
        return result

    def clustingAndSave(
        self, adata, model, graph, x, device, sample, path, refinement=False
    ):
        """Cluster and save an AnnData result."""
        result = self.clusting_for_plot(
            adata, model, graph, x, device, refinement=refinement
        )
        os.makedirs(path, exist_ok=True)
        result.write(os.path.join(path, f"{sample}.h5ad"), compression="gzip")
        return result

    def clusting_no_label(self, adata, model, graph, x, device, refinement=False):
        """Compatibility wrapper for unsupervised clustering."""
        return self.clusting_for_plot(
            adata, model, graph, x, device, refinement=False, quiet=True
        )

    def _view_pca_part(self, tensor, n_components, weight):
        n_components = int(n_components)
        weight = float(weight)
        if not 0 < n_components or not 0 < weight < float("inf"):
            raise ValueError(
                "Every mainline PCA view requires positive dimensions and weight."
            )

        tensor = torch.nn.functional.normalize(tensor, p=2, dim=-1)
        values = tensor.detach().cpu().numpy()
        max_components = min(values.shape[0] - 1, values.shape[1])
        if max_components <= 0:
            raise ValueError("PCA clustering requires at least two observations.")

        n_components = min(n_components, max_components)
        reduced = PCA(n_components=n_components, random_state=42).fit_transform(values)
        reduced = torch.as_tensor(reduced, dtype=tensor.dtype, device=tensor.device)
        return reduced * weight

    def _build_cluster_embedding(
        self,
        model,
        graph,
        x,
        device,
        cluster_fused_weight=1.0,
        cluster_struct_weight=1.0,
        cluster_sem_weight=0.5,
        cluster_fused_pca_dim=27,
        cluster_struct_pca_dim=6,
        cluster_sem_pca_dim=3,
    ):
        views = model.embed_views(graph.to(device), x.to(device))
        specs = (
            ("fused", cluster_fused_pca_dim, cluster_fused_weight),
            ("struct", cluster_struct_pca_dim, cluster_struct_weight),
            ("sem", cluster_sem_pca_dim, cluster_sem_weight),
        )
        return torch.cat(
            [
                self._view_pca_part(views[name], dimensions, weight)
                for name, dimensions, weight in specs
            ],
            dim=-1,
        )

    def clusting_for_plot(
        self,
        adata,
        model,
        graph,
        x,
        device,
        refinement=False,
        quiet=False,
        cluster_fused_weight=1.0,
        cluster_struct_weight=1.0,
        cluster_sem_weight=0.5,
        cluster_fused_pca_dim=27,
        cluster_struct_pca_dim=6,
        cluster_sem_pca_dim=3,
        refinement_radius=50,
        cluster_method="mclust",
    ):
        """Return clustering results for evaluation and visualization."""
        model.eval()

        with torch.no_grad():
            x = (
                self._build_cluster_embedding(
                    model,
                    graph,
                    x,
                    device,
                    cluster_fused_weight=cluster_fused_weight,
                    cluster_struct_weight=cluster_struct_weight,
                    cluster_sem_weight=cluster_sem_weight,
                    cluster_fused_pca_dim=cluster_fused_pca_dim,
                    cluster_struct_pca_dim=cluster_struct_pca_dim,
                    cluster_sem_pca_dim=cluster_sem_pca_dim,
                )
                .detach()
                .cpu()
                .numpy()
            )

        adata.obsm["emb"] = x
        adata.obsm["emb_pca"] = x

        from .utils import clustering

        use_refinement = refinement and self._datatype not in ("Stereo", "Slide")
        clustering(
            adata,
            self.n_clusters,
            radius=int(refinement_radius),
            method=cluster_method,
            refinement=use_refinement,
            verbose=not quiet,
            use_precomputed_pca=True,
        )
        self._guard_refinement_cluster_count(
            adata, refinement_requested=use_refinement, quiet=quiet
        )
        return self.evaluate_domain_clustering(adata, quiet=quiet)

    def create_visium_style_plots(self, adata, save_path):
        """Create standard spatial and UMAP plots for Visium-style data."""
        ARI = adata.uns.get("ARI", 0)
        NMI = adata.uns.get("NMI", 0)
        true_label_col = (
            "true_label" if "true_label" in adata.obs.columns else "ground_truth"
        )

        print("Creating spatial clustering plot...")
        plt.rcParams["figure.figsize"] = (12, 5)
        sc.pl.spatial(
            adata,
            img_key="hires",
            color=[true_label_col, "domain"],
            title=["Ground Truth", f"Prediction (ARI={ARI:.4f}, NMI={NMI:.4f})"],
            show=False,
            save=False,
        )
        plt.savefig(
            os.path.join(save_path, "spatial_comparison.pdf"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.savefig(
            os.path.join(save_path, "spatial_comparison.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        plt.rcParams["figure.figsize"] = (8, 6)
        sc.pl.spatial(
            adata,
            img_key="hires",
            color="domain",
            title=f"Spatial Domains (ARI={ARI:.4f}, NMI={NMI:.4f})",
            show=False,
            save=False,
        )
        plt.savefig(
            os.path.join(save_path, "spatial_domains.pdf"), bbox_inches="tight", dpi=300
        )
        plt.savefig(
            os.path.join(save_path, "spatial_domains.png"), bbox_inches="tight", dpi=300
        )
        plt.close()

        print("Creating UMAP visualization...")
        sc.pp.neighbors(adata, use_rep="emb", n_neighbors=15)
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (12, 5)
        sc.pl.umap(
            adata,
            color=[true_label_col, "domain"],
            title=[
                "Ground Truth (UMAP)",
                f"Prediction (UMAP)\nARI={ARI:.4f}, NMI={NMI:.4f}",
            ],
            legend_loc="right margin",
            show=False,
            save=False,
        )
        plt.savefig(
            os.path.join(save_path, "umap_comparison.pdf"), bbox_inches="tight", dpi=300
        )
        plt.savefig(
            os.path.join(save_path, "umap_comparison.png"), bbox_inches="tight", dpi=300
        )
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sc.pl.umap(
            adata,
            color=true_label_col,
            title="Ground Truth",
            ax=axes[0],
            legend_loc=None,
            show=False,
            frameon=False,
        )
        sc.pl.umap(
            adata,
            color="domain",
            title=f"Prediction (ARI={ARI:.4f})",
            ax=axes[1],
            legend_loc="right margin",
            show=False,
            frameon=False,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, "umap_comparison.pdf"), bbox_inches="tight", dpi=300
        )
        plt.savefig(
            os.path.join(save_path, "umap_comparison.png"), bbox_inches="tight", dpi=300
        )
        plt.close()

        print(f"Generated visualization files in {save_path}")
