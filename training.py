"""Shared sample training and checkpoint selection."""

import numpy as np

LR_SCHEDULER_HORIZON_EPOCHS = 900


def _format_train_status(epoch, loss_dict):
    return (
        "[Train] epoch={} loss={:.4f} rec={:.4f} align={:.4f} "
        "contrast={:.4f}(w={:.4f},nce={:.4f}) gate={:.4f}[{:.3f},{:.3f}] "
        "prune={}/{}({:.2%}) sem_edges={} pos={} neg_avg={:.2f}"
    ).format(
        epoch,
        loss_dict["loss"],
        loss_dict["loss_rec"],
        loss_dict["loss_align"],
        loss_dict["loss_contrast"],
        loss_dict["contrast_loss_weight"],
        loss_dict["contrast_loss_nce"],
        loss_dict["view_fusion_gate"],
        loss_dict["view_fusion_gate_min"],
        loss_dict["view_fusion_gate_max"],
        loss_dict["pruned_edges"],
        loss_dict["total_edges"],
        loss_dict["prune_ratio"],
        loss_dict["semantic_view_edges"],
        loss_dict["contrast_positive_pairs"],
        loss_dict["contrast_negative_pairs_avg"],
    )


def _safe_metric_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unsupervised_selection_score(sc):
    sc_value = _safe_metric_float(sc)
    if not np.isfinite(sc_value):
        return float("-inf")
    return sc_value


def _format_eval_status(epoch, ari, nmi, best_ari, is_best):
    best_text = "*" if is_best else f"{best_ari:.4f}"
    return "[Eval] epoch={} ARI={:.4f} NMI={:.4f} best={}".format(
        epoch,
        float(ari),
        float(nmi),
        best_text,
    )


def _format_unsupervised_eval_status(epoch, sc, db, best_sc, is_best):
    if is_best:
        best_text = "*"
    elif np.isfinite(float(best_sc)):
        best_text = f"{best_sc:.4f}"
    else:
        best_text = "nan"
    return "[Eval] epoch={} SC={:.4f} DB={:.4f} best={}".format(
        epoch,
        float(sc),
        float(db),
        best_text,
    )


def train_single_sample(
    args,
    sample,
    seed,
    dataset,
    device,
    logger=None,
    adata=None,
    u=None,
    v=None,
    n_clusters=0,
    show_plots=False,
):
    """Train one sample and return the best checkpoint's clustering result."""
    import torch
    import dgl
    import numpy as np
    import logging
    from tqdm import tqdm
    from pathlib import Path
    from process.function import Function
    from models import build_model
    from utils import set_random_seed, create_optimizer, get_current_lr

    configured_n_clusters = int(n_clusters or getattr(args, "n_clusters", 0) or 0)
    preprocess_config = {
        "force": bool(getattr(args, "preprocess_force", False)),
        "gene_num": int(getattr(args, "preprocess_gene_num", 0) or 0),
        "hvg_flavor": getattr(args, "preprocess_hvg_flavor", "seurat_v3"),
        "gene_min_cells": int(getattr(args, "preprocess_gene_min_cells", 0) or 0),
        "target_sum": float(getattr(args, "preprocess_target_sum", 1e4)),
        "scale_max_value": float(getattr(args, "preprocess_scale_max_value", 10.0)),
    }
    function = Function(
        sample,
        dataset,
        n_clusters=configured_n_clusters,
        data_root=getattr(args, "data_root", None),
        preprocess_config=preprocess_config,
    )

    if adata is None:
        adata = function.loadData()
        adata, u, v = function.process(adata)
    else:
        if u is None or v is None:
            raise ValueError(
                "When providing adata, must also provide u and v (edge lists)"
            )

        if n_clusters > 0:
            function.n_clusters = n_clusters
        else:
            function.resolve_cluster_count(adata)

    graph = dgl.graph(
        (torch.as_tensor(u, dtype=torch.int64), torch.as_tensor(v, dtype=torch.int64)),
        num_nodes=adata.n_obs,
    )
    if args.self_loop:
        graph = dgl.add_self_loop(graph)
    else:
        isolated = torch.nonzero(graph.in_degrees() == 0, as_tuple=True)[0]
        if isolated.numel():
            graph = dgl.add_edges(graph, isolated, isolated)
    x = torch.tensor(adata.obsm["feat"], dtype=torch.float32)
    args.num_features = x.shape[1]

    set_random_seed(seed)

    model = build_model(args)
    model.to(device)

    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    logging.info("start training..")
    best_ari = 0.0
    best_sc = float("-inf")
    best_model_state = None
    best_adata = None
    best_epoch = None
    last_eval_metrics = None
    evaluation_trace = []

    def clone_model_state(current_model):
        return {
            key: value.detach().clone()
            for key, value in current_model.state_dict().items()
        }

    graph = graph.to(device)
    x = x.to(device)
    from llm_semantic.evidence import load_semantic_evidence

    evidence = load_semantic_evidence(
        adata,
        dataset,
        sample,
        evidence_path=getattr(args, "evidence_path", None),
        evidence_root=getattr(args, "evidence_root", "cache/semantic_evidence"),
        model_name=getattr(args, "llm_model_name", "gpt-5.5"),
    )
    model.update_semantic_graph(torch.as_tensor(evidence, device=device))
    static_graph_init_stats = model.initialize_static_training_graphs(graph, x.device)

    result_path = Path(getattr(args, "output_root", "results")) / dataset / sample
    result_path.mkdir(parents=True, exist_ok=True)

    logging.info("Use fixed-horizon cosine scheduler")
    scheduler_horizon = float(
        getattr(args, "scheduler_horizon", LR_SCHEDULER_HORIZON_EPOCHS)
    )
    if scheduler_horizon <= 0:
        raise ValueError("scheduler_horizon must be positive.")

    def scheduler_fn(epoch):
        progress = min(float(epoch), scheduler_horizon) / scheduler_horizon
        return (1.0 + np.cos(progress * np.pi)) * 0.5

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)

    eval_interval = min(
        max(1, int(getattr(args, "eval_interval", 100))),
        max(1, int(args.max_epoch)),
    )
    monitor_interval = max(1, int(getattr(args, "monitor_interval", 100)))
    eval_report_interval = 100
    eval_verbose = bool(getattr(args, "eval_verbose", False))
    lightweight_evaluation = bool(getattr(args, "evaluation_lightweight", False))
    evaluation_template = None
    if lightweight_evaluation:
        from anndata import AnnData

        evaluation_template = AnnData(
            X=np.empty((adata.n_obs, 0), dtype=np.float32),
            obs=adata.obs.copy(),
        )
        if "spatial" in adata.obsm:
            evaluation_template.obsm["spatial"] = np.asarray(
                adata.obsm["spatial"]
            ).copy()

    def cluster_for_evaluation(adata_for_evaluation, quiet):
        requested_kwargs = {
            "device": device,
            "refinement": True,
            "quiet": quiet,
            "cluster_method": getattr(args, "cluster_method", "mclust"),
            "cluster_fused_weight": getattr(args, "cluster_fused_weight", 1.0),
            "cluster_struct_weight": getattr(args, "cluster_struct_weight", 1.0),
            "cluster_sem_weight": getattr(args, "cluster_sem_weight", 0.5),
            "cluster_fused_pca_dim": getattr(args, "cluster_fused_pca_dim", 27),
            "cluster_struct_pca_dim": getattr(args, "cluster_struct_pca_dim", 6),
            "cluster_sem_pca_dim": getattr(args, "cluster_sem_pca_dim", 3),
            "refinement_radius": getattr(args, "refinement_radius", 50),
        }
        return function.clusting_for_plot(
            adata_for_evaluation, model, graph, x, **requested_kwargs
        )

    epoch_iter = tqdm(range(args.max_epoch), ascii=True)
    for epoch in epoch_iter:
        model.train()
        loss_output = model(graph, x, epoch=epoch, return_details=True)
        loss = loss_output["loss"]

        loss_dict = {
            name: value.detach().item() if isinstance(value, torch.Tensor) else value
            for name, value in loss_output.items()
        }
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        epoch_iter.set_description(
            "# Epoch {}: loss={:.4f} rec={:.4f} align={:.4f} contrast={:.4f} gate={:.4f} prune={}/{}({:.2%}) sem_edges={}".format(
                epoch,
                loss_dict["loss"],
                loss_dict["loss_rec"],
                loss_dict["loss_align"],
                loss_dict["loss_contrast"],
                loss_dict["view_fusion_gate"],
                loss_dict["pruned_edges"],
                loss_dict["total_edges"],
                loss_dict["prune_ratio"],
                loss_dict["semantic_view_edges"],
            )
        )
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % monitor_interval == 0 or epoch == 0:
            print(_format_train_status(epoch + 1, loss_dict))

        if (epoch + 1) % eval_interval == 0:
            eval_adata = cluster_for_evaluation(
                evaluation_template.copy() if lightweight_evaluation else adata.copy(),
                quiet=not eval_verbose,
            )
            ari = eval_adata.uns.get("ARI", 0.0)
            nmi = eval_adata.uns.get("NMI", 0.0)
            sc_score = eval_adata.uns.get("SC", float("nan"))
            db_score = eval_adata.uns.get("DB", float("nan"))
            completeness = eval_adata.uns.get("completeness", 0.0)
            homogeneity = eval_adata.uns.get("homogeneity", 0.0)
            v_measure = eval_adata.uns.get("v_measure", 0.0)
            acc = eval_adata.uns.get("ACC", float("nan"))
            f1 = eval_adata.uns.get("F1", float("nan"))
            last_eval_metrics = {
                "epoch": epoch + 1,
                "ARI": ari,
                "NMI": nmi,
                "SC": sc_score,
                "DB": db_score,
                "completeness": completeness,
                "homogeneity": homogeneity,
                "v_measure": v_measure,
                "ACC": acc,
                "F1": f1,
            }
            evaluation_trace.append(dict(last_eval_metrics))

            has_ground_truth = bool(
                eval_adata.uns.get("has_ground_truth", "ARI" in eval_adata.uns)
            )
            if not has_ground_truth:
                selection_score = _unsupervised_selection_score(sc_score)
                is_best = best_model_state is None or selection_score > best_sc
                if is_best:
                    best_sc = selection_score
                    best_model_state = clone_model_state(model)
                    best_adata = eval_adata.copy()
                    best_epoch = epoch + 1
                if is_best or (epoch + 1) % eval_report_interval == 0:
                    print(
                        _format_unsupervised_eval_status(
                            epoch + 1, sc_score, db_score, best_sc, is_best
                        )
                    )

            else:
                is_best = ari > best_ari
                if is_best:
                    best_ari = ari
                    best_model_state = clone_model_state(model)
                    best_adata = eval_adata.copy()
                    best_epoch = epoch + 1
                if is_best or (epoch + 1) % eval_report_interval == 0:
                    print(_format_eval_status(epoch + 1, ari, nmi, best_ari, is_best))
            if show_plots:
                import matplotlib.pyplot as plt
                import scanpy as sc

                sc.pl.embedding(
                    eval_adata,
                    basis="spatial",
                    color="domain",
                    show=False,
                    title=f"Epoch {epoch + 1}",
                )
                plt.savefig(result_path / f"seed_{seed}_epoch_{epoch + 1}.png", dpi=300)
                plt.show()
                plt.close()

    if best_model_state is not None and best_adata is not None:
        model.load_state_dict(best_model_state)

        if not bool(best_adata.uns.get("has_ground_truth", "ARI" in best_adata.uns)):
            best_sc_value = best_adata.uns.get("SC", float("nan"))
            best_db_value = best_adata.uns.get("DB", float("nan"))
            print(
                "[Best] epoch={} SC={:.4f} DB={:.4f}".format(
                    best_epoch, float(best_sc_value), float(best_db_value)
                )
            )
            if last_eval_metrics is not None:
                print(
                    "[Last] epoch={} SC={:.4f} DB={:.4f}".format(
                        last_eval_metrics["epoch"],
                        float(last_eval_metrics["SC"]),
                        float(last_eval_metrics["DB"]),
                    )
                )
        else:
            best_nmi = best_adata.uns.get("NMI", 0.0)
            print(
                "[Best] epoch={} ARI={:.4f} NMI={:.4f}".format(
                    best_epoch, best_ari, best_nmi
                )
            )
            if last_eval_metrics is not None:
                print(
                    "[Last] epoch={} ARI={:.4f} NMI={:.4f}".format(
                        last_eval_metrics["epoch"],
                        float(last_eval_metrics["ARI"]),
                        float(last_eval_metrics["NMI"]),
                    )
                )

        final_adata = best_adata
    else:
        print("No best model found, using final epoch model")
        final_adata = cluster_for_evaluation(adata, quiet=False)

    trace_fields = (
        "epoch",
        "ARI",
        "NMI",
        "SC",
        "DB",
        "completeness",
        "homogeneity",
        "v_measure",
        "ACC",
        "F1",
    )
    final_adata.uns["evaluation_trace"] = {
        field: np.asarray(
            [row.get(field, np.nan) for row in evaluation_trace],
            dtype=np.float64,
        )
        for field in trace_fields
    }
    final_adata.uns["static_graph_initialization"] = dict(static_graph_init_stats)

    if not bool(final_adata.uns.get("has_ground_truth", True)):
        return_score = final_adata.uns.get("SC", best_sc)
    else:
        return_score = final_adata.uns["ARI"]

    return return_score, final_adata, function
