from configuration import (
    build_args as build_args,
    build_args_known as build_args_known,
    load_best_configs as load_best_configs,
)
from training import train_single_sample as train_single_sample

__all__ = [
    "build_args",
    "build_args_known",
    "load_best_configs",
    "train_single_sample",
    "set_random_seed",
    "get_current_lr",
    "create_activation",
    "create_norm",
    "create_optimizer",
    "TBLogger",
    "save_and_visualize",
    "save_and_visualize_visium",
    "fix_rpy2_encoding",
]
import os
import random
import logging
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def identity_norm(x):
    def func(x):
        return x

    return func


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        return None


def create_optimizer(
    opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None
):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")

    return optimizer


class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


def save_and_visualize(adata, function, sample, dataset, save_path="./results/"):
    """Save clustering results and generate visualization files."""
    from pathlib import Path

    result_save_path = Path(save_path) / dataset
    result_save_path.mkdir(parents=True, exist_ok=True)

    h5ad_file = result_save_path / f"{sample}.h5ad"
    print(f"\nSaving results to {h5ad_file}...")
    adata.write(str(h5ad_file), compression="gzip")

    sample_dir = result_save_path / sample
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")
    function.create_visium_style_plots(adata, str(sample_dir))

    print(f"Results saved to {result_save_path}")
    print(f"Plots saved to {sample_dir}")


def save_and_visualize_visium(
    final_adata, function, sample, dataset="DLPFC", base_path="./results/"
):
    """Save and plot the same selected clustering result."""
    return save_and_visualize(final_adata, function, sample, dataset, base_path)


def fix_rpy2_encoding():
    """Set R's text encoding without patching rpy2 internals."""
    os.environ.setdefault("R_ENCODING", "UTF-8")
