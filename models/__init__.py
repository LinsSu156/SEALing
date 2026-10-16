from inspect import Parameter, signature

from .edcoder import PreModel


def build_model(args):
    """Build the model using shared constructor defaults and argument aliases."""
    argument_names = {
        "in_dim": "num_features",
        "num_hidden": "num_hidden",
        "num_layers": "num_layers",
        "num_dec_layers": "num_dec_layers",
        "nhead": "num_heads",
        "nhead_out": "num_out_heads",
        "activation": "activation",
        "feat_drop": "in_drop",
        "attn_drop": "attn_drop",
        "negative_slope": "negative_slope",
        "residual": "residual",
        "norm": "norm",
        "alpha_l": "alpha_l",
    }
    parameters = {
        name: getattr(args, source) for name, source in argument_names.items()
    }
    for name, parameter in signature(PreModel).parameters.items():
        if name not in parameters and parameter.default is not Parameter.empty:
            parameters[name] = getattr(args, name, parameter.default)
    parameters.update(
        zero_init=args.dataset in ("cora", "pubmed", "citeseer", "10x"),
    )
    return PreModel(**parameters)
