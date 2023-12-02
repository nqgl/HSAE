from .. import gradcool_functions
import torch
from functools import partial


nonlinearity_dict = {
    "relu" : torch.nn.functional.relu,
    "undying_relu" : gradcool_functions.undying_relu,
    "undying_relu_2phase" : gradcool_functions.undying_relu_2phase,
}

def cfg_to_nonlinearity(cfg):
    nonlinearity = nonlinearity_dict[cfg.nonlinearity[0]]
    nonlinearity_kwargs = cfg.nonlinearity[1]
    return partial(nonlinearity, **nonlinearity_kwargs)