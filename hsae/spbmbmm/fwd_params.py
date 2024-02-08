import torch
import torch.nn as nn
from torch import jit

from typing import Optional
from dataclasses import dataclass, field

from unpythonic import box


@dataclass
class ForwardOptions:
    batches: jit.Final[int]
    d_data: jit.Final[int]
    d_dict: jit.Final[int]
    n_sae: jit.Final[int]
    sub_b_dec: jit.Final[bool] = False
    # scale_b_dec: jit.Final[bool] = True
    scale_acts: jit.Final[bool] = False
    # scale_b_enc: jit.Final[bool] = True
    # cache_acts: jit.Final[bool] = True
    # norm_acts: jit.Final[bool] = True
    norm_gate_before_scaling_acts: jit.Final[bool] = False
    device: str = "cuda"
    nonlinearity: nn.Module = torch.relu


@dataclass
class CacheBoxes:
    acts: Optional[box] = None
    flat_acts: Optional[box] = None
    l1: box = field(default_factory=box)  # TODO implement caching these
    l0: box = field(default_factory=box)  # TODO implement caching these
