import transformer_lens.utils as utils


from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class AutoEncoderConfig: #TODO some of these types are wrong. possibly some fields are unused, too
    seed :int = 49
    batch_size :int = 256
    buffer_mult :int = 10000
    lr :int = 3e-4
    num_tokens :int = int(2e9)
    l1_coeff :Union[float, List[float]] = 8e-4
    beta1 :float = 0.9
    beta2 :float = 0.99
    dict_mult :int = 32
    seq_len :int = 128
    layer :int = 0
    enc_dtype :str = "fp32"
    model_name :str = "gelu-2l"
    site :str = "" # z?
    device :str = "cuda"
    remove_rare_dir :bool = False
    act_size :int = -1
    flatten_heads :bool = True
    model_batch_size : int = None
    buffer_size :int = None
    buffer_batches :int = None
    act_name :str = None
    dict_size :int = None
    name :str = None
    buffer_refresh_ratio :float = 0.1
    nonlinearity :tuple = ("relu", {})
    cosine_l1 :Optional[Dict] = None
    experimental_type: Optional[str] = None
    gram_shmidt_trail :int = 5000
    num_to_resample :int = 128
    data_rescale :float = 1.0

    def __post_init__(self):
        print("Post init")
        self.post_init_cfg()

    def post_init_cfg(self):
        self.model_batch_size = self.batch_size // self.seq_len * 16
        self.buffer_size = self.batch_size * self.buffer_mult
        self.buffer_batches = self.buffer_size // self.seq_len
        self.act_name = utils.get_act_name(self.site, self.layer)
        self.dict_size = self.act_size * self.dict_mult
        self.name = f"{self.model_name}_{self.layer}_{self.dict_size}_{self.site}"
        return self

