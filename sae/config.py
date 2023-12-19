import transformer_lens.utils as utils


from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class AutoEncoderConfig: #TODO some of these types are wrong. possibly some fields are unused, too
    
    
    lr :int = 3e-4
    l1_coeff :Union[float, List[float]] = 8e-4
    beta1 :float = 0.9
    beta2 :float = 0.99
    
    # dimensionality
    d_dict :int = None
    d_data :int = -1
    dict_mult :int = 32
    
    # LM target
    layer :int = 0
    model_name :str = "gelu-2l"
    site :str = "" # z?
    flatten_heads :bool = False
    
    # hw usage
    enc_dtype :str = "fp32"
    device :str = "cuda"
    act_name :str = None
    
    # buffer
    # tunable
    batch_size :int = 256
    buffer_mult :int = 10000
    buffer_refresh_ratio :float = 0.1
    # 
    num_tokens :int = int(2e9)
    seq_len :int = 128
    # calculated
    model_batch_size : int = None
    buffer_size :int = None
    buffer_batches :int = None
    
    # modifications
    nonlinearity :tuple = ("relu", {})
    cosine_l1 :Optional[Dict] = None
    # resampling
    gram_shmidt_trail :int = 5000
    num_to_resample :int = 128
    # rescaling
    data_rescale :float = 10
    scale_in_forward :bool = True

    # bookkeeping
    experimental_type: Optional[str] = None
    name :str = None
    seed :int = 49

    # optimization
    subshuffle :Optional[int] = None

    def __post_init__(self):
        print("Post init")
        self.post_init_cfg()

    def post_init_cfg(self):
        self.model_batch_size = self.batch_size // self.seq_len * 16
        self.buffer_size = self.batch_size * self.buffer_mult
        self.buffer_batches = self.buffer_size // self.seq_len
        self.act_name = utils.get_act_name(self.site, self.layer)
        self.d_dict = int(self.d_data * self.dict_mult)
        self.name = f"{self.model_name}_{self.layer}_{self.d_dict}_{self.site}"
        return self

