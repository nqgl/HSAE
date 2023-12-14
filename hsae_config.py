
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from sae_config import AutoEncoderConfig

@dataclass
class HierarchicalAutoEncoderConfig(AutoEncoderConfig):
    geometric_median_dataset: bool = False
    features_per_sae_per_layer = List[int] = [None, 32]
    num_layers :int = 2
    sublayer_cfgs :List["HierarchicalAutoEncoderLayerConfig"] = None
    sublayers_train_on_error :bool = False
    layer_cfg_params :dict = {}
    layer_cfg_params_per_layer :List[Dict] = [{}]

    def __post_init__(self):
        super().__post_init__()

        if self.features_per_sae_per_layer[0] is None:
            self.features_per_sae_per_layer[0] = self.dict_size

        if self.sublayer_cfgs is None:
            num_features_at_layer = 1 * self.dict_size
            for i in range(0, self.num_layers - 1):
                cfg_params = {**self.layer_cfg_params, **self.layer_cfg_params_per_layer[i]}
                hlcfg = HierarchicalAutoEncoderLayerConfig(
                    layer_index = i,
                    dict_size = self.features_per_sae_per_layer[i],
                    num_saes = num_features_at_layer,

                )
                num_saes *= hlcfg.dict_size

            
                

@dataclass
class HierarchicalAutoEncoderLayerConfig():
    layer_index :int

    

    # calculated numbers
    num_saes: int = None               # alt name layer_width
    dict_size :int = None

    # normal params
    nonlinearity :tuple = ("relu", {})
    num_to_resample :int = 128
    data_rescale :float = 1.0
    ## lr params
    lr :int = 3e-4
    l1_coeff :Union[float, List[float]] = 8e-4
    beta1 :float = 0.9
    beta2 :float = 0.99



    def __init__(self, cfg :HierarchicalAutoEncoderConfig, **kwargs):

        super().__init__(**kwargs)

