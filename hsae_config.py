
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from sae_config import AutoEncoderConfig

@dataclass
class HierarchicalAutoEncoderConfig(AutoEncoderConfig):
    geometric_median_dataset: bool = False
    features_per_sae_per_layer :List[Optional[int]] = field(default_factory = lambda :[None, 32])
    num_layers :int = 2
    sublayer_cfgs :List["HierarchicalAutoEncoderLayerConfig"] = field(default_factory= lambda :[])
    sublayers_train_on_error :bool = False
    layer_cfg_params :dict = field(default_factory = lambda :{})
    layer_cfg_params_per_layer :List[Dict] = field(default_factory = lambda :[])
    gate_mode :str = "binary"

    
    def __post_init__(self):
        super().__post_init__()

        if self.features_per_sae_per_layer[0] is None:
            self.features_per_sae_per_layer[0] = self.d_dict

        if len(self.layer_cfg_params_per_layer) < self.num_layers:
            self.layer_cfg_params_per_layer += [{} for _ in range(self.num_layers - len(self.layer_cfg_params_per_layer))]


        if self.sublayer_cfgs == []:
            num_features_at_layer = 1 * self.d_dict
            for i in range(0, self.num_layers - 1):
                cfg_params = {**self.layer_cfg_params, **self.layer_cfg_params_per_layer[i + 1]}
                hlcfg = HierarchicalAutoEncoderLayerConfig(cfg=self,
                    layer_index = i,
                    d_dict = self.features_per_sae_per_layer[i + 1],
                    n_sae = num_features_at_layer,
                    **cfg_params

                )
                num_features_at_layer *= hlcfg.d_dict
                self.sublayer_cfgs.append(hlcfg)
        self.scale_in_forward = False

            
                

@dataclass
class HierarchicalAutoEncoderLayerConfig():
    layer_index :int
    cfg :HierarchicalAutoEncoderConfig

    # calculated numbers
    n_sae :int = None               # alt name layer_width
    d_dict :int = None

    # normal params
    nonlinearity :tuple = ("relu", {})
    num_to_resample :int = 128
    data_rescale :float = 1.0
    ## lr params
    lr :int = 3e-4
    l1_coeff :Union[float, List[float]] = 8e-4
    beta1 :float = 0.9
    beta2 :float = 0.99



    # def __init__(self, cfg :HierarchicalAutoEncoderConfig, **kwargs):

    #     super().__init__(**kwargs)

