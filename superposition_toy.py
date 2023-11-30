import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, List




class SuperpositionToyConfig:
    n_features :int
    feature_importances :torch.Tensor
    feature_prob_weights :torch.Tensor
    feature_occurence_sparsity_mean :float
    feature_occurence_sparsity_std :float
    dimensions :int
    correlation_matrix :torch.Tensor
    device :torch.device
    n_binary_features :Optional[int]
    n_uniform_features :Optional[int]
    n_ubiquitous_uniform_features :Optional[int]
    

class SuperpositionToy:
    def __init__(self, cfg:SuperpositionToyConfig):
        self.n_features :int
        self.feature_importances :torch.Tensor
        self.feature_prob_weights :torch.Tensor
        self.feature_
        self.feature_occurence_sparsity_mean :float
        self.feature_occurence_sparsity_std :float
        self.dimension :int
        # self.correlation_matrix :torch.Tensor

        self.features = torch.randn(self.n_features, self.dimension)
        self.device = cfg.device


    def generate(self, n_samples:int, sparsity_override :Optional[Any[List[float], float]] = None) ->torch.Tensor:
        """
        generate samples from the distribution
        """
        if isinstance(sparsity_override, list):
            sparsity = torch.tensor(sparsity_override)
        elif isinstance(sparsity_override, float):
            sparsity = sparsity_override * torch.ones(n_samples) 
        else:
            sparsity = torch.relu(torch.randn(n_samples) * self.feature_occurence_sparsity_std + self.feature_occurence_sparsity_mean - 1) + 1
        sparsity = sparsity.to(self.device)


    def L2_loss(self, x:torch.Tensor, y:torch.Tensor) ->torch.Tensor:
        return torch.mean(torch.pow(x - y, 2) * self.feature_importances)
