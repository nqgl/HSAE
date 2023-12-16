import torch
from dataclasses import dataclass
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, List, Union


@dataclass
class ToyModelConfig:
    d_data :int
    n_features :int
    features_per_round_negative :Optional[List[int]] = None
    features_per_round :Union[int, List[int]] = 10
    num_correlation_rounds :int = 1
    batch_size :int = 1
    device :str = "cuda"

class ToyModel:
    def __init__(self, cfg :ToyModelConfig):
        self.cfg = cfg
        features = torch.randn(cfg.n_features, cfg.d_data).to(cfg.device)
        self.features = F.normalize(features, dim=-1).to(cfg.device)
        self.f_means = torch.randn(cfg.n_features).to(cfg.device)
        self.f_probs = torch.rand(cfg.n_features).to(cfg.device)
        self.f_stds = torch.rand(cfg.n_features).to(cfg.device)
        self.correlations = [torch.randn(cfg.n_features, cfg.n_features).to(cfg.device) for _ in range(cfg.num_correlation_rounds)]
        self.features_per_round = [cfg.features_per_round] * len(self.correlations) if isinstance(cfg.features_per_round, int) else cfg.features_per_round
        self.features_per_round_negative = self.features_per_round if cfg.features_per_round_negative is None else cfg.features_per_round_negative
        self.features_first_round = 10
        # binary vs magnitude correalations

    @torch.no_grad()
    def next(self):
        return self.get_sample(self.cfg.batch_size)
        
    @torch.no_grad()
    def get_sample(self, batch):
        active = torch.zeros(batch, self.cfg.n_features).to(self.cfg.device)
        probs = self.f_probs.unsqueeze(0).expand(batch, self.cfg.n_features)
        indicies = self.activated_features(probs, self.features_first_round)
        active.scatter_(1, indicies, 1)

        for i in range(len(self.correlations)):
            corr = self.correlations[i]

            probs_i = F.relu(active @ corr)
            indicies = self.activated_features(probs_i, self.features_per_round[i])

            neg_indicies = self.activated_features(F.relu(-1 * active @ corr), self.features_per_round_negative[i])
            active.scatter_(1, indicies, 1)
            active.scatter_(1, neg_indicies, 0)

        feature_magnitudes = self.f_means + torch.randn(batch, self.cfg.n_features).to(self.cfg.device) * self.f_stds
        features = feature_magnitudes.unsqueeze(-1) * active.unsqueeze(-1) * self.features
        x = features.sum(dim=1)
        return x
    
    @torch.no_grad()
    def activated_features(self, probs, num_samples):
        return torch.multinomial(probs, num_samples, replacement=True)
    


    # def update_by_magnitude(self):
    #     active = torch.zeros(batch, self.cfg.n_features)
    #     indicies = self.activated_features(self.f_probs)
    #     active[indicies] = 1
    #     # feature_magnitudes = 
    #     for corr in self.correlations:
    #         probs_i = F.relu(active @ corr)
    #         # probs_i_m = F.relu(active_magnitudes @ corr_m)
    #         indicies = self.activated_features(probs_i)
    #         # feature_magnitudes += f(active @ magnitude_update_matrix_binary)
    #         #         + f(active_magnitudes @ magnitude_update_matrix_by_magnitude)
    #         active[indicies] = 1


    # def get_sample(self):
    #     active = self.sample_features(self.f_probs)
    #     for corr in self.correlations:
    #         probs_i = active @ corr
    #         active += self.sample_features(probs_i)
        
        


    # def sample_features(self, probs):
    #     return F.relu(torch.rand(probs.shape) - probs)


#%%
    

def main():
    cfg = ToyModelConfig(8, 8, features_per_round=1, num_correlation_rounds=3)
    
    model = ToyModel(cfg)

    model.features = torch.eye(cfg.n_features)
    model.f_means = torch.arange(cfg.n_features) + 1
    model.f_probs = torch.zeros(cfg.n_features)
    model.f_probs[0] = 1
    model.f_stds = torch.zeros(cfg.n_features)
    model.correlations = [torch.zeros(cfg.n_features, cfg.n_features) for _ in range(3)]
    model.correlations[0][0, 5] = 1
    model.correlations[0][0, 4] = 1

    model.correlations[0][0, 7] = -1
    model.correlations[1][5, 1] = 1
    model.correlations[1][4, 2] = 1
    model.correlations[1][0, 7] = -1

    model.correlations[2][0, 3] = -1
    model.correlations[2][1, 5] = -100
    model.correlations[2][0, 7] = 1


    print(model.get_sample(12))


if __name__ == "__main__":
    main()
# %%
