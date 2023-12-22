import torch
from dataclasses import dataclass
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, List, Union


@dataclass
class ToyModelConfig:
    d_data: int
    n_features: int
    initial_features: int = 10
    features_per_round_negative: Optional[List[int]] = None
    features_per_round: Union[int, List[int]] = 10
    num_correlation_rounds: int = 1
    batch_size: int = 1
    device: str = "cuda"
    seed: Optional[int] = None
    blank_correlations: bool = True
    correlation_drop: float = 0
    source_prob_drop: float = 0
    replacement: bool = True
    max_initial_feature_frequency_disparity = 100


class ToyModel:
    def __init__(self, cfg: ToyModelConfig):
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        self.cfg = cfg
        features = torch.randn(cfg.n_features, cfg.d_data).to(cfg.device)
        self.features = F.normalize(features, dim=-1).to(cfg.device)
        self.f_means = torch.randn(cfg.n_features).to(cfg.device)

        self.f_stds = torch.rand(cfg.n_features).to(cfg.device)

        self.f_probs = F.dropout(
            torch.rand(cfg.n_features).to(cfg.device)
            + (1 / cfg.max_initial_feature_frequency_disparity),
            cfg.source_prob_drop,
            training=True,
        ) * (1 - cfg.source_prob_drop)

        if cfg.blank_correlations:
            self.correlations = [
                torch.zeros(cfg.n_features, cfg.n_features).to(cfg.device)
                for _ in range(cfg.num_correlation_rounds)
            ]

        else:
            self.correlations = [
                F.dropout(
                    torch.randn(cfg.n_features, cfg.n_features).to(cfg.device),
                    cfg.correlation_drop,
                    training=True,
                )
                * (1 - cfg.correlation_drop)
                for _ in range(cfg.num_correlation_rounds)
            ]
        self.features_per_round = (
            [cfg.features_per_round] * len(self.correlations)
            if isinstance(cfg.features_per_round, int)
            else cfg.features_per_round
        )
        self.features_per_round_negative = (
            self.features_per_round
            if cfg.features_per_round_negative is None
            else cfg.features_per_round_negative
        )
        self.features_per_round_negative = (
            [self.features_per_round_negative] * len(self.correlations)
            if isinstance(self.features_per_round_negative, int)
            else self.features_per_round_negative
        )
        # binary vs magnitude correalations

    @torch.no_grad()
    def next(self):
        return self.get_sample(self.cfg.batch_size)

    @torch.no_grad()
    def get_sample(self, batch):
        active = torch.zeros(batch, self.cfg.n_features).to(self.cfg.device)
        probs = self.f_probs.unsqueeze(0).expand(batch, self.cfg.n_features)
        activate = self.activated_features(probs, self.cfg.initial_features)
        active[activate] = 1
        # print(active.sum())
        for i in range(len(self.correlations)):
            corr = self.correlations[i]

            probs_i = F.relu(active @ corr)
            # print(probs_i)
            activate = self.activated_features(probs_i, self.features_per_round[i])

            deactivate = self.activated_features(
                F.relu(-1 * active @ corr), self.features_per_round_negative[i]
            )
            active[activate] = 1
            active[deactivate] = 0
            # print(active)

        feature_magnitudes = (
            self.f_means
            + torch.randn(batch, self.cfg.n_features).to(self.cfg.device) * self.f_stds
        )
        features = (
            feature_magnitudes.unsqueeze(-1) * active.unsqueeze(-1) * self.features
        )
        x = features.sum(dim=1)
        # print(x.shape)
        # print("average_activations:", active.sum() / batch, len(self.correlations))
        return x

    @torch.no_grad()
    def activated_features(self, probs, num_samples):
        """
        probs: (batch, n_features)
        num_samples: int

        returns: (batch, n_features) bool


        """
        # print(torch.sum(probs))
        if num_samples == 0:
            return torch.zeros(0, device=self.cfg.device, dtype=torch.bool)
        nonzero_axes = probs.sum(dim=-1) > 0
        # print(probs.shape)
        # print(nonzero_axes)
        if nonzero_axes.shape[0] == 0:
            return torch.zeros(0, device=self.cfg.device, dtype=torch.bool)
        # print(probs[nonzero_axes].sum())
        # print("probs", probs.shape)
        # print("probs", probs)
        # print("nonzero_axes", nonzero_axes.shape)
        # print("probs[nonzero_axes]", probs[nonzero_axes].shape)
        sampled = torch.multinomial(
            probs[nonzero_axes], num_samples, replacement=self.cfg.replacement
        )
        activate = torch.zeros(
            probs.shape[0],
            self.cfg.n_features,
            device=self.cfg.device,
            dtype=torch.bool,
        )
        activate[nonzero_axes] = activate[nonzero_axes].scatter_(1, sampled, True)
        # print("activate", activate.shape)
        # print("sampled", sampled.shape)
        # print("activate", activate)
        # print("sampled", sampled)

        return activate

    def add_hierarchical_feature(self, correlation_matrix, src, dest, weight):
        if isinstance(dest, tuple):
            dest = torch.tensor(dest, device=correlation_matrix.device)
        # correlation_matrix[:, dest] = 0
        correlation_matrix[src, dest] = weight
        if weight > 0:
            self.f_probs[dest] = 0

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


# %%


def main():
    cfg = ToyModelConfig(
        8,
        8,
        features_per_round=1,
        num_correlation_rounds=3,
    )

    torch.set_default_device(torch.device("cuda"))
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
    model.correlations[2][1, 5] = -1
    model.correlations[2][0, 7] = 1

    print(model.get_sample(12))


if __name__ == "__main__":
    main()
# %%
