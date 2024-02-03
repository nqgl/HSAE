import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, List, Union
import sae.novel_nonlinearities as novel_nonlinearities
import os
import json
from dataclasses import dataclass
import v1solo.orthonormal as orthonormal


@dataclass
class AutoEncoderConfig:
    lr: Union[float, torch.Tensor]
    d_act: int
    d_dict: int
    l1_coeff: Optional[Union[float, torch.Tensor]]
    device: str = "cuda"
    ae_id: str = "no-id"
    l0l1: bool = True
    l0l1_coeff: float = 0.01
    lo_coeff: float = 10
    l1_half_coeff: float = 0


def ratio_nice_str(f: float):
    r = f.as_integer_ratio()
    while r[1] > 100 or r[0] > 100:
        r = (r[0] // 2, r[1] // 2)

    return f"{r[0]}-{r[1]}"


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        self.project_in = nn.Linear(cfg.d_act, cfg.d_act, bias=False)
        self.encoder = nn.Linear(cfg.d_act, cfg.d_dict, bias=False)
        self.dd_lin = nn.Linear(cfg.d_dict, cfg.d_dict, bias=False)
        # self.decoder = orthonormal.OrthLinearStack(cfg.d_dict, cfg.d_act, bias=False)
        # self.project_out = orthonormal.OrthLinearStack(cfg.d_act, cfg.d_act, bias=False)
        self.decoder = orthonormal.RowNormPenalizedLinear(
            cfg.d_dict, cfg.d_act, bias=False
        )
        self.project_out = orthonormal.OrthPenalizedLinear(
            cfg.d_act, cfg.d_act, bias=False
        )
        # self.project_out = orthonormal.NoOpModule()

        self.cfg: AutoEncoderConfig = cfg
        self.alive_neurons = torch.zeros(
            cfg.d_dict, dtype=torch.bool, device=cfg.device
        )
        self.acts_cached = None
        self.l1_coeffs = cfg.l1_coeff
        self.b_d = nn.Parameter(torch.zeros(cfg.d_act, device=cfg.device))
        self.b_e = nn.Parameter(torch.zeros(cfg.d_dict, device=cfg.device))

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = x - self.b_d
        x = self.project_in(x)
        x = self.encoder(x)
        x = self.dd_lin(x)
        # x = (gradcool_functions.undying_relu_2phases(x + self.b_e, l = 0.01) \

        # x = gradcool_functions.undying_relu(x + self.b_e, l = 0.01, k = 0.01)

        x = novel_nonlinearities.undying_relu(x + self.b_e, l=0.01, k=1)
        # x = F.relu(x + self.b_e)
        # x = gradcool_functions.undying_relu_2phase_leaky_gradient(x + self.b_e, l = 0.0001)
        self.acts_cached = x
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = self.project_out(x)
        x = x + self.b_d
        return x

    # def norm_decoder(self):
    #     self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=-1)

    def get_l1_loss(self, acts=None):
        acts = self.acts_cached if acts is None else acts
        l1 = torch.mean(torch.abs(acts), dim=-2)
        assert l1.shape[-1] == self.cfg.d_dict

        return l1

    def get_l0_loss(self, acts=None):
        acts = self.acts_cached if acts is None else acts
        l0 = (acts != 0).float().mean(dim=-2)
        return l0

    def get_l0l1_loss(self, acts=None, n=1):
        if self.cfg.l0l1_coeff == 0:
            return torch.tensor(0, device=self.cfg.device)
        acts = self.acts_cached if acts is None else acts
        l0 = (acts.detach() != 0).float().sum(dim=-1)
        l0_multiplier = torch.max(torch.tensor(0, device=self.cfg.device), l0 - n)
        l1 = torch.mean(torch.abs(acts), dim=-1)
        return torch.mean(l0_multiplier * l1) * self.cfg.l0l1_coeff

    def get_l1_half_loss(self, acts=None):
        if self.cfg.l1_half_coeff == 0:
            return torch.tensor(0, device=self.cfg.device)
        else:
            acts = self.acts_cached if acts is None else acts
            l1 = torch.mean(torch.abs(acts), dim=-2)
            assert l1.shape[-1] == self.cfg.d_dict
            eps = torch.finfo(l1.dtype).eps * 4
            l1_half = torch.sqrt(l1 + eps)

            return torch.mean(l1_half) * self.cfg.l1_half_coeff

    def penalties(self, lo=True):
        if lo:
            return (
                self.decoder.penalty() + self.project_out.penalty()
            ) * self.cfg.lo_coeff, (self.get_l1_loss() * self.l1_coeffs).sum()
        else:
            return (
                torch.tensor(0, device=self.cfg.device),
                (self.get_l1_loss() * self.l1_coeffs).sum(),
            )

    def save(self, directory, version, stats=None):
        # TODO add width multiuplier to name

        l1_coeff = self.cfg.l1_coeff
        lr = self.cfg.lr

        if isinstance(l1_coeff, torch.Tensor):
            l1_str = "tensor"
        else:
            l1_str = ratio_nice_str(l1_coeff)
        folder_str = f"sae_v_{version}_id_{self.cfg.ae_id}"
        if not os.path.exists(os.path.join(directory, folder_str)):
            os.makedirs(os.path.join(directory, folder_str))
        modelfolder = os.path.join(directory, folder_str)
        iter = (
            max(
                [
                    int(f.split(".")[1])
                    for f in os.listdir(modelfolder)
                    if f.startswith("model")
                ],
                default=0,
            )
            + 1
        )
        model_name = os.path.join(modelfolder, f"model.{iter}.pt")
        config_name = os.path.join(modelfolder, f"config.{iter}.json")
        torch.save(self, model_name)
        with open(config_name, "w") as f:
            json.dump(self.cfg.__dict__, f)
        if stats is not None:
            stats_name = os.path.join(modelfolder, f"stats.{iter}.json")
            with open(stats_name, "w") as f:
                if isinstance(stats, str):
                    f.write(stats)
                else:
                    json.dump(stats, f)
        return iter

    @classmethod
    def load(cls, directory, folder_str, iter=None):
        # folder_str = f"sae_id_{l1_str}_{ratio_nice_str(lr)}_{str(version)}"
        # if isinstance(l1_coeff, torch.Tensor):
        #     l1_str = "tensor"
        # else:
        #     l1_str = ratio_nice_str(l1_coeff)
        if iter is None:
            iter = max(
                [
                    int(f.split(".")[1])
                    for f in os.listdir(os.path.join(directory, folder_str))
                    if f.startswith("model")
                ]
            )
        model_name = os.path.join(directory, folder_str, f"model.{iter}.pt")
        config_name = os.path.join(directory, folder_str, f"config.{iter}.json")
        with open(config_name, "r") as f:
            cfg = AutoEncoderConfig(**json.load(f))
        ae = torch.load(model_name)
        ae.cfg = cfg
        return ae

    @staticmethod
    def load_stats(directory, folder_str, iter=None):
        if iter is None:
            iter = max(
                [
                    int(f.split(".")[1])
                    for f in os.listdir(os.path.join(directory, folder_str))
                    if f.startswith("model")
                ]
            )
        stats_name = os.path.join(directory, folder_str, f"stats.{iter}.json")
        with open(stats_name, "r") as f:
            stats = json.load(f)
        return stats


def find_maximizing_vec_length(ae, d_dict, device):
    v_2max = F.normalize(torch.randn(10, d_dict, device=device), dim=-1)
    v_2max.requires_grad = True
    # m = nn.Linear(d_dict, d_dict, bias=False).to(device)
    # mo = torch.nn.utils.parametrizations.orthogonal(m, "weight")

    # optim = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.75)
    # ae.train(False)
    optim = torch.optim.SGD([v_2max], lr=0.05, momentum=0.9)
    for i in range(2000):
        # v = m(v_2max)
        loss = -1 * torch.mean(ae.decode(F.normalize(v_2max, dim=-1)).norm(dim=-1))
        if i % 100 == 0:
            print(-1 * loss.item())

        # loss += torch.mean(torch.pow(v_2max.norm(dim=-1) - 1, 2))
        # loss = torch.mean(torch.pow(ae.decode(v) - 1, 2))
        loss.backward()
        optim.step()
        optim.zero_grad()
    print("norm", ae.decode(F.normalize(v_2max, dim=-1)).norm(dim=-1))
    print(ae.cfg.l1_coeff)
    vv = ae.decode(F.normalize(v_2max, dim=-1)).norm(dim=-1)
    return torch.max(vv).item()


def show_heatmap_of_similarity(m, ae, encoder=True):
    import seaborn as sns
    import torch
    import matplotlib.pyplot as plt

    v = torch.eye(ae.cfg.d_dict, device="cuda")
    f = ae.decode(v)
    # features, d_act -> features, d_dict
    if encoder:
        features_similarity = ae.encode(m)
    else:
        m = torch.nn.functional.normalize(m, dim=-2)
        f = torch.nn.functional.normalize(f, dim=-2)
        features_similarity = m @ f.T
        features_similarity = torch.abs(features_similarity)
    features_similarity = F.normalize(features_similarity, dim=-1)
    if features_similarity.shape[-1] > features_similarity.shape[-2]:
        features_similarity = features_similarity.transpose(-1, -2)
    max_args = features_similarity.argmax(dim=-1)
    argsort_max = max_args.argsort()
    features_similarity = features_similarity[argsort_max, :]
    plt.clf()
    sns.heatmap(features_similarity.cpu().detach().numpy())
    plt.pause(2)


def main():
    import time

    # import matplotlib.pyplot as plt
    device = torch.device("cuda")
    d_act = 400
    n_features = 2000
    d_dict = d_act * 8
    avg_n_features = 10
    l1_variability_radius = 0
    l1_coeffs = torch.linspace(
        1 - l1_variability_radius, 1 + l1_variability_radius, d_dict, device=device
    )
    cfg = AutoEncoderConfig(
        lr=3e-4,
        d_act=d_act,
        d_dict=d_dict,
        l1_coeff=0.2,
        l0l1_coeff=0,
        lo_coeff=100,
        l1_half_coeff=0,
    )
    ae = AutoEncoder(cfg)
    ae.l1_coeffs = l1_coeffs * cfg.l1_coeff
    ae.to(device)
    m = torch.randn(n_features, d_act, device=device) * 5
    # m = F.normalize(m, dim=-2)
    # m = m * 2
    optim = torch.optim.Adam(ae.parameters(), lr=cfg.lr)
    # optim = torch.optim.SGD(ae.parameters(), lr=cfg.lr, momentum=0.96)
    # optim = torch.optim.Adam(ae.parameters(), lr=0.02, weight_decay=0.01)
    t0 = time.time()
    torch.set_printoptions(precision=3)
    n = 5000000
    d = {"max_vec": {}, "l2": {}, "l1": {}, "l0l1": {}, "l0": {}, "lo": {}}
    # plt.ion()
    # plt.show()
    p_feature = avg_n_features / n_features
    p_feature_variability_radius = 0.5
    p_feature_distribution = torch.linspace(
        p_feature * (1 + p_feature_variability_radius),
        p_feature * (1 - p_feature_variability_radius),
        n_features,
        device=device,
    )
    for i in range(n):
        x = torch.rand(250, n_features, device=device)
        b = 100
        # x = F.dropout(x, p=(1 - p_feature), training=True) * (p_feature)
        mask = torch.rand(x.shape, device=device) < p_feature_distribution
        x = x * mask
        # for j in range(n_features // b):
        #     p_feature = p_feature * 0.5
        #     x[j*b : (j + 1) * b] = F.dropout(x[j*b : (j + 1) * b], p=(1 - p_feature), training=True) * (p_feature)

        # x = F.relu(x)
        x_nonzero = torch.count_nonzero(x, dim=-1).float().mean().item()
        x = x @ m
        y = ae(x)
        l2 = torch.mean(torch.pow(y - x, 2))
        lo, l1 = ae.penalties(lo=True)
        l0l1 = ae.get_l0l1_loss(n=1)
        # l0l1 = torch.tensor(0)
        # if i % 10 == 0:
        print(
            l2.item(),
            lo.item() / cfg.lo_coeff,
            l1.item() / cfg.l1_coeff,
            l0l1.item(),
            i,
        )
        print(
            ae.get_l0_loss().sum().item(),
            x_nonzero,
            torch.count_nonzero(x, dim=-1).float().mean().item(),
        )
        loss = l2 * 1 + lo + l1 + l0l1
        loss += ae.get_l1_half_loss()
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i % 100 == 0:
            # cfg.lr = cfg.lr * 0.99
            # optim = torch.optim.Adam(ae.parameters(), lr=cfg.lr)

            d["l2"][i] = l2.item()
            d["l1"][i] = l1.item()
            d["l0l1"][i] = l0l1.item()
            d["l0"][i] = ae.get_l0_loss().sum().item()
            d["lo"][i] = lo.item() / cfg.lo_coeff

        # if i % 3000 == 1500 and i > 0:
        #     show_heatmap_of_similarity(m, ae, encoder = i % 6000 == 1500)
        if i % 6000 == 50 and i > 0:
            window = 30
            # plt.clf()
            # plt.subplot(3, 2, 1)
            # plt.plot(list(d["l0l1"].keys())[-window:], list(d["l0l1"].values())[-window:], label="l0l1")
            # plt.plot(list(d["l0"].keys())[-window:], list(d["l0"].values())[-window:], label="l0")
            # plt.legend()

            # plt.subplot(3, 2, 3)
            # plt.plot(list(d["l2"].keys())[-window:], list(d["l2"].values())[-window:], label="l2")
            # plt.plot(list(d["l1"].keys())[-window:], list(d["l1"].values())[-window:], label="l1")
            # plt.legend()

            # plt.subplot(3, 2, 5)
            # plt.plot(list(d["lo"].keys())[-window:], list(d["lo"].values())[-window:], label="lo")
            # # plt.plot(list(d["max_vec"].keys()), list(d["max_vec"].values()), label="max_vec")
            # plt.legend()

            # plt.subplot(3, 2, 2)
            # plt.plot(list(d["l0l1"].keys()), list(d["l0l1"].values()), label="l0l1")
            # plt.plot(list(d["l0"].keys()), list(d["l0"].values()), label="l0")
            # plt.legend()

            # plt.subplot(3, 2, 4)
            # plt.plot(list(d["l2"].keys()), list(d["l2"].values()), label="l2")
            # plt.plot(list(d["l1"].keys()), list(d["l1"].values()), label="l1")
            # plt.legend()

            # plt.subplot(3, 2, 6)
            # plt.plot(list(d["lo"].keys()), list(d["lo"].values()), label="lo")
            # plt.plot(list(d["max_vec"].keys()), list(d["max_vec"].values()), label="max_vec")
            # plt.legend()

            # plt.draw()
            # plt.pause(1)

        if i % 10000 == 0 and i > 0:
            d["max_vec"][i] = find_maximizing_vec_length(ae, d_dict, device)
            ae.save("/home/g/mats/sae/models", 2, {"d": d, "m": m.tolist(), "i": i})
            optim.zero_grad()
    ae.save("/home/g/mats/sae/models", 2)
    t1 = time.time()
    x = torch.randn(100, d_act, device=device) @ m
    y = ae(x)
    print("l2", torch.mean((y - x).norm(dim=-1)).item())
    print("l0", ae.get_l0_loss(x).float().mean().item())
    v = torch.randn(100, d_dict, device=device)
    v = torch.abs(v)
    v = F.normalize(v, dim=-1)
    print("norm", ae.decoder(v).norm(dim=-1))
    print("norm", ae.decode(v).norm(dim=-1))
    # TODO check orth for just one block
    vb = torch.randn(100, d_act, device=device)
    vb = F.normalize(vb, dim=-1)
    v = torch.zeros(100, d_dict, device=device)
    v[:, :d_act] = vb
    print("norm", ae.decoder(v).shape)
    print(v.shape)

    print("norm_dec", ae.decoder(v).norm(dim=-1))
    print("norm_dec_proj", ae.decode(v).norm(dim=-1))
    print("norm_proj_only", ae.project_out(vb).norm(dim=-1))
    # print("d2norm", ae.decoder(v).norm(dim=-2))
    # print("d2norm", ae.decode(v).norm(dim=-2))
    # print("d2norm_proj_only", ae.project_out(vb).norm(dim=-2))
    print("time", t1 - t0)

    # print("a norm maximizer", ae.decoder(v_2max), v_2max)


if __name__ == "__main__":
    main()
