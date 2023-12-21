import torch
import torch.nn.functional as F


def re_init_neurons_gram_shmidt(x_diff):
    n_reset = x_diff.shape[0]
    v_orth = torch.zeros_like(x_diff)
    # print(x_diff.shape)
    v_orth[0] = F.normalize(x_diff[0], dim=-1)
    for i in range(1, n_reset):
        v_bar = v_orth[:i].sum(0)
        v_bar = v_bar / v_bar.norm(dim=-1)

        v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
        # print(v_.shape)
        v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
    return v_orth


def re_init_neurons_gram_shmidt_precise(x_diff):
    n_reset = x_diff.shape[0]
    v_orth = torch.zeros_like(x_diff)
    # print(x_diff.shape)
    # v_orth[0] = F.normalize(x_diff[0], dim=-1)
    for i in range(n_reset):
        v_orth[i] = x_diff[i]
        for j in range(i):
            v_orth[i] -= (
                torch.dot(v_orth[j], v_orth[i])
                * v_orth[j]
                / torch.dot(v_orth[j], v_orth[j])
            )
        v_orth[i] = F.normalize(v_orth[i], dim=-1)
        # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
        # # print(v_.shape)
        # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
    return v_orth


def re_init_neurons_gram_shmidt_precise2(x_diff):
    n_reset = x_diff.shape[0]
    v_orth = x_diff.clone()
    # print(x_diff.shape)
    # v_orth[0] = F.normalize(x_diff[0], dim=-1)
    for i in range(1, n_reset):
        v_orth[:i] -= (v_orth[:i] @ v_orth[i] / (v_orth[i] @ v_orth[i])).unsqueeze(
            -1
        ) * v_orth[i]

    return v_orth


x_d = torch.randn(100, 20, dtype=torch.float64)
u = re_init_neurons_gram_shmidt_precise(x_d)
q = u @ u.T
print(q)
print(q.sum(dim=0))
print(u.norm(dim=-1))
