import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, List
from gradcool_functions import undying_relu
import os
from dataclasses import dataclass


d = 100
d1 = 200
d2 = 50
matrix = torch.randn(d1, d2)
vecs = torch.randn(100, d1)
m = nn.Linear(d1, d2, bias=False)

nmat0 = F.normalize(m.weight, dim=0)
nmat1 = F.normalize(m.weight, dim=1)
# nmat1 = F.normalize(nmat1, dim=0)
def orth_penalty(mat):
    n, m = mat.shape[-2:]
    if n <  m:
        mat = mat.transpose(-1, -2)
    opmat1 = mat @ mat.transpose(-1, -2) - torch.eye(mat.shape[-2], device=mat.device)
    opmat2 = mat.transpose(-1, -2) @ mat - torch.eye(mat.shape[-1], device=mat.device)
    print(opmat1.shape)
    return torch.mean(torch.pow(opmat1, 2)) + torch.mean(torch.pow(opmat2, 2)) * 0
    
def orth_stacked_penalty(mat, row = False, col = True):
    n_out, n_in = mat.shape[-2:]
    # m, n = n_out, n_i
    assert n_in >= n_out
    assert n_in % n_out == 0
    i = n_in // n_out
    l = 0
    if row:
        for j in range(i):
            matj = mat[:, j*n_out:(j+1)*n_out]
            opmat_row = matj @ matj.transpose(-1, -2) - torch.eye(matj.shape[-2], device=matj.device)
            l += torch.mean(torch.pow(opmat_row, 2))
        # l += torch.mean(torch.abs(opmat_row))
    if col:
        opmat_col = mat.transpose(-1, -2) @ mat - torch.eye(mat.shape[-1], device=mat.device)
        l += torch.mean(torch.pow(opmat_col, 2))
    # return torch.mean(torch.pow(opmat2, 2))
    if torch.isnan(l):
        raise ValueError("nan")
    return l


def main():
    # omat = nn.utils.parametrizations.orthogonal(m, "weight")
    omat = nn.Linear(d1, d2, bias=False)
    nvecs = F.normalize(vecs, dim=-1)
    # print(nmat0)
    # print(nmat1)
    # print((nvecs @ nmat0).norm(dim=-1))
    # print((nvecs @ nmat1).norm(dim=-1))
    # print((omat(nvecs)).norm(dim=-1))
    l = nn.Parameter(torch.zeros(d2))
    # optimize omat and l
    optim = torch.optim.SGD([l] + list(omat.parameters()), lr=0.01, momentum=0.7)

    for i in range(5000):
        V = torch.randn(100, d1)
        y = V @ matrix
        y_ = omat(V) * l
        la = torch.mean(torch.pow(y - y_, 2)) * 1
        lb = orth_stacked_penalty(omat.weight) * 1000
        print(f"{la.item()} {lb.item()}")
        loss = la + lb
        loss.backward()
        optim.step()
        optim.zero_grad()
        # print(loss.item())
    # print(l.item())
    print((omat(nvecs)).norm(dim=-1))
    print((omat(nvecs)).norm(dim=-1).shape)
    print(omat.weight.shape)
    mo = omat.weight
    print(1)
    print(mo.shape)

    print(mo[:25].shape)
    print(mo[:, :25].shape)
    print(mo[:25, :].shape)
    print(mo[:5, :6].shape)

    b = 4

    for i in range(b):
        nbvecs = torch.randn(100, d1//b)
        nbvecs = F.normalize(nbvecs, dim=-1)
        
        m = omat.weight[:, i * d2 : (i + 1) * d2]
        v = torch.zeros(100, d1)
        v[:, i * d2 : (i + 1) * d2] = nbvecs
        print(m.shape)
        print(v.shape)
        # print((m @ nbvecs.transpose(-2, -1)).norm(dim=-1))
        # print((nbvecs @ m).norm(dim=-1))
        print((omat(v)).norm(dim=-1))
    v = torch.randn(100, d1)
    v = F.normalize(v, dim=-1)
    print((omat(v)).norm(dim=-1))

    x = torch.randn(100, 200)
    y = x @ matrix
    y_ = omat(x) * l

    print(y - y_)


@dataclass
class OrthAEConfig:
    by_row :bool
    by_col :bool
    


class OrthPenalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(OrthPenalizedLinear, self).__init__()
        assert in_features % out_features == 0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)
    
    def penalty(self):
        return orth_stacked_penalty(self.linear.weight)


class RowNormPenalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(RowNormPenalizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)
    
    def penalty(self):
        normed = self.linear.weight.norm(dim=-2)
        return torch.mean(torch.pow(self.linear.weight.norm(dim=-2) - 1, 2))
    
class NoOpModule(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=False):
        super(NoOpModule, self).__init__()

    def forward(self, x):
        return x
    
    def penalty(self):
        return 0