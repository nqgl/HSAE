import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, List, Union
import gradcool_functions
import os
import json
from dataclasses import dataclass
import orthonormal
from abc import ABC, abstractmethod

class AutoEncoderVariant(nn.Module):
    def __init__(self, nonlinearity, cache_pre = False, cache_post = True):
        self.nonlinearity = nonlinearity
        self.cache_pre = cache_pre
        self.cache_post = cache_post
        self.cached_acts = None
        self.cached_acts_pre = None

    def forward(self, x):
        acts_pre = self.encode(x)
        acts = self.nonlinearity(acts_pre)
        x_hat = self.decode(acts)
        if self.cache_pre:
            self.cached_acts_pre = acts_pre
        if self.cache_post:
            self.cached_acts = acts
        return x_hat

    @abstractmethod
    def encode(self, x):
        """Do encoder step. DO NOT CALL NONLINEARITY"""
        pass

    @abstractmethod
    def decode(self, x):
        pass

class SharkeyEtAlAutoEncoder(nn.Module):
    def __init__(self, d_act, d_dict, nonlinearity = nn.ReLu):
        self.W = nn.Parameter(nn.init.kaiming_normal_(torch.empty(d_dict, d_act)))
        self.b = nn.Parameter(torch.zeros(d_dict))
        self.nonlinearity = nonlinearity
        self.cached_acts = None

    def forward(self, x):
        """W.T @ nonlinearity(W @ x + b)"""
        # return equivalent to self.W.transpose(-1, -2) @ self.nonlinearity(self.W @ x + self.b)
        # but with cached activations
        return self.decode(self.encode(x))

    def encode(self, x):
        """do encoder step and cache activations"""
        acts = self.nonlinearity(self.W @ x + self.b)
        return acts

    def decode(self, x):
        return self.W.transpose(-1, -2) @ x
    
class SharkeyEtAlAEV(AutoEncoderVariant):
    def __init__(self, d_act, d_dict, nonlinearity = nn.ReLu):
        super().__init__(nonlinearity)
        self.W = nn.Parameter(nn.init.kaiming_normal_(torch.empty(d_dict, d_act)))
        self.b = nn.Parameter(torch.zeros(d_dict))

    def encode(self, x):
        """do encoder step and cache activations"""
        acts = self.nonlinearity(self.W @ x + self.b)
        return acts

    def decode(self, x):
        return self.W.transpose(-1, -2) @ x


# class AnthropicAutoEncoder(nn.Module):
#     def __init__(self, d_act, d_dict, nonlinearity = nn.ReLu):

        