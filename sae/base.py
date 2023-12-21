from nqgl.sae.setup_utils import SAVE_DIR, DTYPES

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
import pprint
from nqgl.sae.sae.config import AutoEncoderConfig
from pathlib import Path
import json
from abc import abstractmethod, ABC
import logging
from typing import Optional


class BaseSAE(nn.Module, ABC):
    sae_type = None
    CONFIG = None

    def __init__(self):
        super().__init__()
        self.cfg :AutoEncoderConfig = AutoEncoderConfig()
        self.steps_since_activation_frequency_reset = torch.zeros(1, device=self.cfg.device)
        self.neuron_activation_frequency = None
        self.frozen = False
        self.neurons_to_be_reset = None


    @classmethod
    def get_version(cls):
        import glob
        type_files = glob.glob(str(SAVE_DIR) + (f"/*_{cls.sae_type}*_cfg.json"))
        logging.info("type", cls.sae_type, cls)
        logging.info("type_files", type_files)
        version_list = [
            int(file.split("/")[-1].split("_")[0])
            for file in type_files
        ]
        # version_list = [
        #     int(file.name.split("_")[0])
        #     for file in list(SAVE_DIR.iterdir())
        #     if f"_cfg.json" in str(file)
        # ]
        if len(version_list):
            return 1 + max(version_list)
        else:
            return 0



    @torch.no_grad
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed




# saving and loading
    def save(self, name=""):
        version = self.__class__.get_version()
        vname = str(version) + "_" + self.__class__.sae_type + "_" + name
        torch.save(self.state_dict(), SAVE_DIR / (vname + ".pt"))
        with open(SAVE_DIR / (str(vname) + "_cfg.json"), "w") as f:
            json.dump(asdict(self.cfg), f)
        logging.info("Saved as version", version)

    @classmethod
    def load(cls, version = "*", name=None, cfg=None, save_dir=None, omit_type = False):
        save_dir = SAVE_DIR if save_dir is None else Path(save_dir)
        # get correct name with globbing
        import glob

        vname = str(version) + "_" + cls.sae_type if not omit_type else str(version)
        vname = vname + "_" + name if name is not None else vname
        if cfg is None:
            cfg_search = str(save_dir) + f"/{vname}*_cfg.json"
            logging.info("seeking", cfg_search)
            cfg_name = glob.glob(cfg_search)
            cfg = json.load(open(cfg_name[0]))
            cfg = cls.CONFIG(**cfg)
        pt_name = glob.glob(str(save_dir / (str(vname) + "*.pt")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(pt_name[0]))
        return self

    @classmethod
    def load_latest(cls, new_cfg=None):
        version = cls.get_version() - 1
        ae = cls.load(version=version, cfg=new_cfg)
        return ae










    # neuron resampling

    def resampling_check(self):
        neurons_to_reset = self.get_dead_neurons()
        self.queue_neurons_to_reset(neurons_to_reset)
        self.reset_activation_frequencies()

    # @torch.no_grad
    # def reset_neurons(
    #     self, new_directions: torch.Tensor, norm_encoder_proportional_to_alive=True
    # ):
    #     if new_directions.shape[0] > self.neurons_to_be_reset.shape[0]:
    #         new_directions = new_directions[: self.neurons_to_be_reset.shape[0]]
    #     num_resets = new_directions.shape[0]
    #     to_reset = self.neurons_to_be_reset[:num_resets]
    #     self.neurons_to_be_reset = self.neurons_to_be_reset[num_resets:]
    #     if self.neurons_to_be_reset.shape[0] == 0:
    #         self.neurons_to_be_reset = None
    #     new_directions = new_directions / new_directions.norm(dim=-1, keepdim=True)
    #     # print(f"to_reset shape", to_reset.shape)
    #     # print(f"new_directions shape", new_directions.shape)
    #     # print(f"self.W_enc shape", self.W_enc.shape)
    #     if norm_encoder_proportional_to_alive:
    #         self.W_enc.data[to_reset[:-1], ..., to_reset[-1]] = (
    #             new_directions.T * self.alive_norm_along_feature_axis * 0.2
    #         )
    #     else:
    #         self.W_enc.data[to_reset[:-1], ..., to_reset[-1]] = new_directions.T
    #     self.W_dec.data[to_reset, :] = new_directions
    #     self.b_enc.data[to_reset] = 0

    @torch.no_grad
    def reset_neurons(
        self, new_directions: torch.Tensor, norm_encoder_proportional_to_alive=True
    ):
        if new_directions.shape[0] > self.neurons_to_be_reset.shape[0]:
            new_directions = new_directions[: self.neurons_to_be_reset.shape[0]]
        num_resets = new_directions.shape[0]
        to_reset = self.neurons_to_be_reset[:num_resets]
        self.neurons_to_be_reset = self.neurons_to_be_reset[num_resets:]
        if self.neurons_to_be_reset.shape[0] == 0:
            self.neurons_to_be_reset = None
        new_directions = new_directions / new_directions.norm(dim=-1, keepdim=True)
        print(f"to_reset shape", to_reset.shape)
        print(f"new_directions shape", new_directions.shape)
        print(f"self.W_enc shape", self.W_enc.shape)
        if norm_encoder_proportional_to_alive:
            self.W_enc.data[:, to_reset] = (
                new_directions.T * self.alive_norm_along_feature_axis * 0.2
            )
        else:
            self.W_enc.data[to_reset] = new_directions.T
        self.W_dec.data[to_reset] = new_directions
        self.b_enc.data[to_reset] = 0


    @torch.no_grad
    def reset_activation_frequencies(self, selective=True):
        viable_samples = self.get_neuron_death_viable_samples()
        if not selective or torch.all(viable_samples):
            self.neuron_activation_frequency[:] = 0
            self.steps_since_activation_frequency_reset = torch.zeros_like(self.steps_since_activation_frequency_reset, device=self.cfg.device)
        elif not torch.all(~viable_samples):
            self.neuron_activation_frequency[viable_samples.squeeze()] = 0
            self.steps_since_activation_frequency_reset[
                viable_samples.expand(
                    self.steps_since_activation_frequency_reset.shape
                )
            ] = 0


    def get_activation_frequencies(self):
        return self.neuron_activation_frequency / self.steps_since_activation_frequency_reset.unsqueeze(-1)
    
    @torch.no_grad
    def get_dead_neurons(self):
        activity_freq = self.get_activation_frequencies()
        # this gets the counts in the multihead case
        viable_samples = self.get_neuron_death_viable_samples()
        dead_neurons = activity_freq < self.cfg.dead_threshold
        return viable_samples.unsqueeze(-1).expand(dead_neurons.shape) & dead_neurons
        
    @torch.no_grad
    def get_neuron_death_viable_samples(self):
        """
        returns a boolean tensor of shape (n_heads) where True indicates that the neuron is viable for death
        in a normal autoencoder this is just 1-element 1-dim bool
        """
        return self.steps_since_activation_frequency_reset > self.cfg.neuron_death_min_samples

    


    @torch.no_grad
    def queue_neurons_to_reset(self, to_be_reset: torch.Tensor):
        from nqgl.sae.sae.model import AutoEncoder
        if type(self) is AutoEncoder:
            to_be_reset = to_be_reset.squeeze()            
        if to_be_reset.sum() > 0:
            if self.neurons_to_be_reset is not None:
                to_be_reset[self.neurons_to_be_reset] = True
            self.neurons_to_be_reset = torch.argwhere(to_be_reset).squeeze(1)
            if self.neurons_to_be_reset.ndim > 1:
                to_be_reset_reshaped = self.neurons_to_be_reset.transpose(-1, -2)
            else:
                to_be_reset_reshaped = self.neurons_to_be_reset
            
            if self.W_enc.ndim == 2:
                w_enc_norms = self.W_enc[
                    :, to_be_reset_reshaped
                ].norm(dim=0)
            else:
                w_enc_norms = self.W_enc[
                    to_be_reset_reshaped[:-1].squeeze(), 
                    :, 
                    to_be_reset_reshaped[-1]
                ].norm(dim=0)
            # logging.info("w_enc_norms", w_enc_norms.shape)
            # logging.info("to_be_reset", self.to_be_reset.sum())
            self.alive_norm_along_feature_axis = torch.mean(w_enc_norms)


    @torch.no_grad
    def re_init_neurons(self, x_diff):
        self.re_init_neurons_gram_shmidt_precise_iterative_argmax(x_diff)






    # Various reinit methods below

    @torch.no_grad
    def re_init_neurons_gram_shmidt_precise_iterative_argmax(self, x_diff):
        n_reset = min(x_diff.shape[0], self.cfg.d_data // 2, self.cfg.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            magnitudes = x_diff.norm(dim=-1)
            i_max = torch.argmax(magnitudes)
            v_orth[i] = x_diff[i_max]
            for j in range(max(0, i - self.cfg.gram_shmidt_trail), i):
                v_orth[i] -= (
                    torch.dot(v_orth[j], v_orth[i])
                    * v_orth[j]
                    / torch.dot(v_orth[j], v_orth[j])
                )
            if v_orth[i].norm() < 1e-6:
                n_succesfulselfly_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            x_diff -= (
                (x_diff @ v_orth[i]).unsqueeze(1)
                * v_orth[i]
                / torch.dot(v_orth[i], v_orth[i])
            )
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # logging.info(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])

    @torch.no_grad
    def re_init_neurons_gram_shmidt_precise_topk(self, x_diff):
        t = self.cfg.gram_shmidt_trail
        n_reset = min(x_diff.shape[0], self.cfg.d_data // 2, self.cfg.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        # logging.info(x_diff.shape)
        # v_orth[0] = F.normalize(x_diff[0], dim=-1)
        magnitudes = x_diff.norm(dim=-1)
        indices = torch.topk(magnitudes, n_reset).indices
        x_diff = x_diff[indices]
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            v_orth[i] = x_diff[i]
            for j in range(max(0, i - t), i):
                v_orth[i] -= (
                    torch.dot(v_orth[j], v_orth[i])
                    * v_orth[j]
                    / torch.dot(v_orth[j], v_orth[j])
                )
            if v_orth[i].norm() < 1e-6:
                n_succesfully_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # logging.info(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])

    @torch.no_grad
    def re_init_neurons_gram_shmidt_precise(self, x_diff):
        t = self.cfg.gram_shmidt_trail
        n_reset = min(x_diff.shape[0], self.cfg.d_data // 2, self.cfg.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        # logging.info(x_diff.shape)
        # v_orth[0] = F.normalize(x_diff[0], dim=-1)
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            v_orth[i] = x_diff[i]
            for j in range(max(0, i - t), i):
                v_orth[i] -= (
                    torch.dot(v_orth[j], v_orth[i])
                    * v_orth[j]
                    / torch.dot(v_orth[j], v_orth[j])
                )
            if v_orth[i].norm() < 1e-6:
                n_succesfully_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # logging.info(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])
