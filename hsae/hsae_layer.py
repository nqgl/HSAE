from nqgl.sae.hsae.config import (
    HierarchicalAutoEncoderConfig,
    HierarchicalAutoEncoderLayerConfig,
)
from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig
from nqgl.sae.setup_utils import DTYPES
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F


class HierarchicalAutoEncoderLayer(AutoEncoder, nn.Module):
    sae_type = "HSAE_LAYER"
    CONFIG = HierarchicalAutoEncoderLayerConfig

    def __init__(
        self,
        cfg: HierarchicalAutoEncoderLayerConfig,
        cfg_0: HierarchicalAutoEncoderConfig,
    ):
        super().__init__(cfg)
        self.cfg :HierarchicalAutoEncoderLayerConfig = cfg
        self.cfg_0 = cfg_0

        dtype = DTYPES[cfg_0.enc_dtype]

        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.n_sae, self.cfg.d_data, dtype=dtype)
        )

        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.n_sae, self.cfg.d_dict, dtype=dtype)
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.n_sae, self.cfg.d_data, self.cfg.d_dict, dtype=dtype
                )
            )
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.n_sae, self.cfg.d_dict, self.cfg.d_data, dtype=dtype
                )
            )
        )


        # self.W_enc = nn.Parameter(
        #     1e-9 + torch.zeros(
        #         self.cfg.n_sae, self.cfg.d_data, self.cfg.d_dict, dtype=dtype
        #     )
        # )

        # self.W_dec = nn.Parameter(
        #     1e-9 + torch.zeros(
        #         self.cfg.n_sae, self.cfg.d_dict, self.cfg.d_data, dtype=dtype
        #     )
        # )

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.steps_since_activation_frequency_reset = torch.zeros(cfg.n_sae, device=self.cfg.device)
        self.to(self.cfg.device)
        self.prev_layer_L0 = None
        self.neuron_activation_frequency = torch.zeros(
            self.cfg.n_sae,
            self.cfg.d_dict,
            dtype=torch.float32,
            device=self.cfg.device
        )
        self.cached_gate = None
    

    def encode_flat(
        self,
        x,
        W_enc,
        b_dec,
        b_enc
    ):
        # batch, n_sae, d_data
        x_cent = x.float() #- b_dec
        # x_cent = x_cent.unsqueeze(-2)
        # logging.info("\nmult", x_cent.shape, W_enc.shape)
        # input()
        m = x_cent @ W_enc
        acts = self.nonlinearity(m + b_enc)

        # logging.info("x_cent", x_cent.shape)
        # logging.info("m", m.shape)
        # logging.info("acts", acts.shape)
        return acts

    def decode_flat(self, acts, W_dec, b_dec):
        m = acts @ W_dec
        o = m + b_dec
        return o

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        dense=True,
        prev_sae=None,
        **cache_kwargs
    ):
        self.cached_gate = gate
        self.prev_layer_L0 = prev_sae.cached_l0_norm if prev_sae is not None else None
        # if dense:
        #     gate = torch.zeros_like(gate)
        return self.sparse_forward(x, gate, **cache_kwargs)

    def get_loss(self):
        return self.cached_l1_loss * self.cfg.l1_coeff

    def sparse_forward(self, x: torch.Tensor, gate: torch.Tensor, **cache_kwargs):
        # x: (batches, d_data)
        # gate: (batches, n_sae)
        x_dumb_shape = x.shape
        if len(x_dumb_shape) > 2:
            x = x.reshape(-1, x_dumb_shape[-1])
            gate = gate.reshape(-1, gate.shape[-1])
        batches = x.shape[0]
        x = self.scale(x)

        gate = gate.unsqueeze(-1).unsqueeze(-1).transpose(0, 1)  # n_sae batches 1 1
        bgate = gate != 0

        x_s = (x.unsqueeze(-2) * bgate).to_sparse(2)
        flat_indices = x_s.indices()
        batch_idxs = flat_indices[1]
        sae_idxs = flat_indices[0]
        assert torch.sum(gate[sae_idxs, batch_idxs] == 0) == 0

        x_flat = x_s.values()

        W_enc_flat = self.W_enc[sae_idxs]
        b_enc_flat = self.b_enc[sae_idxs].unsqueeze(-2)
        W_dec_flat = self.W_dec[sae_idxs]
        b_dec_flat = self.b_dec[sae_idxs].unsqueeze(-2)
        
        if self.cfg.scale_b_dec:
            b_dec_flat = b_dec_flat * self.scale(gate[flat_indices[0], flat_indices[1]])

        flat_acts = self.encode_flat(
            x=x_flat, W_enc=W_enc_flat, b_dec=b_dec_flat, b_enc=b_enc_flat
        )

        acts = self.full_acts(flat_acts, flat_indices, batches)
        self.cache(acts=acts, flat_acts=flat_acts, gate=gate, **cache_kwargs)

        # logging.info("flat_acts", flat_acts.shape)

        saes_out_flat = self.decode_flat(
            gate[sae_idxs, batch_idxs] * flat_acts, W_dec=W_dec_flat, b_dec=b_dec_flat
        )

        # logging.info("saes_out_flat", saes_out_flat.shape)
        # logging.info("flat_indicies", flat_indices.shape)

        flatsize = saes_out_flat.shape[0]
        z = torch.zeros(batches, self.cfg.d_data, device=self.cfg.device)
        bids = batch_idxs.reshape(flatsize, 1).expand(-1, self.cfg.d_data)
        sae_re = saes_out_flat.reshape(flatsize, self.cfg.d_data)

        # logging.info("z", z.shape)
        # logging.info("bids", bids.shape)
        # logging.info("sae_re", sae_re.shape)
        # logging.info("batch_id_max", batch_idxs.max())

        x_out = torch.scatter_add(
            torch.zeros(batches, self.cfg.d_data, device=self.cfg.device),
            0,
            batch_idxs.reshape(flatsize, 1).expand(-1, self.cfg.d_data),
            saes_out_flat.reshape(flatsize, self.cfg.d_data),
        )

        # logging.info("x_out", x_out.shape)
        # logging.info(x_out.is_sparse)
        # logging.info(x_out[0].sum(), x_out[1].sum())
        # logging.info(x_out[:, 0].sum(), x_out[:, 1].sum())
        return self.unscale(x_out.reshape(x_dumb_shape))

    def full_acts(self, flat_acts, flat_indices, batches):
        acts = torch.zeros(
            batches, self.cfg.n_sae, self.cfg.d_dict, device=self.cfg.device
        )
        acts[flat_indices[1], flat_indices[0]] = flat_acts.squeeze(-2)
        return acts

    def cache(
        self,
        acts,
        flat_acts,
        gate,
        cache_l0=True,
        cache_acts=True,
        record_activation_frequency=True,
    ):
        batches = acts.shape[0]

        self.cached_l1_loss = flat_acts.float().abs().sum() / batches
        self.cached_acts = acts if cache_acts else None
        self.cached_l0_norm = torch.count_nonzero(flat_acts) / flat_acts.shape[0]

        n_active_batches_per_head = (
            (gate != 0).squeeze(-1).squeeze(-1).float().sum(dim=-1)
        )

        self.cached_l0_norms = (  # this is the per-head measure
            torch.count_nonzero(acts, dim=-1).float().mean(dim=0)
            / n_active_batches_per_head # TODO check if this is okay or if it needs to be unsqueezed
            if cache_l0 else None
        )
        

        if record_activation_frequency:
            activated = (
                torch.count_nonzero(acts, dim=0)
                / acts.shape[0]
            )
            self.neuron_activation_frequency = (
                activated + self.neuron_activation_frequency.detach()
            )
            self.steps_since_activation_frequency_reset += (
                # torch.ones_like *
                n_active_batches_per_head
            )

    @torch.no_grad()
    def update_scaling(self, x: torch.Tensor):
        if self.cfg.sublayers_train_on_error:
            x_cent = x - x.mean(dim=0)
            # var = (x_cent ** 2).sum(dim=-1)
            # std = torch.sqrt(var).mean()
            std = x_cent.norm(dim=-1).mean()
            self.std_dev_accumulation += (
                std  # x_cent.std(dim=0).mean() is p diferent I believe
            )
            self.std_dev_accumulation_steps += 1
            self.scaling_factor = (
                self.std_dev_accumulation / self.std_dev_accumulation_steps
            )

    # @torch.no_grad()
    def scale(self, x):
        return x * self.cfg.data_rescale

    # @torch.no_grad()
    def unscale(self, x):
        return x / self.cfg.data_rescale

    def dense(self, x: torch.Tensor, gate: torch.Tensor):
        # x: (batches, d_data)
        # gate: (batches, n_sae)

        x = x.unsqueeze(-2).unsqueeze(-2)  # b 1 1 d_data
        b_dec = self.b_dec.unsqueeze(-2)  # n_sae 1 d_data
        # x: b, n_sae, 1 d_data
        W_enc = self.W_enc  # n_sae d_data d_dict
        # x: b, n_sae, 1 d_dict
        b_enc = self.b_enc.unsqueeze(-2)  # n_sae 1 d_dict
        # x: b, n_sae, 1, d_dict
        W_dec = self.W_dec  # n_sae d_dict d_data
        # x: b, n_sae, 1, d_data
        # logging.info("layer parts", b_dec.shape, W_enc.shape, b_enc.shape, W_dec.shape)
        # logging.info("x", x.shape)
        acts = self.encode_flat(x=x, W_enc=W_enc, b_dec=b_dec, b_enc=b_enc)
        # logging.info("acts", acts.shape)
        self.cache(acts)
        saes_out = self.decode_flat(acts, W_dec=W_dec, b_dec=b_dec)
        # logging.info("saes_out", saes_out.shape)
        saes_out = saes_out * gate.unsqueeze(-1).unsqueeze(-1)
        return saes_out.sum(dim=-2).sum(dim=-2)




    def re_init_neurons(self, x_diff, gate, norm_encoder_proportional_to_alive=True):
        self.get_neuron_death_viable_samples()
        ready = gate[:, self.neurons_to_be_reset[:, 0]]
        has_data = ready.sum(dim=0) > 0
        head_has_data = gate.sum(dim=0) > 0
        heads_with_data = self.neurons_to_be_reset[has_data]
        num_heads_with_data = (gate.sum(dim=0) > 0).sum()
        indices_with_data = (
            torch.arange(
                0, 
                self.neurons_to_be_reset.shape[0], 
                device=self.cfg.device
            )[has_data]
        )

        indices = torch.scatter(
            torch.zeros(self.cfg.n_sae, dtype=torch.long, device=self.cfg.device) - 1,
            0,
            self.neurons_to_be_reset[:, 0][has_data],
            torch.arange(0, self.neurons_to_be_reset.shape[0], device=self.cfg.device)[has_data]
        )
        i = indices[indices != -1]

        indices = torch.zeros(num_heads_with_data, dtype=torch.long, device=self.cfg.device)
        i = torch.scatter(
            torch.zeros(self.cfg.n_sae, dtype=torch.long, device=self.cfg.device) - 1,
            0,
            heads_with_data[:, 1],
            indices_with_data
        )
        indices = i[i != -1]
        heads = self.neurons_to_be_reset[indices][:, 0]

        dirs = (gate.transpose(-2, -1) @ x_diff)[heads]
        neuron_mask = torch.zeros_like(self.neurons_to_be_reset[:, 0], dtype=torch.bool, device=self.cfg.device)
        neuron_mask[indices] = True
        assert dirs.shape[0] == neuron_mask.sum()
        self.reset_neurons(dirs, neuron_mask, norm_encoder_proportional_to_alive)


    def reset_neurons(        
        self, 
        new_directions: torch.Tensor, 
        replacement_mask: torch.Tensor,
        norm_encoder_proportional_to_alive=True
    ):

        to_reset = self.neurons_to_be_reset[replacement_mask]
        self.neurons_to_be_reset = self.neurons_to_be_reset[~replacement_mask]
        if self.neurons_to_be_reset.shape[0] == 0:
            self.neurons_to_be_reset = None
        new_directions = F.normalize(new_directions, dim=-1) 
        if norm_encoder_proportional_to_alive:
            self.W_enc.data[to_reset[:, 0], :, to_reset[:, -1]] = (
                (new_directions.T * self.alive_norm_along_feature_axis * 0.2).T
            )
        else:
            self.W_enc.data[to_reset[:, 0], :, to_reset[:, -1]] = new_directions
        self.W_dec.data[to_reset[:, 0], to_reset[:, 1]] = new_directions
        self.b_enc.data[to_reset] = 0




    @torch.no_grad()
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
