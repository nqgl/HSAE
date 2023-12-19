from nqgl.sae.hsae.config import (
    HierarchicalAutoEncoderConfig, 
    HierarchicalAutoEncoderLayerConfig
)
from sae.model import AutoEncoder, AutoEncoderConfig
from setup_utils import DTYPES
import torch
import torch.nn as nn


class HierarchicalAutoEncoderLayer(AutoEncoder, nn.Module):
    def __init__(self, 
                 cfg :HierarchicalAutoEncoderLayerConfig, cfg_0 :HierarchicalAutoEncoderConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.cfg_0 = cfg_0

        dtype = DTYPES[cfg_0.enc_dtype]

        self.b_dec = nn.Parameter(
            torch.zeros(
                self.cfg.n_sae,
                self.cfg.d_data,
                dtype=dtype
            )
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.n_sae,
                    self.cfg.d_data,
                    self.cfg.d_dict,
                    dtype=dtype
                )
            )
        )

        self.b_enc = nn.Parameter(
            torch.zeros(
                self.cfg.n_sae,
                self.cfg.d_dict,
                dtype=dtype
            )
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.n_sae,
                    self.cfg.d_dict,
                    self.cfg.d_data,
                    dtype=dtype
                )
            )
        )

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to(self.cfg.device)
        self.prev_layer_L0 = None



    def encode_flat(self, x, W_enc, b_dec, b_enc, cache_acts=False, cache_l0=False, record_activation_frequency=False):
        # batch, n_sae, d_data
        x_cent = x - b_dec
        # x_cent = x_cent.unsqueeze(-2)
        print("\nmult", x_cent.shape, W_enc.shape)
        # input()
        m = x_cent @ W_enc
        acts = self.nonlinearity(m + b_enc)

        # print("x_cent", x_cent.shape)
        # print("m", m.shape)
        # print("acts", acts.shape)
        return acts



    def decode_flat(self, acts, W_dec, b_dec):
        # print("acts shape:", acts.shape)
        # print("W_dec shape:", W_dec.shape)
        # print("b_dec shape:", b_dec.shape)
        m =  acts @ W_dec
        o = m + b_dec
        # print(m.shape, o.shape)
        return o




    def forward(self, x: torch.Tensor, gate: torch.Tensor, dense=True, prev_sae=None):
        self.prev_layer_L0 = prev_sae.cached_l0_norm if prev_sae is not None else None
        return self.ad_hoc_sparse2(x, gate)
        if dense:
            return self.dense(x, gate)
        else:
            print("sparse", x.shape)
            return self.ad_hoc_sparse(x, gate)



    def dense(self, x: torch.Tensor, gate: torch.Tensor):
        # x: (batches, d_data)
        # gate: (batches, n_sae)

        x = x.unsqueeze(-2).unsqueeze(-2) # b 1 1 d_data
        b_dec = self.b_dec.unsqueeze(-2) # n_sae 1 d_data
        # x: b, n_sae, 1 d_data
        W_enc = self.W_enc # n_sae d_data d_dict
        # x: b, n_sae, 1 d_dict
        b_enc = self.b_enc.unsqueeze(-2) # n_sae 1 d_dict
        # x: b, n_sae, 1, d_dict
        W_dec = self.W_dec # n_sae d_dict d_data
        # x: b, n_sae, 1, d_data
        # print("layer parts", b_dec.shape, W_enc.shape, b_enc.shape, W_dec.shape)
        # print("x", x.shape)
        acts = self.encode_flat(x=x, W_enc=W_enc, b_dec=b_dec, b_enc=b_enc)
        # print("acts", acts.shape)
        self.cache(acts)
        saes_out = self.decode_flat(acts, W_dec=W_dec, b_dec=b_dec)
        # print("saes_out", saes_out.shape)
        saes_out = saes_out * gate.unsqueeze(-1).unsqueeze(-1)
        return saes_out.sum(dim=-2).sum(dim=-2)


    def sparse(self, x: torch.Tensor, gate: torch.Tensor):
        # x: (batches, d_data)
        # gate: (batches, n_sae)
        batch = x.shape[0]
        d_data = self.cfg.d_data
        d_dict = self.cfg.d_dict
        sgate = gate.to_sparse().unsqueeze(-1).unsqueeze(-1)
        x = sgate * x.unsqueeze(-2).unsqueeze(-2) # b 1 1 d_data
        b_dec = (sgate * self.b_dec.unsqueeze(-2)).view(-1, 1, d_data) # n_sae 1 d_data
        # x: b, n_sae, 1 d_data
        W_enc = (sgate * self.W_enc).view(-1, d_data, d_dict) # n_sae d_data d_dict
        # x: b, n_sae, 1 d_dict
        b_enc = (sgate * self.b_enc.unsqueeze(-2), d_data, d_dict) # n_sae 1 d_dict
        # x: b, n_sae, 1, d_dict
        W_dec = (sgate * self.W_dec) # n_sae d_dict d_data
        print("sparse?", W_dec.is_sparse)
        # x: b, n_sae, 1, d_data
        # print("layer parts", b_dec.shape, W_enc.shape, b_enc.shape, W_dec.shape)
        # print("x", x.shape)
        acts = self.encode_sparse(x=x, W_enc=W_enc, b_dec=b_dec, b_enc=b_enc)
        # print("acts", acts.shape)
        self.cache(acts)
        saes_out = self.decode_sparse(acts, W_dec=W_dec, b_dec=b_dec)
        # print("saes_out", saes_out.shape)
        saes_out = saes_out * gate.unsqueeze(-1).unsqueeze(-1)
        return saes_out.sum(dim=-2).sum(dim=-2)


    def encode_sparse(self, x, W_enc, b_dec, b_enc, cache_acts=False, cache_l0=False, record_activation_frequency=False):
        x_cent = x - b_dec
        m = torch.sparse.mm(x, self.W_enc)
        acts = self.nonlinearity(m + b_enc)

        return acts
    def decode_sparse(self, acts, W_dec, b_dec):
        m = torch.sparse.mm(acts, W_dec)
        o = m + b_dec
        return o


    def ad_hoc_sparse2(self, x: torch.Tensor, gate: torch.Tensor):
        # x: (batches, d_data)
        # gate: (batches, n_sae)
        x_dumb_shape = x.shape
        if len(x_dumb_shape) > 2:
            x = x.reshape(-1, x_dumb_shape[-1])
            gate = gate.reshape(-1, gate.shape[-1])
        batches = x.shape[0]

        # x = self.scale(x)
        gate = gate.unsqueeze(-1).unsqueeze(-1).transpose(0,1) # n_sae batches 1 1
        bgate = gate != 0
        # gate = gate.to_sparse()
        # print("flat_indices", flat_indices.shape)
        # if flat_indices.shape[1]/batches > 100:
        #     newgate = torch.zeros(batches, self.cfg.n_sae, device=self.cfg.device)
        #     torch.multinomial(

        # batches 1 d_data  *  n_sae batches 1 1
        # -> n_sae batches 1 d_data
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
        # print(W_enc.shape, b_enc.shape, W_dec.shape, b_dec.shape)
        # print(self.W_enc.shape, self.b_enc.shape, self.W_dec.shape, self.b_dec.shape)
        # print("x_flat", x_flat.shape)
        flat_acts = self.encode_flat(x=x_flat, W_enc=W_enc_flat, b_dec=b_dec_flat, b_enc=b_enc_flat)
        print("flat_acts", flat_acts.shape)


        # acts = acts.scatter_add(
        #     0, 
        #     flat_indices.unsqueeze(-1).unsqueeze(-1).expand(2, -1, self.cfg.n_sae, self.cfg.d_dict), 
        #     flat_acts.reshape(-1, 1, self.cfg.d_dict)
        # )
        # acts[flat_indices] = flat_acts
        # print(acts.shape, flat_acts.shape)
        self.cache_flat(flat_acts, sae_idxs=sae_idxs, batches=batches, gate=gate)
        # flat_acts = flat_acts * dgate[flat_indices]
        saes_out_flat = self.decode_flat(gate[sae_idxs, batch_idxs] * flat_acts, W_dec=W_dec_flat, b_dec=b_dec_flat)
        print("saes_out_flat", saes_out_flat.shape)
        print("flat_indicies", flat_indices.shape)
        flatsize = saes_out_flat.shape[0]
        print()
        z = torch.zeros(batches, self.cfg.d_data, device=self.cfg.device)
        bids = batch_idxs.reshape(flatsize, 1).expand(-1, self.cfg.d_data)
        sae_re = saes_out_flat.reshape(flatsize, self.cfg.d_data)
        print("z", z.shape)
        print("bids", bids.shape)
        print("sae_re", sae_re.shape)
        print("batch_id_max", batch_idxs.max())
        x_out = torch.scatter_add(
            torch.zeros(batches, self.cfg.d_data, device=self.cfg.device),
            0,
            batch_idxs.reshape(flatsize, 1).expand(-1, self.cfg.d_data),
            saes_out_flat.reshape(flatsize, self.cfg.d_data)
        )

        # x_reconstruct = self.unscale(x_out)
        print("x_out", x_out.shape)
        print(x_out.is_sparse)
        # input()
        # print("x_out sum", x_out.sum())
        print(x_out[0].sum(), x_out[1].sum())
        print(x_out[:, 0].sum(), x_out[:, 1].sum())
        return x_out.reshape(x_dumb_shape)


    def cache_flat(self, flat_acts, sae_idxs, batches, gate):
        # TODO this currently does not translate to the >2 layer case 
        # because acts are not cached full detail, they are stored
        # summed along the batch axis
        feat_acts = torch.scatter_add(
            torch.zeros(self.cfg.n_sae, self.cfg.d_dict, device=self.cfg.device),
            0,
            sae_idxs.unsqueeze(-1).expand(-1, self.cfg.d_dict),
            flat_acts.reshape(-1, self.cfg.d_dict)
        )

        print("feat acts sum:", feat_acts.sum())

        n_active_batches_per_head = (gate != 0).squeeze(-1).squeeze(-1).float().sum(dim=-1, keepdim=True)
        mean_acts = feat_acts / (n_active_batches_per_head.reshape(-1, 1) + 1e-9)
        self.cache(mean_acts)
        self.cached_l0_norm = torch.count_nonzero(flat_acts) / flat_acts.shape[0]
        self.cached_l1_loss = flat_acts.float().abs().sum() / batches



    def get_loss(self):
        return self.cached_l1_loss * self.cfg.l1_coeff


    def cache(self, acts, cache_l0 = True, cache_acts = True):
        # self.cached_l1_loss = acts.float().abs().sum(dim=-1).mean()
        # self.cached_l0_norm = torch.count_nonzero(acts, dim=-1).float().mean() if cache_l0 else None
        self.cached_acts = acts if cache_acts else None
        record_activation_frequency = False
        if record_activation_frequency:
            activated = torch.mean((acts > 0).float(), dim=0)
            activated = torch.count_nonzero(acts, dim=0) / acts.shape[0]
            self.neuron_activation_frequency = activated + self.neuron_activation_frequency.detach()
            self.steps_since_activation_frequency_reset += 1
        return acts



    @torch.no_grad()
    def update_scaling(self, x :torch.Tensor):
        if self.cfg.sublayers_train_on_error:
            x_cent = x - x.mean(dim=0)
            # var = (x_cent ** 2).sum(dim=-1)
            # std = torch.sqrt(var).mean()
            std = x_cent.norm(dim=-1).mean()
            self.std_dev_accumulation += std #x_cent.std(dim=0).mean() is p diferent I believe
            self.std_dev_accumulation_steps += 1
            self.scaling_factor = self.std_dev_accumulation / self.std_dev_accumulation_steps


    # @torch.no_grad()
    def scale(self, x):
        return x * self.cfg.data_rescale

    # @torch.no_grad()
    def unscale(self, x):
        return x / self.cfg.data_rescale