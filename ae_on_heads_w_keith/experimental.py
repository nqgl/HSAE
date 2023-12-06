from z_sae import AutoEncoderConfig, AutoEncoder


class LinearRootAutoEncoder(AutoEncoder):
    def __init__(self, cfg):
        cfg.experimental_type = "linear_root"
        super().__init__(cfg)

    def forward(self, x, cache_l0 = True, cache_acts = False, record_activation_frequency = False):
        x_cent = x - self.b_dec
        # print(x_cent.dtype, x.dtype, self.W_dec.dtype, self.b_dec.dtype)
        acts = self.nonlinearity(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        # self.l2_loss_cached = (x_reconstruct.float() - x.float()).pow(2).mean(-1).mean(0)
        self.l1_loss_cached = (acts.float().abs() + 0.1).pow(0.5).mean(dim=(-2))
        self.l2_loss_cached = (x_reconstruct.float() - x.float()).abs().mean(-1).mean(0)
        if cache_l0:
            self.l0_norm_cached = (acts > 0).float().sum(dim=-1).mean()
        else:
            self.l0_norm_cached = None
        if cache_acts:
            self.cached_acts = acts
        else:
            self.cached_acts = None
        if record_activation_frequency:
            # print(acts.shape)
            activated = torch.mean((acts > 0).float(), dim=0)
            # print("activated shape", activated.shape)
            # print("freq shape", self.activation_frequency.shape)
            self.activation_frequency = activated + self.activation_frequency
            self.steps_since_activation_frequency_reset += 1
        return x_reconstruct
