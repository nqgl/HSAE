import torch
from nqgl.sae.hsae.hsae import HierarchicalAutoEncoder, HierarchicalAutoEncoderConfig

class AdamResetter:
    def __init__(self, param):
        self.param = param
    
    def __getitem__(self, indices):
        return AdamResetterCallable(self.param, indices)        
    
class AdamResetterCallable:
    def __init__(self, param, indices):
        self.param = param
        self.indices = indices

    def __call__(self, adam: torch.optim.Adam):
        state = adam.state[self.param]
        state["exp_avg"][self.indices] = 0 # zero momentum
        eps = 1e-5
        ratio = 0.99
        state["exp_avg_sq"][self.indices] = (
            (
                eps 
                + torch.sum(state["exp_avg_sq"])
                - torch.sum(state["exp_avg_sq"][self.indices] * ratio)
            ) / (eps + state["exp_avg_sq"].numel() - self.indices.numel() * ratio)
        )
        
        # leave step as is


def reset_adam(adam: torch.optim.Adam, param, indices):
    state = adam.state[p]
    state["exp_avg"]
    exp_avg_sq
    step
    # amsgrad?
        # if so, then max_exp_avg_sq
    

def main():
    cfg = HierarchicalAutoEncoderConfig(d_data=4)
    hsae = HierarchicalAutoEncoder(cfg)
    adam = torch.optim.Adam(hsae.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    groups = adam.param_groups
    # print(groups[0].keys())
    x = torch.randn(10, 4, device="cuda")
    x_reconstruct = hsae(x)
    loss = hsae.get_loss()
    loss.backward()
    adam.step()
    for key in adam.state[hsae.saes[0].b_enc].keys():
        print(key)
        print(adam.state[hsae.saes[0].b_enc][key].shape)
    # for p in groups[0]["params"]:
    #     print(p.shape)



if __name__ == "__main__":
    main()