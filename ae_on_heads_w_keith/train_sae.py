import z_sae
import wandb
import tqdm
import torch
from calculations_on_sae import get_recons_loss, get_freqs, re_init
from transformer_lens import HookedTransformer

def train(encoder :z_sae.AutoEncoder, cfg :z_sae.AutoEncoderConfig, buffer :z_sae.Buffer):
   
    wandb.login(key="763967cb34da114063379b7b42fec47c0be2fdb8")

    try:
        wandb.init(project="autoencoders", entity="hiibb", config=cfg)
        num_batches = cfg["num_tokens"] // cfg["batch_size"]
        # model_num_batches = cfg["model_batch_size"] * num_batches
        encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        recons_scores = []
        act_freq_scores_list = []
        for i in tqdm.trange(num_batches):
            i = i % buffer.all_tokens.shape[0]
            acts = buffer.next()
            x_reconstruct = encoder(acts)  
            l2_loss = encoder.l2_loss_cached
            l1_loss = encoder.l1_loss_cached
            l0_norm = encoder.l0_norm_cached # TODO condisder turning this off if is slows down calculation
            loss = encoder.get_loss()
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            encoder_optim.step()
            encoder_optim.zero_grad()
            loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item(), "l0_norm": l0_norm.item()}
            del loss, x_reconstruct, l2_loss, l1_loss, acts, l0_norm
            if (i) % 100 == 0:
                wandb.log(loss_dict)
                print(loss_dict)
            if (i) % 1000 == 0:
                x = (get_recons_loss(local_encoder=encoder))
                print("Reconstruction:", x)
                recons_scores.append(x[0])
                freqs = get_freqs(5, local_encoder=encoder)
                act_freq_scores_list.append(freqs)
                # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
                wandb.log({
                    "recons_score": x[0],
                    "dead": (freqs==0).float().mean().item(),
                    "below_1e-6": (freqs<1e-6).float().mean().item(),
                    "below_1e-5": (freqs<1e-5).float().mean().item(),
                })
            if (i+1) % 30000 == 0:
                encoder.save()
                freqs = get_freqs(50, local_encoder=encoder)
                to_be_reset = (freqs<10**(-5.5))
                wandb.log({"reset_neurons": to_be_reset.sum()})

                print("Resetting neurons!", to_be_reset.sum())
                re_init(to_be_reset, encoder)
    finally:
        encoder.save()


def main():
    ae_cfg = z_sae.AutoEncoderConfig(site="z", act_size=512)
    cfg = z_sae.post_init_cfg(ae_cfg)
    model = z_sae.get_model(cfg)
    all_tokens = z_sae.load_data(model)
    encoder = z_sae.AutoEncoder(cfg)
    buffer = z_sae.Buffer(cfg, all_tokens, encoder)
    train(encoder, cfg, buffer)

if __name__ == "__main__":
    main()