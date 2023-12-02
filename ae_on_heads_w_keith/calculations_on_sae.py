
import torch
import z_sae
from functools import partial
import tqdm
import einops


@torch.no_grad()
def get_recons_loss(model, encoder, buffer, num_batches=5, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for i in range(num_batches):
        tokens = buffer.all_tokens[torch.randperm(len(buffer.all_tokens))[:encoder.cfg.model_batch_size]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(encoder.cfg.act_name, partial(replacement_hook, encoder=local_encoder))])
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(encoder.cfg.act_name, mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(encoder.cfg.act_name, zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss

# Frequency
@torch.no_grad()
def get_freqs(model, encoder, buffer, num_batches=25, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).to(encoder.cfg.device)
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = buffer.all_tokens[torch.randperm(len(buffer.all_tokens))[:encoder.cfg.model_batch_size]]

        _, cache = model.run_with_cache(tokens, stop_at_layer=encoder.cfg.layer + 1)
        acts = cache[encoder.cfg.act_name]
        acts = acts.reshape(-1, encoder.cfg.act_size)

        hidden = local_encoder(acts)[2]

        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores

@torch.no_grad()
def re_init(model, encoder, buffer, indices):
    new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_enc)))
    new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_dec)))
    new_b_enc = (torch.zeros_like(encoder.b_enc))
    print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)
    encoder.W_enc.data[:, indices] = new_W_enc[:, indices]
    encoder.W_dec.data[indices, :] = new_W_dec[indices, :]
    encoder.b_enc.data[indices] = new_b_enc[indices]


def replacement_hook(acts, hook, encoder):
    print(acts.shape)
    if encoder.cfg.flatten_heads:
        acts = einops.rearrange(acts, "... n_head d_head -> ... (n_head d_head)")
    mlp_post_reconstr = encoder(acts)
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post



