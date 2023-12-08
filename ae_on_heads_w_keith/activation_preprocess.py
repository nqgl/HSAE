from .z_sae import Buffer, get_model, AutoEncoderConfig
import torch


def precompute_activations(model, cfg :AutoEncoderConfig, tokens, token_start = 0, token_stop = None, chunk_size = 10000, freshen_buffer = False, storage_location="~/activations"):
    import tqdm
    model.eval()
    model.to(cfg.device)
    tokens = tokens[token_start:token_stop]
    buffer = Buffer(cfg, tokens, model=model)
    if freshen_buffer:
        buffer.freshen_buffer(2)
    buffer.skip_first_tokens_ratio(0.08)
    # next_chunk = torch.zeros((chunk_size * cfg.))
    for i in tqdm.trange(cfg.num_tokens // batch_size):
        acts = buffer.next()
        print(acts.shape)
        assert False
    
    return acts