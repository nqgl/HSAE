from nqgl.sae.sae.config import AutoEncoderConfig
from nqgl.sae.setup_utils import get_model
from nqgl.sae.buffer import Buffer

import torch


def precompute_activations(
    model,
    cfg: AutoEncoderConfig,
    tokens,
    token_start=0,
    proportion_of_data=0.01,
    chunk_size=10000,
    num_chunks=100,
    freshen_buffer=False,
    storage_location="~/activations",
):
    import tqdm

    model.eval()
    model.to(cfg.device)
    tokens = tokens[token_start:token_stop]
    num_batches = tokens.shape[0] // cfg.batch_size
    buffer = Buffer(cfg, tokens, model=model)
    if freshen_buffer:
        buffer.freshen_buffer(2)
    # buffer.skip_first_tokens_ratio(0.08)
    prev_chunk = None
    next_chunk = torch.zeros((cfg.buffer_size, cfg.d_data), device="cuda")
    for i in tqdm.trange(proportion_of_data * cfg.num_tokens // cfg.batch_size):
        acts = buffer.next()
        print(acts.shape)
        if buffer.token_pointer > tokens.shape[0] - chunk_size:
            print("breaking with token pointer = ", buffer.token_pointer)
            break
        assert False

    return acts
