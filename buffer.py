from sae_config import AutoEncoderConfig


import einops
import torch


import time

# I might come back to this and think about changing refresh ratio up
# also is there a pipelining efficiency we could add?
# is it bad to have like 2 gb of tokens in memory?

class Buffer():
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder.
    It'll automatically run the model to generate more when it gets halfway empty.
    """
    def __init__(self, cfg, tokens, model):
        self.buffer = torch.zeros((cfg.buffer_size, cfg.act_size), dtype=torch.float16, requires_grad=False).to(cfg.device)
        self.cfg :AutoEncoderConfig = cfg
        self.token_pointer = 0
        self.first = True
        self.all_tokens = tokens
        self.model = model
        self.time_shuffling = 0
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """
        Refreshes the buffer by populating it with new activations, then shuffles it.

        Note: This method assumes that the necessary attributes and configurations are already set.
        """
        t0 = time.time()
        self.pointer = 0
        with torch.autocast("cuda", torch.float16):
            if self.first:
                num_batches = self.cfg.buffer_batches
            else:
                num_batches = int(self.cfg.buffer_batches * self.cfg.buffer_refresh_ratio)
            self.first = False
            for _ in range(0, num_batches, self.cfg.model_batch_size):
                tokens = self.all_tokens[self.token_pointer:self.token_pointer+self.cfg.model_batch_size]
                _, cache = self.model.run_with_cache(tokens, stop_at_layer=self.cfg.layer+1)
                if self.cfg.flatten_heads:
                    acts = einops.rearrange(cache[self.cfg.act_name], "batch seq_pos n_head d_head -> (batch seq_pos) (n_head d_head)")
                else:
                    acts = einops.rearrange(cache[self.cfg.act_name], "batch seq_pos d_act -> (batch seq_pos) d_act")
                assert acts.shape[-1] == self.cfg.d_data
                self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg.model_batch_size

        self.pointer = 0
        if self.cfg.subshuffle is None:
            self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.cfg.device)]
        else:
            ssize = self.buffer.shape[0] // self.cfg.subshuffle
            assert self.buffer.shape[0] % self.cfg.subshuffle == 0
            rperm = torch.randperm(ssize).to(self.cfg.device)
            # self.buffer[::self.cfg.subshuffle] = self.buffer[::self.cfg.subshuffle][rperm]
            perm = torch.arange(ssize)

            for i in range(self.cfg.subshuffle):
                self.buffer[i::self.cfg.subshuffle] = self.buffer[i::self.cfg.subshuffle][rperm][perm - i]
            # for i in range(self.cfg.subshuffle):
            #     self.buffer[i*ssize:(i+1)*ssize] = self.buffer[i*ssize:(i+1)*ssize][rperm]



        self.time_shuffling += time.time() - t0
        # torch.cuda.empty_cache()
    
    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg.batch_size]
        self.pointer += self.cfg.batch_size
        if self.pointer > int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio) - self.cfg.batch_size:
            # print("Refreshing the buffer!")
            self.refresh()

        return out


    @torch.no_grad()
    def freshen_buffer(self, fresh_factor=1, half_first=True):
        """
        Refreshes the buffer by moving the pointer and calling the refresh method.
        Warning: this burns data

        Args:
            fresh_factor (int): The factor by which the buffer should be refreshed.
            half_first (bool): Whether to refresh half of the buffer first.

        Returns:
            None
        """
        if half_first:
            n = (0.5 * self.cfg.buffer_size) // self.cfg.batch_size
            self.pointer += n * self.cfg.batch_size
            self.refresh()
        n = ((self.cfg.buffer_refresh_ratio) * self.cfg.buffer_size) // self.cfg.batch_size
        for _ in range(1 + int(fresh_factor / (self.cfg.buffer_refresh_ratio))):
            self.pointer += (n + 1) * self.cfg.batch_size
            self.refresh()


    @torch.no_grad()
    def skip_first_tokens_ratio(self, skip_percent, skip_batches):
        """
        Fast-forwards through skip_percent proportion of the data
        """
        self.token_pointer += int(self.all_tokens.shape[0] * skip_percent)
        self.first = True
        self.refresh()