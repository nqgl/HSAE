#%%
from transformers import AutoTokenizer, AutoModel, pipeline, GPT2Model
from transformer_lens import HookedTransformer
import torch
# model = GPT2Model.from_pretrained("gpt2")


# W_embeds = model.get_input_embeddings
# print(type(W_embeds))

# print(W_embeds.bias.shape)


model = HookedTransformer.from_pretrained("gpt2-small")

W_embeds = model.embed.W_E.data
print(W_embeds.shape)
#%%

# W_encode = torch.zeros((768, d_dict + W_embeds.shape[0]))



tokenizer = model.tokenizer

token_ids = torch.arange(0, 50257)


tokens_all = tokenizer.convert_ids_to_tokens(token_ids)




# %%
