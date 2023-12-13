#%%
from transformers import AutoTokenizer, AutoModel, pipeline, GPT2Model
from transformer_lens import HookedTransformer

# model = GPT2Model.from_pretrained("gpt2")


# W_embeds = model.get_input_embeddings
# print(type(W_embeds))

# print(W_embeds.bias.shape)


model = HookedTransformer.from_pretrained("gpt2-small")

W_embeds = model.embed.W_E.data
#%%

W_encode = torch.zeros((768, d_dict + W_embeds.shape[0]))










