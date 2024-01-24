from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig
import einops
import circuitsvis
import transformer_lens
import torch
from matplotlib import pyplot as plt

ae = AutoEncoder.load(171, save_dir="/home/g/mats/sae/models-from-remote/")

print(f"encoder_size:{ae.W_enc.shape}")
print(f"decoder_size:{ae.W_dec.shape}")

model = z_sae.get_model(ae.cfg)

decoder = ae.W_dec
head_matricies = einops.rearrange(
    ae.W_dec, "d_dict (n_heads d_head) -> d_dict n_heads d_head", n_heads=8, d_head=64
)


first_five_features = decoder[250:255]

# plot the first 5 features
plt.figure()
for i in range(first_five_features.shape[0]):
    plt.plot(first_five_features[i].detach().cpu().numpy(), label=f"feature {i}")
    plt.legend()
    plt.title("First 5 features of the decoder")
    plt.show()


# abs_head_matricies = torch.abs(head_matricies)
# shape of feature_head_sum: (d_dict, n_heads)
feature_head_sum = torch.sum(head_matricies.abs(), -1)
fh_means = torch.mean(feature_head_sum, dim=-1, keepdim=True)
maximally_superposed = torch.ones_like(feature_head_sum) / 8


fh_diffs = feature_head_sum - fh_means
var = torch.mean(torch.pow(fh_diffs, 2.0))
std = torch.pow(var, 0.5)
zscores = fh_diffs / std

skews = torch.mean(torch.pow(zscores, 3.0), dim=-1)
print(skews.shape)

plt.figure()
plt.hist(skews.detach().cpu().numpy())
plt.title("Zscores of the head features")
plt.show()

max_skew = torch.argmax(skews)
print(max_skew)


plt.figure()
plt.plot(decoder[max_skew].detach().cpu().numpy(), label=f"max skew feature {max_skew}")
plt.legend()
plt.title("max skew")
plt.show()
