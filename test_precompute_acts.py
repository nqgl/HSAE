import ae_on_heads_w_keith.z_sae as z_sae

model = z_sae.get_model(z_sae.AutoEncoderConfig())
tokens = z_sae.load_data(model, dataset="NeelNanda/c4-code-20k")
print(tokens[0:10])
