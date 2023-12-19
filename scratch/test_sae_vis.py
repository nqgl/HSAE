from repos import sae_visualizer
from sae.model import z_sae
ae = z_sae.AutoEncoder.load(171, save_dir="./models-from-remote/")
model = z_sae.get_model(ae.cfg)
tokens = z_sae.load_data(model, dataset="NeelNanda/c4-code-20k")



f = sae_visualizer.get_feature_data(ae, ae, model, tokens, 20)
print(f)



