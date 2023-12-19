#%%
import matplotlib.pyplot as plt
from nqgl.sae.sae import AutoEncoder, AutoEncoderConfig
from nqgl.sae.toy_models import two_features
from importlib import reload

cfg = AutoEncoderConfig(site="toy_model", d_data=128, layer=1, gram_shmidt_trail = 512, num_to_resample = 4,
                                l1_coeff=28e-4, dict_mult=10 / 128 + 1e-4, batch_size=128, beta2=0.999,
                                lr=1e-3) 

plt.show()
# toycfg = ToyModelConfig(d_data = cfg.d_data, n_features=1024, num_correlation_rounds=2, batch_size=cfg.batch_size)
# toy = ToyModel(toycfg)
toy = two_features.get_simple_hierarchy_model(d_data = cfg.d_data)
toy.correlations = []
toy.cfg.initial_features = 1
toy.f_probs[:4] = 0.5

# %%
encoder = AutoEncoder.load_latest()
# %%
encoder
from nqgl.sae.analysis import visualize_features
def view():
    reload(visualize_features)
    visualize_features.visualize_by_heatmap(toy, encoder)

# %%
import time
for i in range(100):
    time.sleep(1)
    view()

# %%
