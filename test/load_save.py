from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig
from nqgl.sae.hsae.hsae import HierarchicalAutoEncoder, HierarchicalAutoEncoderConfig
import time

test_time = int(time.time())
hname = f"hsae_test{test_time}"
sname = f"sae_test{test_time}"
hsae = HierarchicalAutoEncoder(HierarchicalAutoEncoderConfig(name=hname, d_data=4))

sae = AutoEncoder(AutoEncoderConfig(name=sname, d_data=4))

hsae.save(hname)
sae.save(sname)


hsae_loaded = HierarchicalAutoEncoder.load_latest()
sae_loaded = AutoEncoder.load_latest()

assert hsae_loaded.cfg.name == hname
assert sae_loaded.cfg.name == sname
