import osae
from osae import AutoEncoder, AutoEncoderConfig
# import __main__
# setattr(__main__, "AutoEncoder", osae.AutoEncoder)
import torch
itern = 811
ae = osae.AutoEncoder.load("./models", "sae_id_6-64_0-64_2", iter=itern)
ae.cuda()
stats = osae.AutoEncoder.load_stats("./models", "sae_id_6-64_0-64_2", iter=itern)
m = torch.tensor(stats["m"], device="cuda")
print(stats.keys())
# print("l2", stats["d"]["l2"])
v = torch.eye(ae.cfg.d_dict, device="cuda")
ae.decode(v)
print(m.shape)
f = ae.decode(v)
print(f.shape)
m = torch.nn.functional.normalize(m, dim=-2)
f = torch.nn.functional.normalize(f, dim=-2)

# for i in range(100, 0, -1):
#     f1 = f[i, :]
#     print(f1.shape)
#     print(m @ f1)
#     print(f1 @ m.T)
#     print((m @ f1).shape)

#     import matplotlib.pyplot as plt

#     # Your existing code...

#     result = m @ f1

#     # Plotting the bar graph
#     plt.bar(range(len(result)), result.cpu().detach().numpy())
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.title('m @ f1')
#     plt.show()
# heatmap of m @ f
features_similarity = m @ f.T
import seaborn as sns
max_args = features_similarity.argmax(dim=-1)
argsort_max = max_args.argsort()
features_similarity = features_similarity[argsort_max, :]
sns.heatmap(features_similarity.cpu().detach().numpy())


vm = torch.eye(m.shape[-2])
vm = vm.cuda()
vm = torch.nn.functional.normalize(vm, dim=-2)
vm = vm @ m
vin = ae.encode(vm)



import matplotlib.pyplot as plt

# Your existing code...

# Plotting the bar graph
result = vin[0]
print(vin.shape)
print(sum(vin))
plt.bar(range(len(result)), result.cpu().detach().numpy())
for i in range(200, 0, -1):
    plt.plot(vin[i, :].cpu().detach().numpy(), label=f"{i}")
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('m @ f1')
plt.show()

