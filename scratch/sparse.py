#%%
import torch

a = torch.zeros(5, 2)
i = (2 + torch.arange(20)) % 3

# i = torch.LongTensor(i)
b = torch.arange(50).float()

i = i.reshape(10,2)
b = b.reshape(-1, 2)
b[:, 0] = -1
a.scatter_add_(0, i, b)

a
b
a
b[0]
a[i[:,1]].shape
#%%

s = torch.zeros(5, 5)
s[2,3] = 1
s[4,1] = 1

ss = s.to_sparse()
ss.indices()[0]
# %%
