# %%
import torch

a = torch.zeros(5, 2)
i = (2 + torch.arange(20)) % 3

# i = torch.LongTensor(i)
b = torch.arange(50).float()

i = i.reshape(10, 2)
b = b.reshape(-1, 2)
b[:, 0] = -1
a.scatter_add_(0, i, b)

a
b
a
b[0]
a[i[:, 1]].shape
# %%

s = torch.zeros(5, 5)
s[2, 3] = 1
s[4, 1] = 1

ss = s.to_sparse()
ss.indices()[0]
# %%
batches = 2
d_data = 4
n_sae = 3

gate = torch.arange(batches * n_sae).reshape(batches, n_sae) % 3
x = torch.arange(batches * d_data).reshape(batches, d_data)

bgate = gate > 0

sgate = gate.to_sparse()
flat_indices = sgate.indices()
batch_idxs = flat_indices[0]
sae_idxs = flat_indices[1]
x_flat = x[batch_idxs].unsqueeze(-2)
x_flat, batch_idxs, sae_idxs


# %%


a = torch.eye(100).unsqueeze(0).expand(1000, -1, -1).cuda()
b = torch.zeros(1000, 100, 1000).cuda()

b[:, 4, 5] = 1
b[:, 5, 5] = 1
b[:, 9, 5] = 1
a.requires_grad = True
b.requires_grad = True
ac = a.to_sparse_csr()
bc = b.to_sparse_csr()

c = torch.sparse.addmm(ac, bc)
torch.optim.SparseAdam
s = c.sum()
s.backward()

a.grad


# %%
import torch_sparse


indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]])
valueA = torch.Tensor([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]])

indexB = torch.tensor([[0, 2], [1, 0]])
valueB = torch.Tensor([2, 4])


indexC, valueC = torch_sparse.spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)


# %%
import torch

# Example 3D tensor
a, b, c = 2, 3, 4  # dimensions
tensor_3d = torch.randn(a, b, c)  # random tensor for demonstration

# Reshape the tensor to 2D
tensor_2d = tensor_3d.view(a * b, b * c)

# %%
