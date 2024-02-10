# bsr does not work, requires batches have same nse

import torch
from torch.sparse import _triton_ops as sto


class SampledAddmmBSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mat2, mat1):
        ctx.save_for_backward(input, mat2, mat1)
        return sto.sampled_addmm(input, mat2, mat1)

    @staticmethod
    def backward(ctx, grad_output):
        input, mat2, mat1 = ctx.saved_tensors
        grad_input, grad_mat2, grad_mat1 = sampled_addmm_backward(
            grad_output, input, mat2, mat1
        )
        print("backward")
        return grad_input.to_dense(), grad_mat2, grad_mat1


def sampled_addmm_backward(grad_output, input, mat2, mat1):
    mask = input.to_dense() != 0
    print(input.layout, mat2.layout, mat1.layout)
    grad_mat1 = mat2.t() @ (grad_output * mask)
    grad_mat2 = (grad_output * mask) @ mat1.t()
    grad_input = grad_output
    return grad_input, grad_mat2, grad_mat1
