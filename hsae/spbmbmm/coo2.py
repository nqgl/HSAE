import torch
from torch.sparse import _triton_ops as sto


def coo2_W(gate, W):
    batch = gate.shape[0]
    fullshape = (batch,) + W.shape
    b_coo2_gate = goodgate(gate, fullshape)
    return mult_spd(b_coo2_gate, W.unsqueeze(0).expand(*fullshape))


def goodgate(gate, goodshape):
    # values = spgate.values().expand(spgate.values().shape[:1] + goodshape[2:])
    index = (gate != 0).nonzero().t()
    vshape = (index.shape[1],) + goodshape[2:]
    values = torch.ones(1, 1, 1, device=gate.device, dtype=gate.dtype).expand(*vshape)
    return _index_make(
        index,
        values,
        goodshape,
    )


def coo2_x(gate, x):
    fullshape = gate.shape + (1, x.shape[1])
    b_coo2_gate = gate.unsqueeze(-1).unsqueeze(-1).expand(*fullshape).to_sparse(2)
    return mult_spd(b_coo2_gate, x.unsqueeze(-2).unsqueeze(-2).expand(*fullshape))


def coo2_b(gate, b):
    fullshape = gate.shape + (1, b.shape[1])
    b_coo2_gate = gate.unsqueeze(-1).unsqueeze(-1).expand(*fullshape).to_sparse(2)
    return mult_spd(b_coo2_gate, b.unsqueeze(-2).unsqueeze(0).expand(*fullshape))


def coo2transpose(mat):
    mat = mat.coalesce()
    shape = mat.shape[:-2] + (mat.shape[-1], mat.shape[-2])
    return _index_clone(mat, values=mat.values().transpose(-2, -1))


def _coo2matmul(mat2, mat1):
    mat1 = mat1.coalesce()
    mat2 = mat2.coalesce()
    assert mat2.layout == torch.sparse_coo == mat1.layout
    assert mat2.shape[-1] == mat1.shape[-2]
    assert torch.all(mat2.indices() == mat1.indices())
    shape = mat2.shape[:-1] + mat1.shape[-1:]
    out = torch.sparse_coo_tensor(
        mat2.indices(),
        mat2.values() @ mat1.values(),
        shape,
        device=mat2.device,
    )
    return out


def _coo2matmul_spd(mat2, mat1):
    mat2 = mat2.coalesce()
    assert mat2.layout == torch.sparse_coo
    assert mat1.layout == torch.strided
    assert mat2.shape[-1] == mat1.shape[-2]
    shape = mat2.shape[:-1] + mat1.shape[-1:]
    out = torch.sparse_coo_tensor(
        mat2.indices(),
        mat2.values() @ mat1[mat2.indices()[0], mat2.indices()[1]],
        shape,
        device=mat2.device,
    )
    return out


class BCOOd2_MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat2, mat1):
        mat2 = mat2
        mat1 = mat1
        # ctx.save_for_backward(mat2.detach(), mat1.detach())
        # ctx.save_for_backward(mat2.detach(), mat1.detach())
        ctx.save_for_backward(mat2, mat1)
        out = _coo2matmul(mat2, mat1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.to_dense()
        mat2, mat1 = ctx.saved_tensors
        # print(mat2.shape, mat1.shape, grad_output.shape)
        if grad_output.is_sparse:
            grad_output = grad_output.coalesce()
            grad_mat2 = coo2transpose(_coo2matmul(mat1, coo2transpose(grad_output)))
            grad_mat1 = _coo2matmul(coo2transpose(mat2), grad_output)
        else:
            grad_output_T = grad_output.transpose(-2, -1)
            grad_mat2 = coo2transpose(
                _coo2matmul_spd(mat1, grad_output.transpose(-2, -1))
            )
            grad_mat1 = _coo2matmul_spd(coo2transpose(mat2), grad_output)
        return grad_mat2, grad_mat1


def coo2matmul(mat2, mat1, joindim=None):
    if joindim is None:
        return BCOOd2_MM.apply(mat2, mat1)
    mat2, mat1 = _joindim(mat2, mat1, joindim)


def _joindim(mat2, mat1, joindim):
    mat1 = mat1.coalesce()
    mat2 = mat2.coalesce()
    assert mat2.layout == torch.sparse_coo == mat1.layout
    newshape = (mat2.shape[0], 1, mat2.shape[-2], mat1.shape[-1])


class IndexClone(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spinput, values):
        ctx.save_for_backward(values, spinput)
        return torch.sparse_coo_tensor(
            indices=spinput.indices(),
            values=values,
            size=spinput.shape[:-2] + values.shape[-2:],
            device=spinput.device,
        )

    @staticmethod
    def backward(ctx, grad_output):
        # print("grad_output", grad_output.shape, grad_output.layout)
        values, spinput = ctx.saved_tensors
        if not grad_output.is_sparse:
            return None, grad_output[spinput.indices()[0], spinput.indices()[1]]
        grad_output = grad_output.coalesce()
        spinput = spinput.coalesce()
        if torch.all(grad_output.indices() == spinput.indices()):
            return None, grad_output.values()
        assert False


def _index_make(index, values, size, device=None):
    device = device or values.device
    return torch.sparse_coo_tensor(
        indices=index,
        values=values,
        size=size,
        device=device,
    )


def index_clone(spinput, values):
    return _index_clone(spinput, values)
    return IndexClone.apply(spinput, values)


def _index_clone(spinput, values):
    return torch.sparse_coo_tensor(
        indices=spinput.indices(),
        values=values,
        size=spinput.shape[:-2] + values.shape[-2:],
        device=spinput.device,
    )


class COO2mult_spd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat2, mat1):
        ctx.save_for_backward(mat2, mat1)
        values = mat2.values() * mat1[mat2.indices()[0], mat2.indices()[1]]


class COO2mult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat2, mat1):
        ctx.save_for_backward(mat2, mat1)


def mult_spd(mat2, mat1):
    mat2 = mat2.coalesce()
    values = mat2.values() * mat1[mat2.indices()[0], mat2.indices()[1]]
    return index_clone(
        mat2,
        values,
    )
