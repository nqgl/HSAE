# by Glen Taggart @nqgl
import torch
import torch.nn.functional as F

class PositiveGradthruIdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp(min=0)

class NegativeGradthruIdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp(max=0)
    
def grad_clamped_identity_function(grad_min, grad_max):
    class GradClampedIdentityFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input

        @staticmethod
        def backward(ctx, grad_output):
            return torch.clamp(grad_output, min=grad_min, max=grad_max)
    return GradClampedIdentityFunction.apply


class GradSignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return torch.sign(grad_output)
    
def undying_relu(x, l=0.01, k=1):
    """x>0: normal relu
    0 > x > -k : gradient is scaled by l like a leaky relu
    -k > x : gradient only pushes towards x increasing, so that it is able to 'un-die'"""
    y_forward = F.relu(x)
    y_backward1 = x * (x > 0)
    y_backward2 = l * x * (torch.logical_and(x <= 0, x > -k))  
    y_backward3 = l * NegativeGradthruIdentityFunction.apply(x) * (x <= -k)
    y_backward = y_backward1 + y_backward2 + y_backward3
    return y_backward + (y_forward - y_backward).detach()

def undying_relu_extra_negative(x, l=0.001, k=0.01):
    """x>0: normal relu
    0 > x > -k : gradient is scaled by l like a leaky relu but 2x gradient on negative side,
                     so it stays dead unless it's really needed
    -k > x : gradient only pushes towards x increasing, so that it is able to 'un-die'"""
    y_forward = F.relu(x)
    y_backward1 = x * (x > 0)
    y_backward2 = l * (x + PositiveGradthruIdentityFunction.apply(x)) * (torch.logical_and(x <= 0, x > -k)) / 2
    y_backward3 = l * NegativeGradthruIdentityFunction.apply(x) * (x <= -k)
    y_backward = y_backward1 + y_backward2 + y_backward3
    return y_backward + (y_forward - y_backward).detach()



def undying_relu_2phases(x, l=0.01, k=0):
    """only gradients bigger than 0 in the positive direction come thru the 0 side of the relu"""

    y_forward = F.relu(x)
    y_backward1 = x * (x >= 0)
    y_backward2 = l * NegativeGradthruIdentityFunction.apply(x) * (x < 0)  
    y_backward = y_backward1 + y_backward2
    return y_backward + (y_forward - y_backward).detach()


def undying_relu_2phase_leaky_gradient(x, l=0.01):
    """gradient looks like leaky relu, output looks like regular relu"""
    y_forward = F.relu(x)
    y_backward1 = x * (x > 0)
    y_backward2 = l * x * (x <= 0)
    y_backward = y_backward1 + y_backward2
    return y_backward + (y_forward - y_backward).detach()

# class UndyingReLU2Phases(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, k, l):
#         ctx.save_for_backward(input)
#         ctx.k = k
#         ctx.l = l
#         return input.clamp(min=0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         k = ctx.k
#         l = ctx.l
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = torch.where(grad_output[input < 0] + k > 0, torch.tensor(0.0, device=grad_input.device), (grad_output[input < 0] + k) * l)
#         grad_input[input > 0] = grad_output[input > 0]
#         return grad_input, None, None

# def undying_relu_2phases(x, l=0.001, k=0):
#     return UndyingReLU2Phases.apply(x, k, l)

def main():
    x_ = torch.arange(10)/10 - 0.5
    x_.requires_grad = True
    y = (undying_relu_2phases(x_, k = 0.) - 1) ** 2
    y.float().sum().backward(retain_graph=True)
    print(x_)
    print(x_.grad)
    k = 0.1
    l = 0.01
    x = x_

if __name__ == "__main__":
    main()