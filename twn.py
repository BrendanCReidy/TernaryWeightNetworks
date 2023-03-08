import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Methods
"""
def ternarize(tensor):
    delta = get_delta(tensor)
    alpha = get_alpha(tensor,delta)
    pos = torch.where(tensor>delta,1,tensor)
    neg = torch.where(pos<-delta,-1,pos)
    ternary = torch.where((neg>=-delta) & (neg<=delta),0,neg)
    return ternary*alpha


def get_alpha(tensor,delta,epsilon=1e-4):
    ndim = len(tensor.shape)
    view_dims = (-1,) + (ndim-1)*(1,)
    i_delta = (torch.abs(tensor)>delta)
    i_delta_count = i_delta.view(i_delta.shape[0],-1).sum(1)
    i_delta_count[i_delta_count==0] = 1 # fixed divide by zero issue that was causing nan loss... nvm
    tensor_thresh = torch.where((i_delta),tensor,0)
    alpha = (1/i_delta_count)*(torch.abs(tensor_thresh.view(tensor.shape[0],-1)).sum(1))
    alpha = alpha.view(view_dims)
    alpha[alpha<epsilon] = epsilon
    if not torch.min(alpha).item() > epsilon:
        raise ValueError('Loss boutta go nan')
    return alpha


def get_delta(tensor):
    ndim = len(tensor.shape)
    view_dims = (-1,) + (ndim-1)*(1,)
    n = tensor[0].nelement()
    norm = tensor.norm(1,ndim-1).view(tensor.shape[0],-1)
    norm_sum = norm.sum(1)
    delta = (0.7/n)*norm_sum
    return delta.view(view_dims)

"""
Gradients
"""
class TernaryLinearGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, ternarized):

        ternary_w = weight
        if ternarized:
            ternary_w = ternarize(weight)
        ctx.save_for_backward(input, weight, ternary_w)
        return torch.matmul(input, ternary_w) + bias

    @staticmethod
    def backward(ctx, grad_output):
        input,weight,ternary_w = ctx.saved_tensors
        d_input = torch.matmul(grad_output, torch.transpose(ternary_w,0,1))
        d_bias = grad_output.sum(dim=0)

        d_weight = torch.matmul(torch.transpose(input, 0, 1), grad_output)
        return d_input, d_weight, d_bias, None

class TernaryConv2DGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, groups, ternarized):
        ctx.stride=stride
        ctx.padding=padding
        ctx.groups=groups

        ternary_w = weight
        if ternarized:
            ternary_w = ternarize(weight)
        ctx.save_for_backward(input, weight, ternary_w, bias)
        return F.conv2d(input, ternary_w, bias, stride=stride, padding=padding, groups=groups)

    @staticmethod
    def backward(ctx, grad_output):
        input,weight,ternary_w,bias = ctx.saved_tensors
        stride=ctx.stride
        padding=ctx.padding
        groups=ctx.groups

        n_bias = bias.size()[0]
        d_bias = torch.reshape(grad_output, (n_bias,-1)).sum(dim=1)

        d_input = torch.nn.grad.conv2d_input(input.shape, ternary_w, grad_output, stride=stride, padding=padding, groups=groups)
        d_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, groups=groups)
        return d_input, d_weight, d_bias, None, None, None, None

class TernaryConv2DGradNoBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, groups, ternarized):

        ctx.stride=stride
        ctx.padding=padding
        ctx.groups=groups
        
        ternary_w = weight
        if ternarized:
            ternary_w = ternarize(weight)
        ctx.save_for_backward(input, weight, ternary_w)
        return F.conv2d(input, ternary_w, None, stride=stride, padding=padding, groups=groups)

    @staticmethod
    def backward(ctx, grad_output):
        input,weight,ternary_w = ctx.saved_tensors
        stride=ctx.stride
        padding=ctx.padding
        groups=ctx.groups

        d_input = torch.nn.grad.conv2d_input(input.shape, ternary_w, grad_output, stride=stride, padding=padding, groups=groups)
        d_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, groups=groups)
        return d_input, d_weight, None, None, None, None
"""
Layers
"""
class TernaryLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ternarized=True):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.weights.data = torch.clamp(self.weights.data,-1,1)
        self.ternarized = ternarized

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def ternarize(self):
        if self.ternarized:
            print("Warning: ternerize was called on already ternerized layer")
        self.ternarized = True

    def unternarize(self):
        if not self.ternarized:
            print("Warning: unternarize was called on already ternerized layer")
        self.ternarized = False

    def forward(self, input):
        return TernaryLinearGrad.apply(input, self.weights, self.bias, self.ternarized)

class TernaryConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True, groups=1, ternarized=True):
        super(TernaryConv2d, self).__init__()
        if type(kernel_size)==int:
            kernel_size = (kernel_size, kernel_size)
        if type(stride)==int:
            stride = (stride, stride)

        kernel_shape = (out_channels, in_channels//groups) + kernel_size

        self.kernel_size = kernel_size
        self.meta_data = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride=stride
        self.ternarized = ternarized
        self.padding=padding
        self.groups = groups
        self.weight = torch.nn.Parameter(torch.empty(kernel_shape))
        self.bias=None
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()
        self.weight.data = torch.clamp(self.weight.data,-1,1)

    def ternarize(self):
        if self.ternarized:
            print("Warning: ternerize was called on already ternerized layer")
        self.ternarized = True

    def unternarize(self):
        if not self.ternarized:
            print("Warning: unternarize was called on already ternerized layer")
        self.ternarized = False

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is not None:
            return TernaryConv2DGrad.apply(input, self.weight, self.bias, self.stride, self.padding, self.groups, self.ternarized)
        return TernaryConv2DGradNoBias.apply(input, self.weight, self.stride, self.padding, self.groups, self.ternarized)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn((4,2,3,3))
    delta = get_delta(x)
    alpha = get_alpha(x,delta)
    ternary = ternarize(x)


    x2 = torch.randn((4,3))
    delta = get_delta(x2)
    alpha = Alpha(x2,delta)
    alpha = get_alpha(x2,delta)
    ternary = ternarize(x2)
    