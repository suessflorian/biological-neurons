import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F

class spikingNeuron(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.surrgrad = SurrGradSpike()
        self.storage = torch.ones(self.out_features).to(device)
        self.weights = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))

        nn.init.xavier_uniform_(self.weights)
        nn.init.normal_(self.bias, mean=0.0, std=1.0)   

    def forward(self, x):
        x = F.linear(x, self.weights, self.bias)
        x = F.relu(x)
        x = self.surrgrad.apply(x)

        #x = x + self.storage
        self.storage = x.clone().detach()
        
        return x
    
    def reset(self):
        self.storage = torch.zeros(self.out_features).to(self.device)

beta = 4

class HeavySide(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.clip(input, 0, 1)
        out = torch.ceil(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid_derivative = torch.sigmoid(beta*input)*(1-torch.sigmoid(beta*input))
        grad = grad_output*sigmoid_derivative
        return grad
    
class SurrGradSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.clip(input, 0, 1)
        out = torch.bernoulli(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #sigmoid_derivative = torch.sigmoid(beta*input)*(1-torch.sigmoid(beta*input))
        #grad = grad_output*sigmoid_derivative
        grad = grad_input*input
        return grad


    