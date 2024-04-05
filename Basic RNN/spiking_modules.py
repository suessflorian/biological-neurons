import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
    
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x

class spikingNeuron(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ste = StraightThroughEstimator()
        self.storage = torch.zeros(self.in_features).to(device)
        self.weights = nn.Parameter(torch.empty(self.in_features, self.out_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))

        nn.init.xavier_uniform_(self.weights)
        nn.init.normal_(self.bias, mean=0.0, std=1.0)

    def forward(self, x, t):
        x = x + self.storage
        self.storage = x.detach()
        x = torch.matmul(x/(t+1), self.weights) + self.bias
        x = F.relu(x)
        x = torch.clip(x, 0, 1)
        x = torch.bernoulli(x)
        x = self.ste(x)

        return x
    