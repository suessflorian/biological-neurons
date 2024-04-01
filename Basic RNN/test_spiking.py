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
        self.storage = torch.zeros(self.in_features)
        self.weights = nn.Parameter(torch.empty(self.in_features, self.out_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))

        nn.init.xavier_uniform_(self.weights)
        nn.init.normal_(self.bias, mean=0.0, std=1.0)

    def forward(self, x, t):
        #input x is binary vector
        x = x + self.storage
        self.storage = x
        x = torch.matmul(self.storage/t, self.weights) + self.bias
        x = F.relu(x)
        x = torch.clip(x, 0, 1)
        x = torch.bernoulli(x)

        return x
    
x = torch.rand(40, 10)
y = x**2
t = 20

model = spikingNeuron(10, 10)

optim = torch.optim.SGD(model.parameters(), lr = 0.01)

y_pred = model(x, 0%t + 1)
loss = torch.sum((y_pred - y)**2)/1000
print(loss)

loss.backward()

optim.step()
optim.zero_grad()

print(x)
print(y_pred)
    