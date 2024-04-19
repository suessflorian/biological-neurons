import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import spikegen, surrogate

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNet_LIF(nn.Module):
    def __init__(self, decay_rate=0.9, window=20):
        super(LeNet_LIF, self).__init__()
        self.window = window

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.lif1 = snn.Leaky(beta=decay_rate, spike_grad=surrogate.atan())
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lif2 = snn.Leaky(beta=decay_rate, spike_grad=surrogate.atan())


        self.fc1   = nn.Linear(16*5*5, 120)
        self.lif3 = snn.Leaky(beta=decay_rate, spike_grad=surrogate.atan())
        self.fc2   = nn.Linear(120, 84)
        self.lif4 = snn.Leaky(beta=decay_rate, spike_grad=surrogate.atan())
        self.fc3   = nn.Linear(84, 10)
        self.lif5 = snn.Leaky(beta=decay_rate, spike_grad=surrogate.atan())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif3.init_leaky()
        mem5 = self.lif3.init_leaky()
            
        # Rate encode the input
        train = spikegen.rate(x, num_steps=self.window)
        output_spikes = []
        for step in range(self.window):
            out, mem1 = self.lif1(self.conv1(train[step]), mem1)
            out = F.max_pool2d(out, 2)
            out, mem2 = self.lif2(self.conv1(out), mem2)
            out = F.max_pool2d(out, 2)
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out, mem3 = self.lif3(self.fc1(out), mem3)
            out, mem4 = self.lif4(self.fc2(out), mem4)
            out, mem5 = self.lif5(self.fc3(out), mem5)
            output_spikes.append(out)

        summed_spikes = torch.stack(output_spikes).sum(0)
        return F.softmax(summed_spikes, dim=1)


class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6*in_channels, 16*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5*in_channels, 120*in_channels),
            nn.Tanh(),
            nn.Linear(120*in_channels, 84*in_channels),
            nn.Tanh(),
            nn.Linear(84*in_channels, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
