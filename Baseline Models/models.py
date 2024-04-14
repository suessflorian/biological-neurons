import torch
import torch.nn.functional as F
import torch.nn as nn
import snntorch as snn
from neurons import ParaLIF
from utils import rate

class LeNet5_CIFAR(nn.Module):
    def __init__(self):
        super(LeNet5_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor for the fully connected layer
        x = self.bn3(self.fc1(x))
        x = F.relu(x)
        x = self.bn4(self.fc2(x))
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
class LeNet5_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor for the fully connected layer
        x = self.bn3(self.fc1(x))
        x = F.relu(x)
        x = self.bn4(self.fc2(x))
        x = F.relu(x)
        x = self.fc3(x)
        return x
    

class SimpleSNN(nn.Module):
    # Slightly modified Florian's implementation
    torch.manual_seed(0) #  deterministic weight initials for perceptrons 
    def __init__(self, input_size, decay_rate=0.9, num_steps=10):
        super(SimpleSNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 2**9)
        self.lif1 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        self.fc2 = nn.Linear(2**9, 2**8)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        self.fc3 = nn.Linear(2**8, 2**7)
        self.lif3 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )

        self.fc4 = nn.Linear(2**7, 10)
        
        self.num_steps = num_steps

    def forward(self, images):  # images (batch, colour_channel, height, width)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        flattened = images.view(images.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc1(spike_train[step])
            x, mem1 = self.lif1(x, mem1)
            
            x = self.fc2(x)
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            x, mem3 = self.lif3(x, mem3)

            x = self.fc4(x)
            output_spikes.append(x)
        return torch.stack(output_spikes, dim=0).sum(dim=0).softmax(dim=1)
    
class LargerSNN(nn.Module):
    # Slightly modified Florian's implementation
    torch.manual_seed(0) #  deterministic weight initials for perceptrons 
    def __init__(self, input_size, decay_rate=0.9, num_steps=10):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 2**11)
        self.lif1 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        self.fc2 = nn.Linear(2**11, 2**10)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        self.fc3 = nn.Linear(2**10, 2**8)
        self.lif3 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )

        self.fc4 = nn.Linear(2**8, 2**7)
        
        self.lif4 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )

        self.fc5 = nn.Linear(2**7, 10)
        
        self.num_steps = num_steps

    def forward(self, images):  # images (batch, colour_channel, height, width)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        flattened = images.view(images.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc1(spike_train[step])
            x, mem1 = self.lif1(x, mem1)
            
            x = self.fc2(x)
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            x, mem3 = self.lif3(x, mem3)
            
            x = self.fc4(x)
            x, mem4 = self.lif4(x, mem4)

            x = self.fc5(x)
            output_spikes.append(x)
        return torch.stack(output_spikes, dim=0).sum(dim=0).softmax(dim=1)
    
class SimpleParaLif(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False, num_steps= 10, 
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3):
        super().__init__()
        
        self.num_steps = num_steps

        # Set the spiking function
        self.paralif1 = ParaLIF(input_size, 2**9, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif2 = ParaLIF(2**9, 2**8, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(2**8, 2**7, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif4 = ParaLIF(2**7, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
    
    def forward(self, images):
        flattened = images.view(images.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        spike_train = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif1(spike_train)
        x = self.paralif2(x)
        x = self.paralif3(x)
        x = self.paralif4(x)
        x = torch.mean(x,1)
        return x.softmax(dim=1)
    
    
# class testParaLIF(nn.Module):
#     torch.manual_seed(1123)
#     def __init__(self, input_size, device, spike_mode='SB', recurrent=False, num_steps=10, 
#                  fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3):
#         super().__init__()
        
#         self.num_steps = num_steps

#         # Set the spiking function
#         self.paralif1 = ParaLIF(input_size, 2**10, device, spike_mode, recurrent, fire, tau_mem,
#                                 tau_syn, time_step)
#         self.paralif2 = ParaLIF(2**10, 2**9, device, spike_mode, recurrent, fire, tau_mem,
#                                 tau_syn, time_step)
#         self.paralif3 = ParaLIF(2**9, 2**8, device, spike_mode, recurrent, fire, tau_mem,
#                                 tau_syn, time_step) #
#         self.paralif4 = ParaLIF(2**8, 2**7, device, spike_mode, recurrent, fire, tau_mem,
#                                 tau_syn, time_step) #
#         self.paralif5 = ParaLIF(2**7, 2**6, device, spike_mode, recurrent, fire, tau_mem,
#                                 tau_syn, time_step)
#         self.paralif6 = ParaLIF(2**6, 10, device, spike_mode, recurrent, fire, tau_mem,
#                                 tau_syn, time_step)
    
#     def forward(self, images):
#         flattened = images.view(images.size(0), -1)  # (batch, colour_channel*width*height)
#         spike_train = rate(flattened, num_steps=self.num_steps)      
#         spike_train = torch.swapaxes(spike_train, 0, 1)

#         x = self.paralif1(spike_train)
#         x = self.paralif2(x)
#         x = self.paralif3(x)
#         x = self.paralif4(x)
#         x = self.paralif5(x)
#         x = self.paralif6(x)
#         x = torch.mean(x,1)
#         return x.softmax(dim=1)
    