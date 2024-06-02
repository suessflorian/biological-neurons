import torch
import torch.nn.functional as F
import torch.nn as nn
import snntorch as snn
from neurons import ParaLIF, ConvParaLIF
from utils import rate
from snntorch import utils as utls

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

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
    
class LeNet5_Flexible(nn.Module):
    def __init__(self, n_classes):
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
        self.fc3 = nn.Linear(84, n_classes)

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
    
class LeNet5_Representations_Flexible(nn.Module):
    def __init__(self, n_classes):
        '''
        Class for extracting representations after training.
        Shapes for extraction layer (MNIST) starting from 0:
        [256, 864], [256, 256], [256, 120], [256, 84], [256, 10]
        '''
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
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x, extraction_layer = None):
        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))
        if extraction_layer == 0: # After one convolution [256, 864]
            return x.view(x.shape[0], -1)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor for the fully connected layer
        if extraction_layer == 1: # After both convolutions [256, 256]
            return x
        x = self.bn3(self.fc1(x))
        x = F.relu(x)
        if extraction_layer == 2: # After one FC layer [256, 120]
            return x
        x = self.bn4(self.fc2(x))
        x = F.relu(x)
        if extraction_layer == 3: # After two FC layers [256, 84]
            return x
        x = self.fc3(x)
        return x # [256, 10]

class LeNet5_Representations_Flexible_CIFAR(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x, extraction_layer = None):
        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))
        if extraction_layer == 0: # After one convolution [256, 1176]
            return x.view(x.shape[0], -1)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(x.shape[0], -1)
        if extraction_layer == 1: # After both convolutions [256, 400]
            return x
        x = self.bn3(self.fc1(x))
        x = F.relu(x)
        if extraction_layer == 2: # After one FC layer [256, 120]
            return x
        x = self.bn4(self.fc2(x))
        x = F.relu(x)
        if extraction_layer == 3: # After two FC layers [256, 84]
            return x
        x = self.fc3(x)
        return x # [256, 10]

class SimpleSNN(nn.Module):
    # Slightly modified Florian's implementation
    def __init__(self, input_size, decay_rate=0.9, num_steps=10, seed=0):
        super(SimpleSNN, self).__init__()
        
        torch.manual_seed(seed)
        
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
        utls.reset(self.lif1)
        utls.reset(self.lif2)
        utls.reset(self.lif3)
        
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

class GeneralSNN(nn.Module):
    def __init__(self, layer_sizes, decay_rate=0.9, num_steps=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = len(layer_sizes) - 1
        
        for i, (l1, l2) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.add_module(f'fc{i}', nn.Linear(l1, l2))
            if i < self.n_layers - 1:
                self.add_module(f'lif{i}', snn.Leaky(beta=decay_rate, spike_grad=snn.surrogate.atan()))
        
        self.num_steps = num_steps
        
    def forward(self, images):
        mems = []
        for key in self._modules:
            if 'lif' in key:
                utls.reset(self._modules[key])
                mems.append(self._modules[key].init_leaky())
        
        flattened = images.view(images.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)
        output_spikes = []
        for step in range(self.num_steps):
            x = spike_train[step]
            for i in range(self.n_layers - 1):
                x = self._modules[f'fc{i}'](x)
                x, mems[i] = self._modules[f'lif{i}'](x, mems[i])
            x = self._modules[f'fc{i+1}'](x)
            output_spikes.append(x)
        return torch.stack(output_spikes, dim=0).sum(dim=0).softmax(dim=1)
    
class GeneralSNN2(nn.Module):
    ''' 
    Fixes some error for the training time computation;
    Otherwise it is identical.
    If it can be confirmed that this implementation works the same as GeneralSNN, 
    then GeneralSNN can be replaced with this implementation.
    '''
    def __init__(self, layer_sizes, decay_rate=0.9, num_steps=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = len(layer_sizes) - 1
        
        for i, (l1, l2) in enumerate(zip(layer_sizes, layer_sizes[1:])):
            self.add_module(f'fc{i}', nn.Linear(l1, l2))
            if i < self.n_layers - 1:
                self.add_module(f'lif{i}', snn.Leaky(beta=decay_rate, spike_grad=snn.surrogate.atan()))
        
        self.num_steps = num_steps
        
    def forward(self, images):
        mems = []
        for key in self._modules:
            if 'lif' in key:
                utls.reset(self._modules[key])
                mems.append(self._modules[key].init_leaky())

        flattened = images.view(images.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)
        output_spikes = []
        for step in range(self.num_steps):
            x = spike_train[step]
            for i in range(self.n_layers - 1):
                x = self._modules[f'fc{i}'](x)
                x, mems[i] = self._modules[f'lif{i}'](x, mems[i])
            x = self._modules[f'fc{self.n_layers - 1}'](x)  # Use self.n_layers - 1
            output_spikes.append(x)
        return torch.stack(output_spikes, dim=0).sum(dim=0).softmax(dim=1)

    
class SimpleSNN_EMNIST(nn.Module):
    # Slightly modified Florian's implementation 
    def __init__(self, input_size, decay_rate=0.9, num_steps=10, seed=0):
        super(SimpleSNN, self).__init__()
        
        torch.manual_seed(seed)
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

        self.fc4 = nn.Linear(2**7, 47)
        
        self.num_steps = num_steps

    def forward(self, images):  # images (batch, colour_channel, height, width)
        utls.reset(self.lif1)
        utls.reset(self.lif2)
        utls.reset(self.lif3)
        
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
    def __init__(self, input_size, decay_rate=0.9, num_steps=10, seed=0):
        super().__init__()
        
        torch.manual_seed(seed)
        
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
        utls.reset(self.lif1)
        utls.reset(self.lif2)
        utls.reset(self.lif3)
        utls.reset(self.lif4)
        
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
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False, num_steps= 10, 
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps

        # Set the spiking function
        self.paralif1 = ParaLIF(input_size, 2**9, device, spike_mode, recurrent, False, tau_mem,
                                tau_syn, time_step)
        self.paralif2 = ParaLIF(2**9, 2**8, device, spike_mode, recurrent, False, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(2**8, 2**7, device, spike_mode, recurrent, False, tau_mem,
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
    
class SimpleConvPara(nn.Module):
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        self.device = device
        
        self.convPara1 = ConvParaLIF(in_channel=num_steps, kernel = 1, stride = 1, padding = 0, 
                                     device=device, spike_mode=spike_mode, fire=True,
                                     tau_mem=tau_mem, tau_syn=tau_syn, time_step=time_step)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.convPara2 = ConvParaLIF(in_channel=num_steps, kernel = 1, stride = 1, padding = 0, 
                                     device=device, spike_mode=spike_mode, fire=True,
                                     tau_mem=tau_mem, tau_syn=tau_syn, time_step=time_step)
        self.convPara3 = ConvParaLIF(in_channel=num_steps, kernel = 1, stride = 1, padding = 0, 
                                     device=device, spike_mode=spike_mode, fire=True,
                                     tau_mem=tau_mem, tau_syn=tau_syn, time_step=time_step)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.thresh = nn.Threshold(0.75, 0)
        self.batchn = nn.BatchNorm2d(num_steps)
        self.batchn1 = nn.BatchNorm1d(num_steps)
        self.paralif1 = ParaLIF(input_size, 240, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step) # or 4x4
        self.paralif2 = ParaLIF(240, 84, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        batch, channel, height, width = images.size()
        final_spike_train = torch.zeros((batch, self.num_steps, height, width)).to(device=self.device)
        each_step = self.num_steps//channel

        for i in range(channel):
            flattened = images[:,i,:,:].view(batch, -1)  # (batch, colour_channel*width*height)
            spike_train = rate(flattened, num_steps=each_step)    
            spike_train = torch.swapaxes(spike_train, 0, 1)
            spike_train = spike_train.view(batch, -1, height, width)
            final_spike_train[:,i*each_step:(i+1)*each_step,:,:] = spike_train

        '''animation = final_spike_train[0]
        fig, ax = plt.subplots()
        anim = splt.animator(animation, fig, ax)
        plt.show()'''
        x = self.batchn(self.convPara1(final_spike_train)[1])
        x = torch.add(x, final_spike_train)
        #x = self.pool(x)
        #x = self.thresh(x)

        '''animation = x.cpu().detach()
        fig, ax = plt.subplots()
        anim = splt.animator(animation[0], fig, ax)
        plt.show()'''

        temp_x = self.batchn(self.convPara2(x)[1])
        x = torch.add(temp_x, x)
        x = self.pool1(x)
        x = self.thresh(x)
        #x[x>0] = 1

        temp_x = self.batchn(self.convPara3(x)[1])
        x = torch.add(temp_x, x)
        x = self.pool2(x)
        x = self.thresh(x)
        #x[x>0] = 1

        '''print(x.shape)
        animation = x.cpu().detach()
        fig, ax = plt.subplots()
        anim = splt.animator(animation[0], fig, ax)
        plt.show()'''
        
        x = x.view(batch, self.num_steps, -1)
        x = self.batchn1(self.paralif1(x))
        x = self.batchn1(self.paralif2(x))
        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x.softmax(dim=1)    

class ConvAndLifMnist(nn.Module):
    def __init__(self, input_size, device, decay_rate = 0.9, num_steps= 10, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, device=device)
        self.bn1 = nn.BatchNorm2d(6, device=device)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, device=device)
        self.bn2 = nn.BatchNorm2d(16, device=device)


        self.fc1 = nn.Linear(16*4*4, 120, device)
        self.lif1 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc2 = nn.Linear(120, 84, device)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc3 = nn.Linear(84, 10, device)

    def forward(self, images):
        # Convolutional layers
        x = self.bn1(self.conv1(images))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        # LIF feed forward
        utls.reset(self.lif1)
        utls.reset(self.lif2)
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)
        
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc1(spike_train[step])
            x, mem1 = self.lif1(x, mem1)
            
            x = self.fc2(x)
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            output_spikes.append(x)

        return torch.stack(output_spikes, dim=0).sum(dim=0)
    
class ConvAndLifMnist1(nn.Module):
    def __init__(self, input_size, device, decay_rate = 0.9, num_steps= 10, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)


        self.fc1 = nn.Linear(16*4*4, 120, device)
        self.bn3 = nn.BatchNorm1d(120)
        
        self.fc2 = nn.Linear(120, 84, device)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc3 = nn.Linear(84, 10, device)

    def forward(self, images):
        # Convolutional layers
        x = self.bn1(self.conv1(images))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(x.size(0), -1) # (batch, colour_channel*width*height)
        x = self.bn3(self.fc1(x))
        x = F.relu(x)

        # LIF feed forward
        utls.reset(self.lif2)
        
        mem2 = self.lif2.init_leaky()
        spike_train = rate(x, num_steps=self.num_steps)
        
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc2(spike_train[step])
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            output_spikes.append(x)

        return torch.stack(output_spikes, dim=0).sum(dim=0)
    
class ConvAndLifFashion(nn.Module):
    def __init__(self, input_size, device, decay_rate = 0.9, num_steps= 10, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)


        self.fc1 = nn.Linear(40*5*5, 120, device)
        self.lif1 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc2 = nn.Linear(120, 84, device)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc3 = nn.Linear(84, 10, device)

    def forward(self, images):
        # Convolutional layers
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        # LIF feed forward
        utls.reset(self.lif1)
        utls.reset(self.lif2)
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)
        
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc1(spike_train[step])
            x, mem1 = self.lif1(x, mem1)
            
            x = self.fc2(x)
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            output_spikes.append(x)

        return torch.stack(output_spikes, dim=0).sum(dim=0)
    
class ConvAndLifFashion1(nn.Module):
    def __init__(self, input_size, device, decay_rate = 0.9, num_steps= 10, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*5*5, 120, device)
        self.bn4 = nn.BatchNorm1d(120)
        
        self.fc2 = nn.Linear(120, 84, device)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc3 = nn.Linear(84, 10, device)

    def forward(self, images):
        # Convolutional layers
        # Convolutional layers
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))
        
        x = x.view(x.size(0), -1) # (batch, colour_channel*width*height)
        x = self.bn4(self.fc1(x))
        x = F.relu(x)

        # LIF feed forward
        utls.reset(self.lif2)
        
        mem2 = self.lif2.init_leaky()
        spike_train = rate(x, num_steps=self.num_steps)
        
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc2(spike_train[step])
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            output_spikes.append(x)

        return torch.stack(output_spikes, dim=0).sum(dim=0)

class ConvAndParaMnist(nn.Module):
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)

        self.paralif1 = ParaLIF(16*4*4, 120, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step) # or 4x4
        self.paralif2 = ParaLIF(120, 84, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif1(x)
        x = self.paralif2(x)
        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x.softmax(dim=1)
    
class ConvAndParaMnist2(nn.Module):
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16*4*4, 120, device)
        self.bn3 = nn.BatchNorm1d(120)
        self.paralif2 = ParaLIF(120, 84, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor for the fully connected layer
        x = self.bn3(self.fc1(x))
        x = F.relu(x)
        
        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif2(x)
        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x.softmax(dim=1)
    
class ConvAndParaMnist1(nn.Module):
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)

        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor for the fully connected layer
        x = self.bn3(self.fc1(x))
        x = F.relu(x)
        x = self.bn4(self.fc2(x))
        x = F.relu(x)

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x.softmax(dim=1)

class LeNet5_FASHION(nn.Module):
    def __init__(self, channels, para_input):
        super().__init__()
        self.para_input = para_input

        self.conv1 = nn.Conv2d(channels, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*para_input**2, 120) 
        self.bn4 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn5 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        x = x.view(-1, 40*self.para_input**2)  # Flatten the tensor for the fully connected layer
        x = self.bn4(self.fc1(x))
        x = F.relu(x)
        x = self.bn5(self.fc2(x))
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
class ConvAndParaFashion(nn.Module):
    def __init__(self, channels, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.paralif1 = ParaLIF(40*5*5, 120, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step) # or 4x4
        self.paralif2 = ParaLIF(120, 84, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif1(x)
        x = self.paralif2(x)
        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x
    
class ConvAndParaFashion2(nn.Module):
    def __init__(self, channels, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*5*5, 120, device)
        self.bn4 = nn.BatchNorm1d(120)
        self.paralif2 = ParaLIF(120, 84, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 40*5**2)  # Flatten the tensor for the fully connected layer
        x = self.bn4(self.fc1(x))
        x = F.relu(x)
        
        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif2(x)
        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x
    
class ConvAndParaFashion1(nn.Module):
    def __init__(self, channels, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*5*5, 120) 
        self.bn4 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn5 = nn.BatchNorm1d(84)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        x = x.view(-1, 40*5**2)  # Flatten the tensor for the fully connected layer
        x = self.bn4(self.fc1(x))
        x = F.relu(x)
        x = self.bn5(self.fc2(x))
        x = F.relu(x)

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x

class ConvAndParaKmnist(nn.Module):
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.paralif1 = ParaLIF(40*5*5, 120, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step) # or 4x4
        self.paralif2 = ParaLIF(120, 84, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif1(x)
        x = self.paralif2(x)
        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x

class ConvAndParaKmnist1(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3):
        super().__init__()
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*5*5, 120) 
        self.bn4 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn5 = nn.BatchNorm1d(84)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        x = x.view(-1, 40*5*5)  # Flatten the tensor for the fully connected layer
        x = self.bn4(self.fc1(x))
        x = F.relu(x)
        x = self.bn5(self.fc2(x))
        x = F.relu(x)

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x
    
class ConvAndParaKmnist2(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3):
        super().__init__()
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*5*5, 120, device)
        self.bn4 = nn.BatchNorm1d(120)
        self.paralif2 = ParaLIF(120, 84, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 40*5*5)  # Flatten the tensor for the fully connected layer
        x = self.bn4(self.fc1(x))
        x = F.relu(x)
        
        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif2(x)
        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x

class ConvAndLifKmnist(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, input_size, device, decay_rate = 0.9, num_steps= 10):
        super().__init__()
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)


        self.fc1 = nn.Linear(40*5*5, 120, device)
        self.lif1 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc2 = nn.Linear(120, 84, device)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc3 = nn.Linear(84, 10, device)

    def forward(self, images):
        # Convolutional layers
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        # LIF feed forward
        utls.reset(self.lif1)
        utls.reset(self.lif2)
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)
        
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc1(spike_train[step])
            x, mem1 = self.lif1(x, mem1)
            
            x = self.fc2(x)
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            output_spikes.append(x)

        return torch.stack(output_spikes, dim=0).sum(dim=0)
    
class ConvAndLifKmnist1(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, input_size, device, decay_rate = 0.9, num_steps= 10):
        super().__init__()
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*5*5, 120, device)
        self.bn4 = nn.BatchNorm1d(120)
        
        self.fc2 = nn.Linear(120, 84, device)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc3 = nn.Linear(84, 10, device)

    def forward(self, images):
        # Convolutional layers
        # Convolutional layers
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))
        
        x = x.view(x.size(0), -1) # (batch, colour_channel*width*height)
        x = self.bn4(self.fc1(x))
        x = F.relu(x)

        # LIF feed forward
        utls.reset(self.lif2)
        
        mem2 = self.lif2.init_leaky()
        spike_train = rate(x, num_steps=self.num_steps)
        
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc2(spike_train[step])
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            output_spikes.append(x)

        return torch.stack(output_spikes, dim=0).sum(dim=0)

class ConvAndParaSVHN(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, channels, para_input, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3):
        super().__init__()
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.paralif1 = ParaLIF(40*para_input*para_input, 120, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step) # or 4x4
        self.paralif2 = ParaLIF(120, 84, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif1(x)
        x = self.paralif2(x)
        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x
    
class ConvAndParaSVHN2(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, channels, para_input, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3):
        super().__init__()
        
        self.num_steps = num_steps
        self.fc_inp = para_input
        
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*para_input*para_input, 120, device)
        self.bn4 = nn.BatchNorm1d(120)
        self.paralif2 = ParaLIF(120, 84, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 40*self.fc_inp**2)  # Flatten the tensor for the fully connected layer
        x = self.bn4(self.fc1(x))
        x = F.relu(x)
        
        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif2(x)
        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x
    
class ConvAndParaSVHN1(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, channels, para_input, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3):
        super().__init__()
        
        self.num_steps = num_steps
        self.fc_inp = para_input
        
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*para_input*para_input, 120) 
        self.bn4 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn5 = nn.BatchNorm1d(84)
        self.paralif3 = ParaLIF(84, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        x = x.view(-1, 40*self.fc_inp**2)  # Flatten the tensor for the fully connected layer
        x = self.bn4(self.fc1(x))
        x = F.relu(x)
        x = self.bn5(self.fc2(x))
        x = F.relu(x)

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        x = self.paralif3(x)
        x = torch.mean(x,1)
        return x

class ConvAndLifSVHN(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, input_size, device, decay_rate = 0.9, num_steps= 10):
        super().__init__()
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)


        self.fc1 = nn.Linear(40*6*6, 120, device)
        self.lif1 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc2 = nn.Linear(120, 84, device)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc3 = nn.Linear(84, 10, device)

    def forward(self, images):
        # Convolutional layers
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))

        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        # LIF feed forward
        utls.reset(self.lif1)
        utls.reset(self.lif2)
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)
        
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc1(spike_train[step])
            x, mem1 = self.lif1(x, mem1)
            
            x = self.fc2(x)
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            output_spikes.append(x)

        return torch.stack(output_spikes, dim=0).sum(dim=0)
    
class ConvAndLifSVHN1(nn.Module):
    torch.manual_seed(1123)
    def __init__(self, input_size, device, decay_rate = 0.9, num_steps= 10):
        super().__init__()
        
        self.num_steps = num_steps
        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 40, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(40)

        self.fc1 = nn.Linear(40*6*6, 120, device)
        self.bn4 = nn.BatchNorm1d(120)
        
        self.fc2 = nn.Linear(120, 84, device)
        self.lif2 = snn.Leaky(
            beta=decay_rate,
            spike_grad=snn.surrogate.atan()
        )
        
        self.fc3 = nn.Linear(84, 10, device)

    def forward(self, images):
        # Convolutional layers
        # Convolutional layers
        x = self.bn1(self.conv1(images))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))
        
        x = x.view(x.size(0), -1) # (batch, colour_channel*width*height)
        x = self.bn4(self.fc1(x))
        x = F.relu(x)

        # LIF feed forward
        utls.reset(self.lif2)
        
        mem2 = self.lif2.init_leaky()
        spike_train = rate(x, num_steps=self.num_steps)
        
        output_spikes = []
        for step in range(self.num_steps):
            x = self.fc2(spike_train[step])
            x, mem2 = self.lif2(x, mem2)
            
            x = self.fc3(x)
            output_spikes.append(x)

        return torch.stack(output_spikes, dim=0).sum(dim=0)

class ConvAndParaCifar(nn.Module):
    def __init__(self, input_size, device, spike_mode='SB', recurrent=False,
                 num_steps= 10, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps

        # VGG9
        self.cnn11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.cnn21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.cnn31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*4*4, 1024, device=device)
        self.fc2 = nn.Linear(1024, 10, device=device)

        self.paralif1 = ParaLIF(256*4*4, 1024, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step) # or 4x4
        self.paralif2 = ParaLIF(1024, 10, device, spike_mode, recurrent, fire, tau_mem,
                                tau_syn, time_step)

    def forward(self, images):
        x = self.bn1(self.cnn11(images))
        x = self.bn1(self.cnn12(x))
        x = self.avgpool1(F.relu(x))

        x = self.bn2(self.cnn21(x))
        x = self.bn2(self.cnn22(x))
        x = self.avgpool2(F.relu(x))

        x = self.bn3(self.cnn31(x))
        x = self.bn3(self.cnn32(x))
        x = self.bn3(self.cnn33(x))
        x = self.avgpool3(F.relu(x))

        '''x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x'''

        flattened = x.view(x.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        x = torch.swapaxes(spike_train, 0, 1)

        '''animation = spike_train[0]
        fig, ax = plt.subplots()
        anim = splt.animator(animation, fig, ax)
        plt.show()'''
        '''animation = x.cpu().detach()
        fig, ax = plt.subplots()
        anim = splt.animator(animation[0], fig, ax)
        plt.show()'''
        '''animation = x.cpu().detach()
        fig, ax = plt.subplots()
        anim = splt.animator(animation[0], fig, ax)
        plt.show()'''

        x = self.paralif1(x)
        x = self.paralif2(x)
        x = torch.mean(x,1)
        return x   

class GeneralParaLIF(nn.Module):
    def __init__(self, layer_sizes, device, spike_mode='SB', recurrent=False, num_steps=10, 
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, seed=1123):
        super().__init__()
        
        torch.manual_seed(seed)
        
        self.num_steps = num_steps
        self.device = device
        self.spike_mode = spike_mode
        self.recurrent = recurrent
        self.fire = fire
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.time_step = time_step
        self.layer_sizes = layer_sizes
        
        self._create_layers()
        
    
    def forward(self, images):
        flattened = images.view(images.size(0), -1)  # (batch, colour_channel*width*height)
        spike_train = rate(flattened, num_steps=self.num_steps)      
        spike_train = torch.swapaxes(spike_train, 0, 1)

        x = self.layers(spike_train)
        x = torch.mean(x,1)
        return x.softmax(dim=1)

    def _create_layers(self):
        layers = []
        for size1, size2 in zip(self.layer_sizes, self.layer_sizes[1:]):
            layers.append(ParaLIF(
                input_size=size1,
                hidden_size=size2,
                device=self.device,
                spike_mode=self.spike_mode, 
                recurrent=self.recurrent, 
                fire=self.fire, 
                tau_mem=self.tau_mem,
                tau_syn=self.tau_syn, 
                time_step=self.time_step
            ))
        self.layers = nn.Sequential(*layers)
        

















# ---------------------------------------------- Experimental models ----------------------------------------------


class EncConvParaLIF(nn.Module):
    '''
    A model to test out the randomness induced by rate encoding and then averaging across the
    time dimension. This process brings the input back to its original shape and is then sent
    through typical ParaLIF layers.
    '''
    def __init__(self, layer_sizes, device, spike_mode='SB', recurrent=False, num_steps=10, 
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3):
    
        super().__init__()    
        self.num_steps = num_steps
        self.device = device
        self.spike_mode = spike_mode
        self.recurrent = recurrent
        self.fire = fire
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.time_step = time_step
        self.layer_sizes = layer_sizes
        
        self._create_layers()
        
        pool = nn.MaxPool2d
        self.conv = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.bn = nn.BatchNorm2d(10)
        self.pool = pool(kernel_size=2, stride=1)
    
    def forward(self, images):
        # Expectation across the time dimension, convolution, pooling
        images = rate(images, self.num_steps).mean(dim=0).to(self.device) #.swapaxes(0, 1)
        images = torch.tanh(self.bn(self.conv(images)))
        images = self.pool(images)
        
        # TYPICAL PARALIF ARCHITECTURE
        flattened = images.view(images.size(0), -1)
        spike_train = rate(flattened, self.num_steps).swapaxes(0, 1).to(self.device)
        x = self.layers(spike_train)
        x = torch.mean(x,1)
        return x.softmax(dim=1)
    
    
    def _create_layers(self):
        layers = []
        for size1, size2 in zip(self.layer_sizes, self.layer_sizes[1:]):
            layers.append(ParaLIF(
                input_size=size1,
                hidden_size=size2,
                device=self.device,
                spike_mode=self.spike_mode, 
                recurrent=self.recurrent, 
                fire=self.fire, 
                tau_mem=self.tau_mem,
                tau_syn=self.tau_syn, 
                time_step=self.time_step
            ))
        self.layers = nn.Sequential(*layers)
        

class Frankenstein(nn.Module):
    '''
    This model is an experiment of combining the LeNet architecture with ParaLIF neurons.
    This model achieves 90.46% test accuracy on Fashion after 20 epochs.
    
    There are five branches:
    
    1. Images -> rate encoding ->  ParaLIF layers -> output
    2. Images -> LeNet1 -> output
    3. LeNet1 pre-output representations -> ParaLIF layers -> output
    4. Images -> LeNet2 -> output
    5. Images -> LeNet3 -> output
    
    Notes:
    LeNet3 only uses 1x1 convolutions
    LeNet1 and 2 have different architectures
    '''
    def __init__(self, layer_sizes, device, spike_mode='SB', recurrent=False, num_steps=10, 
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, lenet_bias=1.):
    
        super().__init__()    
        self.num_steps = num_steps
        self.device = device
        self.spike_mode = spike_mode
        self.recurrent = recurrent
        self.fire = fire
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.time_step = time_step
        self.layer_sizes = layer_sizes
        
        self.layers = self._create_layers()
        
        self.conv = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.bn = nn.BatchNorm2d(10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # Alphas control the weighting of the different branches on the final output
        self.alpha = nn.Parameter(torch.tensor([1.]).to(device))
        self.alpha2 = nn.Parameter(torch.tensor([1. * lenet_bias]).to(device))
        self.alpha3 = nn.Parameter(torch.tensor([1.]).to(device))
        self.alpha4 = nn.Parameter(torch.tensor([1. * lenet_bias]).to(device))
        self.alpha5 = nn.Parameter(torch.tensor([1. * lenet_bias]).to(device))
        
        # These are the layer sizes for the ParaLIF model learning from the first LeNet's representations
        new_layer_sizes = (84, 256, 512, 128, 32, 10)
        self.layers2 = self._create_layers(new_layer_sizes)
        
        # LENET
        self.lenet_conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.lenet_bn1 = nn.BatchNorm2d(6)
        self.lenet_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lenet_conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.lenet_bn2 = nn.BatchNorm2d(16)
        self.lenet_fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.lenet_bn3 = nn.BatchNorm1d(120)
        self.lenet_fc2 = nn.Linear(120, 84)
        self.lenet_bn4 = nn.BatchNorm1d(84)
        self.lenet_fc3 = nn.Linear(84, 10)
        
        # LENET2
        self.lenet2_conv1 = nn.Conv2d(1, 24, kernel_size=7, stride=1, padding=0)
        self.lenet2_bn1 = nn.BatchNorm2d(24)
        self.lenet2_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.lenet2_conv2 = nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0)
        self.lenet2_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.lenet2_bn2 = nn.BatchNorm2d(48)
        self.lenet2_fc1 = nn.Linear(48 * 3 * 3, 128) 
        self.lenet2_bn3 = nn.BatchNorm1d(128)
        self.lenet2_fc2 = nn.Linear(128, 64)
        self.lenet2_bn4 = nn.BatchNorm1d(64)
        self.lenet2_fc3 = nn.Linear(64, 10)
        
        # LENET3 - only 1x1 convolutions here
        
        self.lenet3_conv1 = nn.Conv2d(1, 64, kernel_size=1)
        self.lenet3_bn1 = nn.BatchNorm2d(64)
        self.lenet3_pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.lenet3_conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.lenet3_bn2 = nn.BatchNorm2d(128)
        self.lenet3_fc1 = nn.Linear(128 * 3 * 3, 120) 
        self.lenet3_bn3 = nn.BatchNorm1d(120)
        self.lenet3_fc2 = nn.Linear(120, 84)
        self.lenet3_bn4 = nn.BatchNorm1d(84)
        self.lenet3_fc3 = nn.Linear(84, 10)
        
        
        self.output = nn.Linear(5*10, 10)
        self.output_bn = nn.BatchNorm1d(10)
    
    def forward(self, original_images):
        # # EXPECTATION PART
        
        # random_images = rate(original_images, self.num_steps).mean(dim=0).to(self.device) #.swapaxes(0, 1)
        
        # # CONVOLUTIONS
        # random_images = torch.tanh(self.bn(self.conv(random_images)))
        # random_images = self.pool(random_images)
        
        
        # # ParaLIF Branch
        
        flattened = original_images.view(original_images.size(0), -1)
        spike_train = rate(flattened, self.num_steps).swapaxes(0, 1).to(self.device)
        x = self.layers(spike_train)
        x = torch.mean(x,1)
        x = x.softmax(dim=1)
        
        
        # LeNet branch
        
        x2 = self.lenet_bn1(self.lenet_conv1(original_images))
        x2 = self.lenet_pool(F.relu(x2))
        x2 = self.lenet_bn2(self.lenet_conv2(x2))
        x2 = self.lenet_pool(F.relu(x2))
        x2 = x2.view(-1, 16 * 4 * 4)
        x2 = self.lenet_bn3(self.lenet_fc1(x2))
        x2 = F.relu(x2)
        x2 = self.lenet_bn4(self.lenet_fc2(x2))
        
        lenet_representations = x2.clone().detach()
        
        x2 = F.relu(x2)
        x2 = self.lenet_fc3(x2)
        
        
        # LeNet representation ParaLIF
        
        spike_train2 = rate(lenet_representations, self.num_steps).swapaxes(0, 1).to(self.device)
        x3 = self.layers2(spike_train2)
        x3 = torch.mean(x3,1)
        x3 = x3.softmax(dim=1)
        
        # LeNet 2 branch
        x4 = self.lenet2_bn1(self.lenet2_conv1(original_images))
        x4 = self.lenet2_pool(F.tanh(x4))
        x4 = self.lenet2_bn2(self.lenet2_conv2(x4))
        x4 = self.lenet2_pool2(F.tanh(x4))
        x4 = x4.view(-1, 48 * 3 * 3)
        x4 = self.lenet2_bn3(self.lenet2_fc1(x4))
        x4 = F.relu(x4)
        x4 = self.lenet2_bn4(self.lenet2_fc2(x4))        
        x4 = F.relu(x4)
        x4 = self.lenet2_fc3(x4)
        
        # LeNet 3 branch
        x5 = self.lenet3_bn1(self.lenet3_conv1(original_images))
        x5 = self.lenet3_pool(F.relu(x5))
        x5 = self.lenet3_bn2(self.lenet3_conv2(x5))
        x5 = self.lenet3_pool(F.relu(x5))
        x5 = x5.view(-1, 128 * 3 * 3)
        x5 = self.lenet3_bn3(self.lenet3_fc1(x5))
        x5 = F.relu(x5)
        x5 = self.lenet3_bn4(self.lenet3_fc2(x5))        
        x5 = F.relu(x5)
        x5 = self.lenet3_fc3(x5)
        
        
        # final = torch.cat((x, x2, x3, x4, x5), dim=1)
        # final = F.relu(self.output_bn(self.output(final)))
        # return final.softmax(dim=1)
               
        
        
        # return (self.alpha * x + self.alpha2 * x2 + self.alpha3 * x3).softmax(dim=1)
        return (
            self.alpha * x # ParaLIF
            + self.alpha2 * x2 # LeNet1
            + self.alpha3 * x3 # ParaLIF on LeNet representations
            + self.alpha4 * x4 # LeNet2
            + self.alpha5 * x5 # LeNet3
            ).softmax(dim=1)
        
    
    def _create_layers(self, new_layer_sizes=None):
        layers = []
        layer_sizes = self.layer_sizes if new_layer_sizes is None else new_layer_sizes
        for size1, size2 in zip(layer_sizes, layer_sizes[1:]):
            layers.append(ParaLIF(
                input_size=size1,
                hidden_size=size2,
                device=self.device,
                spike_mode=self.spike_mode, 
                recurrent=self.recurrent, 
                fire=self.fire, 
                tau_mem=self.tau_mem,
                tau_syn=self.tau_syn, 
                time_step=self.time_step
            ))
        return nn.Sequential(*layers)
