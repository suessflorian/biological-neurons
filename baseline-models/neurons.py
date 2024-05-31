import torch
import torch.nn.functional as F
import snntorch as snn
from utils import StochasticStraightThrough, GumbelSoftmax, SurrGradSpike, SpikingFunction
import numpy as np

# Base class implementation from https://github.com/NECOTIS/Parallelizable-Leaky-Integrate-and-Fire-Neuron/blob/main/neurons/base.py
class Base(torch.nn.Module):
    """
    Base class for creating a spiking neural network using PyTorch.

    Parameters:
    - input_size (int): size of input tensor
    - hidden_size (int): size of hidden layer
    - device (torch.device): device to use for tensor computations, such as 'cpu' or 'cuda'
    - recurrent (bool): flag to determine if the neurons should be recurrent
    - fire (bool): flag to determine if the neurons should fire spikes or not
    - tau_mem (float): time constant for the membrane potential
    - tau_syn (float): time constant for the synaptic potential
    - time_step (float): step size for updating the LIF model
    - debug (bool): flag to turn on/off debugging mode
    """
    def __init__(self, input_size, hidden_size, device, recurrent,
                 fire, tau_mem, tau_syn, time_step, debug):
        super(Base, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.recurrent = recurrent
        self.v_th = torch.tensor(1.0)
        self.fire = fire
        self.debug = debug
        self.nb_spike_per_neuron = torch.zeros(self.hidden_size, device=self.device)

        # Neuron time constants
        self.alpha = float(np.exp(-time_step/tau_syn))
        self.beta = float(np.exp(-time_step/tau_mem))

        # Fully connected layer for feedforward synapses
        self.fc = torch.nn.Linear(self.input_size, self.hidden_size, device=self.device)
        # Initializing weights
        torch.nn.init.kaiming_uniform_(self.fc.weight, a=0, mode='fan_in', nonlinearity='linear')
        torch.nn.init.zeros_(self.fc.bias)
        if self.debug: torch.nn.init.ones_(self.fc.weight)
        
        # Fully connected for recurrent synapses 
        if self.recurrent:
            self.fc_recu = torch.nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
            # Initializing weights
            torch.nn.init.kaiming_uniform_(self.fc_recu.weight, a=0, mode='fan_in', nonlinearity='linear')
            torch.nn.init.zeros_(self.fc_recu.bias)
            if self.debug: torch.nn.init.ones_(self.fc_recu.weight)
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.fc.weight, a=0, mode='fan_in', nonlinearity='linear')
        torch.nn.init.zeros_(self.fc.bias)

#ParaLIF implementation from https://github.com/NECOTIS/Parallelizable-Leaky-Integrate-and-Fire-Neuron/blob/main/neurons/paralif.py
class ParaLIF(Base):
    """
    Class for implementing a Parallelizable Leaky Integrate-and-Fire (ParaLIF) neuron model

    Parameters:
    - input_size (int): The number of expected features in the input
    - hidden_size (int): The number of neurons on the layer
    - device (torch.device): device to use for tensor computations, such as 'cpu' or 'cuda'
    - spike_mode (str): "GS", "SB", "TRB", "D", "SD", "TD", "TRD", "T", "ST", "TT" or "TRT"
    - recurrent (bool, optional): flag to determine if the neurons should be recurrent (default: False)
    - fire (bool, optional): flag to determine if the neurons should fire spikes or not (default: True)
    - tau_mem (float, optional): time constant for the membrane potential (default: 1e-3)
    - tau_syn (float, optional): time constant for the synaptic potential (default: 1e-3)
    - time_step (float, optional): step size for updating the LIF model (default: 1e-3)
    - debug (bool, optional): flag to turn on/off debugging mode (default: False)
    """
	
    def __init__(self, input_size, hidden_size, device, spike_mode, recurrent=False, 
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, debug=False):

        super(ParaLIF, self).__init__(input_size, hidden_size, device, recurrent,
                 fire, tau_mem, tau_syn, time_step, debug)
        # Set the spiking function
        self.spike_mode = spike_mode
        if self.fire: self.spike_fn = SpikingFunction(self.device, self.spike_mode)
        else: self.spike_fn = None
        self.nb_spike_per_neuron_rec = torch.zeros(self.hidden_size, device=self.device)
        self.nb_steps = None
        

    def compute_params_fft(self):
        """
        Compute the FFT of the leakage parameters for parallel Leaky Integration

        Returns:
        fft_l_k: Product of FFT of parameters l and k
        """
        if self.nb_steps is None: return None

        l = torch.pow(self.alpha,torch.arange(self.nb_steps,device=self.device))
        k = torch.pow(self.beta,torch.arange(self.nb_steps,device=self.device))*(1-self.beta)
        fft_l = torch.fft.rfft(l, n=2*self.nb_steps).unsqueeze(1)
        fft_k = torch.fft.rfft(k, n=2*self.nb_steps).unsqueeze(1)
        return fft_l*fft_k


    def forward(self, inputs, parallel=True):
        """
        Perform forward pass of the network

        Parameters:
        - inputs (tensor): Input tensor with shape (batch_size, nb_steps, input_size)
        - parallel (bool, optional): If 'True' (default) the parallel forward is used and if 'False' the sequential forward is used

        Returns:
        - Return membrane potential tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is False
        - Return spiking tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is True
        - Return the tuple (spiking tensor, membrane potential tensor) if 'debug' is True and 'fire' is True
        """
        X = self.fc(inputs)
        if not parallel: return self.forward_sequential(X) # Run on sequential mode
        batch_size, nb_steps,_ = X.shape

        # Compute FFT params if nb_steps has changed
        if self.nb_steps!=nb_steps: 
            self.nb_steps = nb_steps
            self.fft_l_k = self.compute_params_fft()

        # Perform parallel leaky integration - Equation (15)
        fft_X = torch.fft.rfft(X, n=2*nb_steps, dim=1)
        mem_pot_hidden = torch.fft.irfft(fft_X*self.fft_l_k, n=2*nb_steps, dim=1)[:,:nb_steps:,]
        
        if self.recurrent:
            mem_pot_hidden_ = F.pad(mem_pot_hidden, (0,0,1,0), "constant", 0)[:,:-1]
            # Computing hidden state - Equation (22)
            hidden_state = self.spike_fn(mem_pot_hidden_) if self.fire else mem_pot_hidden_
            self.nb_spike_per_neuron_rec = torch.mean(torch.mean(hidden_state,dim=0),dim=0)
            # Perform parallel leaky integration for input and hidden state - Equation (23)
            fft_X_hidden_state = torch.fft.rfft(X + self.fc_recu(hidden_state), n=2*nb_steps, dim=1)
            mem_pot_temp = torch.fft.irfft(fft_X_hidden_state*self.fft_l_k, n=2*nb_steps, dim=1)[:,:nb_steps:,]
            mem_pot_final = mem_pot_hidden + mem_pot_temp
        else: mem_pot_final = mem_pot_hidden
            
        if self.fire:
        	# Perform firing - Equation (24)
            spikes = self.spike_fn(mem_pot_final)
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes,dim=0),dim=0)
            return (spikes, mem_pot_final) if self.debug else spikes
        return mem_pot_final
    
    # Sequential ParaLIF forward function
    def forward_sequential(self, X):

        batch_size, nb_steps,_ = X.shape
        syn_cur_hidden = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
        mem_pot_hidden = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
        mem_pot_hidden_prev = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
        if self.recurrent:
            syn_cur_temp = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
            mem_pot_temp = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
            hidden_state = torch.zeros_like(X) # shape: [batch_size, nb_steps, hidden_size]
        mem_pot_final = torch.zeros_like(X) # shape: [batch_size, nb_steps, hidden_size]
        spikes = torch.zeros_like(X) # shape: [batch_size, nb_steps, hidden_size]
        
        for t in range(nb_steps):
            # Integrating input to synaptic current
            syn_cur_hidden = self.alpha*syn_cur_hidden + X[:,t]
            mem_pot_hidden_prev = mem_pot_hidden
            # Integrating synaptic current to membrane potential - Equation (7)
            mem_pot_hidden = self.beta*mem_pot_hidden_prev + (1-self.beta)*syn_cur_hidden
            if self.recurrent:
                # Computing hidden state - Equation (22)
                hidden_state[:,t] = self.spike_fn(torch.stack((mem_pot_hidden_prev,mem_pot_hidden), dim=1))[:,-1]  if self.fire else mem_pot_hidden
                # Integrating input and hidden state to recurrent synaptic current
                syn_cur_temp = self.alpha*syn_cur_temp + X[:,t] + self.fc_recu(hidden_state[:,t-1])
                # Integrating recurrent synaptic current to recurrent membrane potential
                mem_pot_temp = self.beta*mem_pot_temp + (1-self.beta)*syn_cur_temp
                mem_pot_final[:,t] = mem_pot_hidden + mem_pot_temp
            else: mem_pot_final[:,t] = mem_pot_hidden
            if self.fire: spikes[:,t] = self.spike_fn(mem_pot_final[:,[t-1,t]])[:,-1]
        
        if self.fire:
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes,dim=0),dim=0)
            if self.recurrent: self.nb_spike_per_neuron_rec = torch.mean(torch.mean(hidden_state,dim=0),dim=0)
            return (spikes, mem_pot_final) if self.debug else spikes
        return mem_pot_final
    
    def extra_repr(self):
        return f"spike_mode={self.spike_mode}, recurrent={self.recurrent}, fire={self.fire}, alpha={self.alpha:.2f}, beta={self.beta:.2f}"

class ConvBase(torch.nn.Module):
    def __init__(self, input_size, kernel, stride, padding,
                device, fire, tau_mem, tau_syn, time_step):
        super(ConvBase, self).__init__()
        self.input_size = input_size
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.device = device
        self.fire = fire

        # Neuron time constants
        self.alpha = float(np.exp(-time_step/tau_syn))
        self.beta = float(np.exp(-time_step/tau_mem))

        # Convolutional Layer
        self.conv = torch.nn.Conv2d(self.input_size, self.input_size, kernel_size=self.kernel, stride=self.stride, 
                                  padding = self.padding, device=self.device)
        # Initializing weights
        torch.nn.init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_in', nonlinearity='linear')
        torch.nn.init.zeros_(self.conv.bias)

class ConvParaLIF(torch.nn.Module):
    """
    Class for implementing a Parallelizable Leaky Integrate-and-Fire (ParaLIF) neuron model

    Parameters:
    - input_size (int): The number of expected features in the input
    - hidden_size (int): The number of neurons on the layer
    - device (torch.device): device to use for tensor computations, such as 'cpu' or 'cuda'
    - spike_mode (str): "GS", "SB", "TRB", "D", "SD", "TD", "TRD", "T", "ST", "TT" or "TRT"
    - recurrent (bool, optional): flag to determine if the neurons should be recurrent (default: False)
    - fire (bool, optional): flag to determine if the neurons should fire spikes or not (default: True)
    - tau_mem (float, optional): time constant for the membrane potential (default: 1e-3)
    - tau_syn (float, optional): time constant for the synaptic potential (default: 1e-3)
    - time_step (float, optional): step size for updating the LIF model (default: 1e-3)
    - debug (bool, optional): flag to turn on/off debugging mode (default: False)
    """
	
    def __init__(self, in_channel, kernel, stride, padding, device, spike_mode, 
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3):
        super(ConvParaLIF, self).__init__()
        self.in_channel = in_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.device = device
        self.fire = fire

        # Neuron time constants
        self.alpha = float(np.exp(-time_step/tau_syn))
        self.beta = float(np.exp(-time_step/tau_mem))

        # Convolutional Layer
        self.conv = torch.nn.Conv2d(self.in_channel, self.in_channel, kernel_size=self.kernel, stride=self.stride, 
                                  padding = self.padding, device=self.device)
        # Initializing weights
        torch.nn.init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_in', nonlinearity='linear')
        torch.nn.init.zeros_(self.conv.bias)

        # Set the spiking function
        self.spike_mode = spike_mode
        self.spike_fn = SpikingFunction(self.device, self.spike_mode)
        
        self.nb_steps = None
        

    def compute_params_fft(self):
        """
        Compute the FFT of the leakage parameters for parallel Leaky Integration

        Returns:
        fft_l_k: Product of FFT of parameters l and k
        """
        if self.nb_steps is None: return None

        l = torch.pow(self.alpha,torch.arange(self.nb_steps,device=self.device))
        k = torch.pow(self.beta,torch.arange(self.nb_steps,device=self.device))*(1-self.beta)
        fft_l = torch.fft.rfft(l, n=2*self.nb_steps).unsqueeze(1)
        fft_k = torch.fft.rfft(k, n=2*self.nb_steps).unsqueeze(1)
        return fft_l*fft_k


    def forward(self, inputs, parallel=True):
        """
        Perform forward pass of the network

        Parameters:
        - inputs (tensor): Input tensor with shape (batch_size, numsteps, H, W)
        - parallel (bool, optional): If 'True' (default) the parallel forward is used and if 'False' the sequential forward is used

        Returns:
        - Return membrane potential tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is False
        - Return spiking tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is True
        - Return the tuple (spiking tensor, membrane potential tensor) if 'debug' is True and 'fire' is True
        """
        X = self.conv(inputs)
        conv_shape = X.shape
        X = X.view(X.shape[0], X.shape[1], -1)
        batch_size, nb_steps,_ = X.shape
    
        # Compute FFT params if nb_steps has changed
        if self.nb_steps!=nb_steps: 
            self.nb_steps = nb_steps
            self.fft_l_k = self.compute_params_fft()

        # Perform parallel leaky integration - Equation (15)
        fft_X = torch.fft.rfft(X, n=2*nb_steps, dim=1)
        mem_pot_final = torch.fft.irfft(fft_X*self.fft_l_k, n=2*nb_steps, dim=1)[:,:nb_steps:,]
    
        if self.fire:
        	# Perform firing - Equation (24)
            spikes = self.spike_fn(mem_pot_final)
            return mem_pot_final.view(conv_shape), spikes.view(conv_shape)
        
        return mem_pot_final.view(conv_shape)
    
    def extra_repr(self):
        return f"spike_mode={self.spike_mode}, recurrent={self.recurrent}, fire={self.fire}, alpha={self.alpha:.2f}, beta={self.beta:.2f}"