import torch
import torchvision
import foolbox as fb
import matplotlib.pyplot as plt

def printf(string):
    # function for printing stuff that gets removed from the output each iteration
    import sys
    from IPython.display import clear_output
    clear_output(wait = True)
    print(string, end = '')
    sys.stdout.flush()
    
def load_data(dataset = "mnist", 
              path = 'data', 
              train = True, 
              batch_size = 256, 
              transforms = torchvision.transforms.ToTensor(),
              download  = True):
    if dataset.lower() == 'mnist':
        dataset = torchvision.datasets.MNIST(path, train=train, transform=transforms, download=download)
    elif dataset.lower() == 'fashion':
        dataset = torchvision.datasets.FashionMNIST(path, train=train, transform=transforms, download=download)
    elif dataset.lower() == 'cifar':
        dataset = torchvision.datasets.CIFAR10(path, train=train, transform=transforms, download=download)
    else:
        raise ValueError('Invalid dataset. Options: [mnist, cifar, fashion]')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader

def plot_attack(original_images, 
                raw_attacks, 
                perturbed_images, 
                original_labels, 
                original_predictions, 
                adversarial_predictions, 
                index, 
                categories=None):

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(original_images[index,0])
    axs[1].imshow(raw_attacks[index,0])
    axs[2].imshow(perturbed_images[index,0])
    axs[0].set_title('Original Image')
    axs[1].set_title('Raw Attack')
    axs[2].set_title('Perturbed Image')
    if categories is not None:
        fig.suptitle(f'True: {categories[original_labels[index]]},\nPredicted: {categories[original_predictions[index]]},\nAdversarial: {categories[adversarial_predictions[index]]}')
        plt.tight_layout(); fig.subplots_adjust(top=1.1)
    plt.show()

# WARNING: Implementation from: https://github.com/NECOTIS/Parallelizable-Leaky-Integrate-and-Fire-Neuron
# Sigmoid Bernoulli Spikes Generation
class StochasticStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.bernoulli(input) # Equation (17)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input*input # Equation (18)


# Gumbel Softmax Spikes Generation
class GumbelSoftmax(torch.nn.Module):
    def __init__(self, device, hard=True, tau=1.0):
        super().__init__()
        self.hard = hard
        self.tau = tau
        self.uniform = torch.distributions.Uniform(torch.tensor(0.0).to(device),torch.tensor(1.0).to(device))
        self.softmax = torch.nn.Softmax(dim=0)
  
    def forward(self, logits):
        # Sample uniform noise
        unif = self.uniform.sample(logits.shape + (2,))
        # Compute Gumbel noise from the uniform noise
        gumbels = -torch.log(-torch.log(unif))
        # Apply softmax function to the logits and Gumbel noise
        y_soft = self.softmax(torch.stack([(logits + gumbels[...,0]) / self.tau,
                                                     (-logits + gumbels[...,1]) / self.tau]))[0]
        if self.hard:
            # Use straight-through estimator
            y_hard = torch.where(y_soft > 0.5, 1.0, 0.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Use reparameterization trick
            ret = y_soft
        return ret

# Surrogate gradient implementation from https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
class SurrGradSpike(torch.autograd.Function):
    scale = 100.0
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
    
class SpikingFunction(torch.nn.Module):
    """
    Perform spike generation. There is 4 main spiking methods:
        - GS : Gumbel Softmax
        - SB : Sigmoid Bernouilli
        - D: Delta
        - T: Threshold
    Some variants of these methods can be performed by first normalizing the input trough Sigmoid or Hyperbolic Tangeant
    """
    def __init__(self, device, spike_mode):
        super(SpikingFunction, self).__init__()
            
        if spike_mode in ["SB", "SD", "ST"]: self.normalise = torch.sigmoid
        elif spike_mode in ["TD", "TT"]: self.normalise = torch.tanh
        elif spike_mode in ["TRB", "TRD", "TRT"]: self.normalise = lambda inputs : F.relu(torch.tanh(inputs))
        else: self.normalise = lambda inputs : inputs
        
        if spike_mode in ["SB", "TRB"]: self.generate = StochasticStraightThrough.apply
        elif spike_mode =="GS": self.generate = GumbelSoftmax(device)
        elif spike_mode in ["D", "SD", "TD", "TRD"]: 
            self.generate = self.delta_fn
            self.threshold = torch.nn.Parameter(torch.tensor(0.01, device=device))
        elif spike_mode in ["T", "ST", "TT", "TRT"]: 
            self.generate = self.threshold_fn
            self.threshold = torch.nn.Parameter(self.normalise(torch.tensor(1., device=device)))
            
    def forward(self, inputs):
        inputs = self.normalise(inputs) 
        return self.generate(inputs)
    # Delta Spikes Generation - Equation (19)
    def delta_fn(self, inputs):
        inputs_previous = F.pad(inputs, (0,0,1,0), "constant", 0)[:,:-1]
        return SurrGradSpike.apply((inputs - inputs_previous) - self.threshold)
    # Threshold Spikes Generation
    def threshold_fn(self, inputs):
        return SurrGradSpike.apply(inputs - self.threshold)



# WARNING: copied directly from https://github.com/jeshraghian/snntorch/blob/bdc1b4968a53b70f5b5a716e2f9da2a4af47495a/snntorch/spikegen.py#L1

dtype = torch.float

def rate(
    data,
    num_steps=False,
    gain=1,
    offset=0,
    first_spike_time=0,
    time_var_input=False,
):

    """Spike rate encoding of input data. Convert tensor into Poisson spike
    trains using the features as the mean of a
    binomial distribution. If `num_steps` is specified, then the data will be
    first repeated in the first dimension
    before rate encoding.

    If data is time-varying, tensor dimensions use time first.

    Example::

        # 100% chance of spike generation
        a = torch.Tensor([1, 1, 1, 1])
        spikegen.rate(a, num_steps=1)
        >>> tensor([1., 1., 1., 1.])

        # 0% chance of spike generation
        b = torch.Tensor([0, 0, 0, 0])
        spikegen.rate(b, num_steps=1)
        >>> tensor([0., 0., 0., 0.])

        # 50% chance of spike generation per time step
        c = torch.Tensor([0.5, 0.5, 0.5, 0.5])
        spikegen.rate(c, num_steps=1)
        >>> tensor([0., 1., 0., 1.])

        # Increasing num_steps will increase the length of
        # the first dimension (time-first)
        print(c.size())
        >>> torch.Size([1, 4])

        d = spikegen.rate(torch.Tensor([0.5, 0.5, 0.5, 0.5]), num_steps = 2)
        print(d.size())
        >>> torch.Size([2, 4])


    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :param num_steps: Number of time steps. Only specify if input data
        does not already have time dimension, defaults to ``False``
    :type num_steps: int, optional

    :param gain: Scale input features by the gain, defaults to ``1``
    :type gain: float, optional

    :param offset: Shift input features by the offset, defaults to ``0``
    :type offset: torch.optim, optional

    :param first_spike_time: Time to first spike, defaults to ``0``.
    :type first_spike_time: int, optional

    :param time_var_input: Set to ``True`` if input tensor is time-varying.
        Otherwise, `first_spike_time!=0` will modify the wrong dimension.
        Defaults to ``False``
    :type time_var_input: bool, optional

    :return: rate encoding spike train of input features of shape
        [num_steps x batch x input_size]
    :rtype: torch.Tensor

    """

    if first_spike_time < 0 or num_steps < 0:
        raise Exception(
            "``first_spike_time`` and ``num_steps`` cannot be negative."
        )

    if first_spike_time > (num_steps - 1):
        if num_steps:
            raise Exception(
                f"first_spike_time ({first_spike_time}) must be equal to "
                f"or less than num_steps-1 ({num_steps-1})."
            )
        if not time_var_input:
            raise Exception(
                "If the input data is time-varying, set "
                "``time_var_input=True``.\n If the input data is not "
                "time-varying, ensure ``num_steps > 0``."
            )

    if first_spike_time > 0 and not time_var_input and not num_steps:
        raise Exception(
            "``num_steps`` must be specified if both the input is not "
            "time-varying and ``first_spike_time`` is greater than 0."
        )

    if time_var_input and num_steps:
        raise Exception(
            "``num_steps`` should not be specified if input is "
            "time-varying, i.e., ``time_var_input=True``.\n "
            "The first dimension of the input data + ``first_spike_time`` "
            "will determine ``num_steps``."
        )

    device = data.device

    # intended for time-varying input data
    if time_var_input:
        spike_data = rate_conv(data)

        # zeros are added directly to the start of 0th (time) dimension
        if first_spike_time > 0:
            spike_data = torch.cat(
                (
                    torch.zeros(
                        tuple([first_spike_time] + list(spike_data[0].size())),
                        device=device,
                        dtype=dtype,
                    ),
                    spike_data,
                )
            )

    # intended for time-static input data
    else:

        # Generate a tuple: (num_steps, 1..., 1) where the number of 1's
        # = number of dimensions in the original data.
        # Multiply by gain and add offset.
        time_data = (
            data.repeat(
                tuple(
                    [num_steps]
                    + torch.ones(len(data.size()), dtype=int).tolist()
                )
            )
            * gain
            + offset
        )

        spike_data = rate_conv(time_data)

        # zeros are multiplied by the start of the 0th (time) dimension
        if first_spike_time > 0:
            spike_data[0:first_spike_time] = 0

    return spike_data

class Sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        samples = torch.bernoulli(probs)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient approximation: straight-through estimator
        # Pass the gradient through unchanged
        return grad_output.clone()


def rate_conv(data):
    """Convert tensor into Poisson spike trains using the features as
    the mean of a binomial distribution.
    Values outside the range of [0, 1] are clipped so they can be
    treated as probabilities.

        Example::

            # 100% chance of spike generation
            a = torch.Tensor([1, 1, 1, 1])
            spikegen.rate_conv(a)
            >>> tensor([1., 1., 1., 1.])

            # 0% chance of spike generation
            b = torch.Tensor([0, 0, 0, 0])
            spikegen.rate_conv(b)
            >>> tensor([0., 0., 0., 0.])

            # 50% chance of spike generation per time step
            c = torch.Tensor([0.5, 0.5, 0.5, 0.5])
            spikegen.rate_conv(c)
            >>> tensor([0., 1., 0., 1.])

    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :return: rate encoding spike train of input features of shape
        [num_steps x batch x input_size]
    :rtype: torch.Tensor

    """

    # Clip all features between 0 and 1 so they can be used as probabilities.
    # TODO: use torch.nn.functional.softmax instead ?!
    clipped_data = torch.clamp(data, min=0, max=1)

    # pass time_data matrix into bernoulli function.
    spike_data = Sample.apply(clipped_data)

    return spike_data
