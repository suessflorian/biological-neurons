# Spike Trains
Unlike traditional neural networks that process information in a continuous manner, S-NNs use discrete events (spikes) which can encode information both in the rate of firing and the timing between spikes.

`snntorch` is a Python library designed to facilitate the simulation of spiking neural networks (S-NNs) using PyTorch. One of the foundational techniques used in S-NNs for input representation is rate encoding.

See [Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu “Training Spiking Neural Networks Using Lessons From Deep Learning”. Proceedings of the IEEE, 111(9) September 2023.](https://ieeexplore.ieee.org/abstract/document/10242251) for more information about this S-NN framework.

The library; [although exposes](https://snntorch.readthedocs.io/en/latest/snntorch.spikegen.html) spike generation utilities, the one we're most familiar with here is **rate encoding**.
- Given a softmax'd vector, respectively taking bernoulli expectations of each point of that vector, by running a bernoulli trial. If the trial succeeds, the corresponding point in the output vector is a 1, if not, a 0.
- We transform a single vector into `n` vectors, each being it's own bernoulli trial. The size of `n` refers to the spike train size, sometimes known for a network to be the observation window.

The end result, you have `n` vectors that represents the original input vector. The network now, iterates through those vectors and feeds them in.

## Issues
If we look at the implementation of the rate encoding within `snntorch`'s `spikegen` module - it quite [literally uses `torch.bernoulli`](https://github.com/jeshraghian/snntorch/blob/bdc1b4968a53b70f5b5a716e2f9da2a4af47495a/snntorch/spikegen.py#L437) which destroys the relationship of the computation graph which damages propogation of gradients through to the original inputs.

This is problematic because we will not be able to perform gradient based advaserial attacks on the network.

## Solution
We bring over the `rate` function from `snntorch` and specifically tackle this one line, but introducing our own "differentiable" bernoulli function.

```python
import torch

class BernoulliSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        samples = torch.bernoulli(probs)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient approximation: straight-through estimator
        # Pass the gradient through unchanged
        return grad_output
```
