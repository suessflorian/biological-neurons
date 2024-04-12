import torch

class sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        samples = torch.bernoulli(probs)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient approximation: straight-through estimator
        # Pass the gradient through unchanged
        return grad_output
