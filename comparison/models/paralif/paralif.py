# import torch
# import torch.nn.functional as F

# class SpikingFunction(torch.nn.Module):
#     """
#     Perform spike generation. There is 4 main spiking methods:
#         - GS : Gumbel Softmax
#         - SB : Sigmoid Bernouilli
#         - D: Delta
#         - T: Threshold
#     Some variants of these methods can be performed by first normalizing the input trough Sigmoid or Hyperbolic Tangeant
#     """
#     def __init__(self, device, spike_mode):
#         super(SpikingFunction, self).__init__()
            
#         if spike_mode in ["SB", "SD", "ST"]: self.normalise = torch.sigmoid
#         elif spike_mode in ["TD", "TT"]: self.normalise = torch.tanh
#         elif spike_mode in ["TRB", "TRD", "TRT"]: self.normalise = lambda inputs : F.relu(torch.tanh(inputs))
#         else: self.normalise = lambda inputs : inputs
        
#         if spike_mode in ["SB", "TRB"]: self.generate = StochasticStraightThrough.apply
#         elif spike_mode =="GS": self.generate = GumbelSoftmax(device)
#         elif spike_mode in ["D", "SD", "TD", "TRD"]: 
#             self.generate = self.delta_fn
#             self.threshold = torch.nn.Parameter(torch.tensor(0.01, device=device))
#         elif spike_mode in ["T", "ST", "TT", "TRT"]: 
#             self.generate = self.threshold_fn
#             self.threshold = torch.nn.Parameter(self.normalise(torch.tensor(1., device=device)))
            
#     def forward(self, inputs):
#         inputs = self.normalise(inputs) 
#         return self.generate(inputs)
#     # Delta Spikes Generation - Equation (19)
#     def delta_fn(self, inputs):
#         inputs_previous = F.pad(inputs, (0,0,1,0), "constant", 0)[:,:-1]
#         return SurrGradSpike.apply((inputs - inputs_previous) - self.threshold)
#     # Threshold Spikes Generation
#     def threshold_fn(self, inputs):
#         return SurrGradSpike.apply(inputs - self.threshold)

# # Sigmoid Bernoulli Spikes Generation
# class StochasticStraightThrough(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         out = torch.bernoulli(input) # Equation (17)
#         return out
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         return grad_input*input # Equation (18)


# class GumbelSoftmax(torch.nn.Module):
#     def __init__(self, device, hard=True, tau=1.0):
#         super().__init__()
        
#         self.hard = hard
#         self.tau = tau
#         self.uniform = torch.distributions.Uniform(torch.tensor(0.0).to(device),
#                                                    torch.tensor(1.0).to(device))
#         self.softmax = torch.nn.Softmax(dim=0)
  
    
#     def forward(self, logits):
#         # Sample uniform noise
#         unif = self.uniform.sample(logits.shape + (2,))
#         # Compute Gumbel noise from the uniform noise
#         gumbels = -torch.log(-torch.log(unif))
#         # Apply softmax function to the logits and Gumbel noise
#         y_soft = self.softmax(torch.stack([(logits + gumbels[...,0]) / self.tau,
#                                                      (-logits + gumbels[...,1]) / self.tau]))[0]
#         if self.hard:
#             # Use straight-through estimator
#             y_hard = torch.where(y_soft > 0.5, 1.0, 0.0)
#             ret = y_hard - y_soft.detach() + y_soft
#         else:
#             # Use reparameterization trick
#             ret = y_soft
            
#         return ret

# # Surrogate gradient implementation from https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
# class SurrGradSpike(torch.autograd.Function):
#     scale = 100.0
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         out = torch.zeros_like(input)
#         out[input > 0] = 1.0
#         return out
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
#         return grad
