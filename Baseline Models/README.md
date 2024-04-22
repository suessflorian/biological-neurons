Note: This does not display well on github. Download and open the file instead.

************ Standard models ************
MNIST-LeNet5-3-epochs.pt <- (98.95% test accuracy)
FASHION-LeNet5-20-epochs.pt <- (90.32% test accuracy)
CIFAR-10-LeNet5-20-epochs.pt <- (64.24% test accuracy)




************ LIF Models ************
MNIST-SimpleSNN-5-epochs.pt <- (95.36% test) [lr=0.01, optim=sgd, num_steps=20]

FASHION-SimpleSNN-5-epochs.pt <- (75.07% test) [lr=0.01, optim=sgd, num_steps=20]
FASHION-SimpleSNN-10-epochs.pt <- (81.59% test) [lr=0.01, optim=sgd, num_steps=30]
FASHION-SimpleSNN-20-epochs.pt <- (86.06% test) [lr=0.01, optim=sgd, num_steps=30]
FASHION-SimpleSNN-30-epochs.pt <- (86.66% test) [lr=0.01, optim=sgd, num_steps=30]
FASHION-SimpleSNN-50-epochs.pt <- (87.27% test) [lr=0.01, optim=sgd, num_steps=30]
FASHION-SimpleSNN-100-epochs.pt <- (88.19% test) [lr=0.01, optim=sgd, num_steps=30]

CIFAR-10-SimpleSNN-5-epochs.pt <- (22.58% test) [lr=0.01, optim=sgd, num_steps=50, decay_rate=0]
CIFAR-10-SimpleSNN-10-epochs.pt <- (27.84% test) [lr=0.01, optim=sgd, num_steps=50, decay_rate=0]
CIFAR-10-SimpleSNN-50-epochs.pt <- (29.37% test) [lr=0.01, optim=sgd, num_steps=50, decay_rate=0]

************ LIF Experiments ************
CIFAR-10-LargerSNN-5-epochs.pt <- (31.05% test) [lr=0.01, optim=sgd, num_steps=20]
CIFAR-10-LargerSNN-10-epochs.pt <- (38.12% test) [lr=0.01, optim=sgd, num_steps=20]
CIFAR-10-LargerSNN-50-epochs.pt <- (44.02% test) [lr=0.01, optim=sgd, num_steps=20]



************ ParaLIF Models ************ (These are the ones in main - see Matt's branch for the ones whose results are in canva)
MNIST-SimpleParaLIF-5-epochs.pt <- (97.13% test) [lr=0.001, optim=adam, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]

FASHION-SimpleParaLIF-5-epochs.pt <- (85.44% test) [lr=0.001, optim=adam, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
FASHION-SimpleParaLIF-10-epochs.pt <- (86.67% test) [lr=0.001, optim=adamax, num_steps=30, tau_mem=tau_syn=0.02, spike_mode=SB]


************ ParaLIF Experiments ************
FASHION-SimpleParaLIF-10-epochs.pt <- (85.63% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]
FASHION-SimpleParaLIF-20-epochs.pt <- (86.8% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]
FASHION-SimpleParaLIF-50-epochs.pt <- (87.52% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]


