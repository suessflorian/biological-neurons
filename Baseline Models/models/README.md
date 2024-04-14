The best standard models are:
MNIST-LeNet5-3-epochs.pt <- (98.95% test accuracy)
FASHION-LeNet5-20-epochs.pt <- (90.32% test accuracy)
CIFAR-10-LeNet5-20-epochs.pt <- (64.24% test accuracy)

LIF Models:
MNIST-SimpleSNN-5-epochs.pt <- (95.36% test) [lr=0.01, optim=sgd, num_steps=20]
FASHION-SimpleSNN-5-epochs.pt <- (75.07% test) [lr=0.01, optim=sgd, num_steps=20]

ParaLIF Models:
MNIST-SimpleParaLIF-5-epochs.pt <- (97.13% test) [lr=0.001, optim=adam, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
FASHION-SimpleParaLIF-5-epochs.pt <- (85.44% test) [lr=0.001, optim=adam, num_steps=20] # NOT SAVED