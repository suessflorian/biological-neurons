Note: If this does not display well on github, download and open the file instead.

************ Standard models ************
MNIST-LeNet5-3-epochs.pt <- (98.95% test accuracy)
MNIST-LeNet5-30-epochs.pt <- (98.90% test accuracy)
FASHION-LeNet5-20-epochs.pt <- (90.32% test accuracy)
MNIST-LeNet5-3-epochs.pt <- (98.95% test accuracy) *** Baseline ***
FASHION-LeNet5-20-epochs.pt <- (90.32% test accuracy) *** Baseline ***
CIFAR-10-LeNet5-20-epochs.pt <- (64.24% test accuracy)
FASHION-LeNet5-30-epochs.pt <- (90.51% test accuracy)



************ LIF Models ************
MNIST-SimpleSNN-5-epochs.pt <- (95.36% test) [lr=0.01, optim=sgd, num_steps=20] *** Baseline ***

FASHION-SimpleSNN-5-epochs.pt <- (75.07% test) [lr=0.01, optim=sgd, num_steps=20]
FASHION-SimpleSNN-10-epochs.pt <- (81.59% test) [lr=0.01, optim=sgd, num_steps=30]
FASHION-SimpleSNN-20-epochs.pt <- (86.06% test) [lr=0.01, optim=sgd, num_steps=30]
FASHION-SimpleSNN-30-epochs.pt <- (86.66% test) [lr=0.01, optim=sgd, num_steps=30]
FASHION-SimpleSNN-50-epochs.pt <- (87.27% test) [lr=0.01, optim=sgd, num_steps=30]
FASHION-SimpleSNN-100-epochs.pt <- (88.19% test) [lr=0.01, optim=sgd, num_steps=30] *** Baseline ***

CIFAR-10-SimpleSNN-5-epochs.pt <- (22.58% test) [lr=0.01, optim=sgd, num_steps=50, decay_rate=0]
CIFAR-10-SimpleSNN-10-epochs.pt <- (27.84% test) [lr=0.01, optim=sgd, num_steps=50, decay_rate=0]
CIFAR-10-SimpleSNN-50-epochs.pt <- (29.37% test) [lr=0.01, optim=sgd, num_steps=50, decay_rate=0]

************ LIF Experiments ************
CIFAR-10-LargerSNN-5-epochs.pt <- (31.05% test) [lr=0.01, optim=sgd, num_steps=20]
CIFAR-10-LargerSNN-10-epochs.pt <- (38.12% test) [lr=0.01, optim=sgd, num_steps=20]
CIFAR-10-LargerSNN-50-epochs.pt <- (44.02% test) [lr=0.01, optim=sgd, num_steps=20]



************ ParaLIF Models ************ (These are the ones in main)
MNIST-SimpleParaLIF-5-epochs.pt <- (97.13% test) [lr=0.001, optim=adam, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB] *** Baseline ***

FASHION-SimpleParaLIF-5-epochs.pt <- (85.44% test) [lr=0.001, optim=adam, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
FASHION-SimpleParaLIF-10-epochs.pt <- (86.67% test) [lr=0.001, optim=adamax, num_steps=30, tau_mem=tau_syn=0.02, spike_mode=SB] *** Baseline ***


************ Frankenstein Models ************ (layer_sizes=(28*28, 2**9, 2**8, 2**7, 10), alphas=1, no expectation)
MNIST-Frankenstein-1-epochs.pt <- (97.92% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
MNIST-Frankenstein-2-epochs.pt <- (98.65% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
MNIST-Frankenstein-3-epochs.pt <- (98.93% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB] *** Baseline ***

FASHION-Frankenstein-5-epochs.pt <- (89.11% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB] *** Baseline ***
FASHION-Frankenstein-10-epochs.pt <- (88.67% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
FASHION-Frankenstein-20-epochs.pt <- (89.9% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]


************************************************************************************
********************************** SPECIAL USES ************************************
************************************************************************************

*** Model used in expectation_by_steps_by_model_trained.py *** 
MNIST-GeneralParaLIF-5-epochs <- (96.21% test) [lr=0.001, optim=adamax]
GeneralParaLIF(layer_sizes=(28*28, 2**9, 2**8, 2**7, 47), device=device, spike_mode='SB', num_steps=20, tau_mem=0.02, tau_syn=0.02)





*************************************************************************************
************************************ EXPERIMENTS ************************************
*************************************************************************************


***  ParaLIF Experiments *** 
FASHION-SimpleParaLIF-10-epochs.pt <- (85.63% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]
FASHION-SimpleParaLIF-20-epochs.pt <- (86.8% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]
FASHION-SimpleParaLIF-50-epochs.pt <- (87.52% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]

************ Conv + Paralif Experiments ************
MNIST-ConvAndParaMnist-30-epochs.pt <- (98.46% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]
MNIST-ConvAndParaMnist2-30-epochs.pt <- (98.86% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]
MNIST-ConvAndParaMnist1-30-epochs.pt <- (98.91% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]

Fashion-ConvAndParaFashion-30-epochs.pt <- (89.56% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]
Fashion-ConvAndParaFashion2-30-epochs.pt <- (89.96% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]
Fashion-ConvAndParaFashion1-30-epochs.pt <- (90.50% test) [lr=0.001, optim=adamax, num_steps=100, tau_mem=tau_syn=0.02, spike_mode=SB]

************ Conv + Lif Experiments ************
MNIST-ConvAndLifMnist-30-epochs.pt <- (98.95% test) [lr=0.01, optim=SGD, num_steps=30, decay_rate=0.9]
MNIST-ConvAndLifMnist1-30-epochs.pt <- (98.77% test) [lr=0.01, optim=SGD, num_steps=30, decay_rate=0.9]

Fashion-ConvAndLifFashion-30-epochs.pt <- (89.56% test) [lr=0.01, optim=adamax, num_steps=100, decay_rate=0.9]
Fashion-ConvAndLifFashion1-30-epochs.pt <- (90.80 test) [lr=0.01, optim=adamax, num_steps=100, decay_rate=0.9]

*** CIFAR where the layer_sizes = (3*32*32, 1024, 512, 256, 128, 64, 10) *** 
CIFAR-10-GeneralParaLIF01-5-epochs.pt <- (27.24% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
CIFAR-10-GeneralParaLIF01-10-epochs.pt <- (31.29% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
CIFAR-10-GeneralParaLIF01-20-epochs.pt <- (34.1% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
CIFAR-10-GeneralParaLIF01-30-epochs.pt <- (36.41% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
CIFAR-10-GeneralParaLIF01-50-epochs.pt <- (36.19% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
CIFAR-10-GeneralParaLIF01-60-epochs.pt <- (39.93% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
CIFAR-10-GeneralParaLIF01-100-epochs.pt <- (39.8% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
CIFAR-10-GeneralParaLIF01-200-epochs.pt <- (42.38% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]


*** CIFAR where the layer_sizes = (3*32*32, 4096, 256, 10) *** 
CIFAR-10-GeneralParaLIF02-5-epochs.pt <- (28.57% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
CIFAR-10-GeneralParaLIF02-10-epochs.pt <- (32.92% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
CIFAR-10-GeneralParaLIF02-10-epochs.pt <- (34.84% test) [lr=0.001, optim=adamax, num_steps=20, tau_mem=tau_syn=0.02, spike_mode=SB]
