'''
Produces two plots. 
Both plots are line plots for multiple models with any configuration.

Plot 1: epochs vs train accuracies
Plot 2: epochs vs test accuracies

Warning: This script can take a long time to run.
'''


import torch
import torchvision
from models import LeNet5_CIFAR, LeNet5_MNIST, SimpleSNN, SimpleParaLif, LargerSNN , GeneralParaLIF, Frankenstein, LeNet5_Flexible
from scripts import train_model, test_model
from utils import load_data, get_object_name, is_leaky
import time
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

##### Configuration #####

dataset = 'fashion'


batch_size = 256
n_epochs = 10

num_steps = 20
tau_mem = 0.02
tau_syn = 0.02
spike_mode = 'SB'


models = [
    LeNet5_MNIST(),
    SimpleSNN(input_size=28*28, num_steps=num_steps),
    GeneralParaLIF(layer_sizes=(28*28, 2**9, 2**8, 2**7, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn)
]

optimizers = [
    torch.optim.Adam,
    torch.optim.SGD,
    torch.optim.Adamax,
]

learning_rates = [
    0.01,
    0.01,
    0.001
]




##### Initialisation #####

optimizers = [opt(m.parameters(), lr) for m, opt, lr in zip(models, optimizers, learning_rates)]
devices = [device if not is_leaky(m) else torch.device('cpu') for m in models]
models = [m.to(d) for m, d in zip(models, devices)]


##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset in ['cifar', 'svhn'] else torchvision.transforms.Normalize(0, 1)
])

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size, transforms=transforms)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size, transforms=transforms)



##### Train #####

all_results = []

print('\n---------- Training ----------')
start_time = time.time()
for i, (model, optimizer, device) in enumerate(zip(models, optimizers, devices)):
    print(f'\nTraining model {i+1}: {get_object_name(model)}')
    model, results = train_model(model, 
                                loader=train_loader, 
                                optimizer=optimizer,
                                n_epochs=n_epochs, 
                                device=device,
                                val_loader=test_loader)
    models[i] = model
    all_results += [results]

print(f'\nTraining time: {time.time() - start_time:.1f} seconds.')


##### Plot #####

fig, axs = plt.subplots(1, 2)

for result in all_results:    
    axs[0].plot(result['epochs'], result['train accuracies'])
    axs[1].plot(result['epochs'], result['val accuracies'])

for i in [0, 1]:
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel('Accuracy')
    axs[i].legend([get_object_name(m, neat=True) for m in models], loc = 'lower right')
    
axs[0].set_title('Train Accuracies')
axs[1].set_title('Test Accuracies')
    
fig.suptitle(
    f'Models: {[get_object_name(m, neat=True) for m in models]}\n' +
    f'Optims: {[get_object_name(o, neat=True) for o in optimizers]}\n' +
    f'lrs: {[lr for lr in learning_rates]}\n' +
    f'Dataset: {dataset.upper()}'
)

plt.show()