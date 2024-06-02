import torch
import torchvision
from models import *
from scripts import train_model, test_model
from utils import load_data, get_object_name, load_model
import time
import numpy as np

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

##### Hyperparameters #####

batch_size = 256
learning_rate = 0.001 # use 0.001 for ParaLIF (0.001 is possibly best for LIF too?)
n_epochs = 20

# optimizer = torch.optim.SGD # Best for SimpleSNN
# optimizer = torch.optim.Adam # NOTE: Adam doesn't seem to perform well on CIFAR with SimpleSNN
optimizer = torch.optim.Adamax # Best for ParaLIF


### LIF/ParaLIF Hyperparameters ###

num_steps = 20
tau_mem = 0.02
tau_syn = 0.02
decay_rate = 0.
spike_mode = 'SB' # ['SB', 'TRB', 'D', 'SD', 'TD', 'TRD', 'T', 'TT', 'ST', 'TRT', 'GS']


##### Options #####

dataset = 'fashion' # [mnist, cifar, fashion, emnist, kmnist, svhn]


n_iterations = 5




##### ----- Nothing below here needs to be changed unless you're using a new dataset ----- #####



##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset in ['cifar', 'svhn'] else torchvision.transforms.Normalize(0, 1)
])

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size, transforms=transforms)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size, transforms=transforms)


##### Model #####

train_accuracies = []
test_accuracies = []

for iteration in range(n_iterations):
    start_time = time.time()
    # model = LeNet5_CIFAR()
    # model = GeneralSNN(layer_sizes=(3*32*32, 2**9, 2**8, 2**7, 10), num_steps=num_steps, seed=iteration)
    model = GeneralParaLIF(layer_sizes=(28*28, 2**9, 2**8, 2**7, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn, seed=iteration)

    model = model.to(device)

    opt = optimizer(model.parameters(), lr=learning_rate)

    model, results = train_model(model, 
                                loader=train_loader, 
                                optimizer=opt,
                                n_epochs=n_epochs, 
                                device=device)
    
    train_accuracies += [test_model(model, loader=train_loader, device=device)]
    test_accuracies += [test_model(model, loader=test_loader, device=device)]
    
    total_time = time.time() - start_time

    print(f'I: {iteration}, Train: {train_accuracies[-1]}, Test: {test_accuracies[-1]}, time: {total_time:.1f} seconds')

print(f"\nModel: {get_object_name(model, neat=True)}, dataset: {dataset.upper()}")
print(f'Train Mean (SD): {round(np.mean(train_accuracies), 3)} ({round(np.std(train_accuracies), 3)})')
print(f'Test Mean (SD): {round(np.mean(test_accuracies), 3)} ({round(np.std(test_accuracies), 3)})\n')