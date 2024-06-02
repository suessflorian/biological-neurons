import torch
import torchvision
from models import *
from utils import load_data, get_object_name, load_model
import time
import numpy as np
import pandas as pd

##### Hyperparameters #####

batch_size = 256
learning_rate = 0.01 # use 0.001 for ParaLIF (0.001 is possibly best for LIF too?)

optimizer = torch.optim.SGD # Best for SimpleSNN
# optimizer = torch.optim.Adam # NOTE: Adam doesn't seem to perform well on CIFAR with SimpleSNN
# optimizer = torch.optim.Adamax # Best for ParaLIF

device = torch.device('mps')


### LIF/ParaLIF Hyperparameters ###

tau_mem = 0.02
tau_syn = 0.02
decay_rate = 0.
spike_mode = 'SB' # ['SB', 'TRB', 'D', 'SD', 'TD', 'TRD', 'T', 'TT', 'ST', 'TRT', 'GS']


##### Options #####

dataset = 'mnist' # [mnist, cifar, fashion, emnist, kmnist, svhn]

n_iterations = 5




##### ----- Nothing below here needs to be changed unless you're using a new dataset ----- #####



##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset in ['cifar', 'svhn'] else torchvision.transforms.Normalize(0, 1)
])

_, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size, transforms=transforms)


def train_model(model, loader, optim, device, criterion):

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        
        output = model(features)
        loss = criterion(output, labels)
        
        optim.zero_grad()
        loss.backward()
        optim.step()



baseline_results = {
    'baseline':[],
    'baseline_time':[],
    'dataset':[],
    'iteration':[]
}

for iteration in range(n_iterations):
    
    
    model = LeNet5_MNIST()
    # model = LeNet5_CIFAR()

    model = model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    train_model(model, loader=train_loader, optim=opt, device=device, criterion=criterion)
    total_time = time.perf_counter() - start_time
    
    baseline_results['baseline'] += ['LeNet']
    baseline_results['dataset'] += [dataset.upper()]
    baseline_results['baseline_time'] += [total_time]
    baseline_results['iteration'] += [iteration]
    
    print(f'Baseline. I: {iteration}, time: {total_time:.1f} seconds')

results = {
    'dataset':[],
    'iteration':[],
    'model':[],
    'n_steps':[],
    'n_hidden_layers':[],
    'time':[]
    
}


seed = 0


layers = [2**9, 2**8, 2**7]
steps = [5, 10, 20, 30, 50, 100]

for iteration in range(n_iterations):
    
    for num_steps in steps:
        for i in range(len(layers)+1):
            layer = layers[:i]
            for model in range(2):
                if model == 0:
                    model = GeneralSNN2(layer_sizes=[28*28] + layer + [10], num_steps=num_steps, seed=seed)
                elif model == 1:
                    model = GeneralParaLIF(layer_sizes=[28*28] + layer + [10], device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn, seed=seed)

                model_name = get_object_name(model, neat=True)
                
                model = model.to(device)

                opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
                
                start_time = time.perf_counter()
                train_model(model, loader=train_loader, optim=opt, device=device, criterion=criterion)
                total_time = time.perf_counter() - start_time

                results['dataset'] += [dataset.upper()]
                results['iteration'] += [iteration]
                results['model'] += [model_name]
                results['n_steps'] += [num_steps]
                results['time'] += [total_time]
                results['n_hidden_layers'] += [i]


                print(f'I: {iteration}, n_steps: {num_steps}, # hidden layers: {i}, {model_name}, time: {total_time:.1f} seconds')
        
                seed += 1

baseline_results = pd.DataFrame(baseline_results)
results = pd.DataFrame(results)

full_results = results.merge(baseline_results, how='outer', on=['dataset', 'iteration'])

try:
    existing_results = pd.read_csv('Baseline Models/csvs/training_time_comp.csv')
    full_results = pd.concat([existing_results, full_results], axis=0)
except:
    pass
    
full_results.to_csv('Baseline Models/csvs/training_time_comp.csv', index=False)