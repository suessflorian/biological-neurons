'''
Creates a .csv called "transfer.csv" with results on training LIF and paraLIF models on LeNet representations.
'''


import torch
import torch.nn.functional as F
from models import GeneralParaLIF, GeneralSNN, LeNet5_Representations_Flexible, LeNet5_Representations_Flexible_CIFAR
from scripts import test_model
from utils import load_data, get_object_name, load_model, is_leaky, printf
import time
import foolbox as fb
from math import ceil
from attacks import foolbox_attack
import pandas as pd

original_device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')




dataset = 'mnist' # [mnist, cifar, fashion, emnist, kmnist, svhn]
append_results_to_csv = True # setting this to False will replace the .csv


##### Hyperparameters #####

batch_size = 256
n_epochs = 5
n_trials = 5

### LIF/ParaLIF Hyperparameters ###

num_steps = 20
tau_mem = 0.02
tau_syn = 0.02
spike_mode = 'SB' # ['SB', 'TRB', 'D', 'SD', 'TD', 'TRD', 'T', 'TT', 'ST', 'TRT', 'GS']






##### Model Configuration #####

pretrained_model = LeNet5_Representations_Flexible(10) #MNIST/FASHION
# pretrained_model = LeNet5_Representations_Flexible_CIFAR(10) # SVHN


# ONLY USE TRANSFER MODELS HERE - THEY ARE SPECIFICALLY TRAINED ON [0, 1] DATA

pretrained_load_name = 'MNIST-LeNet5-3-epochs-transfer'
# pretrained_load_name = 'FASHION-LeNet5-20-epochs-transfer'
# pretrained_load_name = 'SVHN-LeNet5-6-epochs-transfer'


paraLIF_models = [
    # MNIST/FASHION
    GeneralParaLIF(layer_sizes=(864, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    GeneralParaLIF(layer_sizes=(256, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    GeneralParaLIF(layer_sizes=(120, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    GeneralParaLIF(layer_sizes=(84, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    GeneralParaLIF(layer_sizes=(10, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn)
    
    # # CIFAR/SVHN
    # GeneralParaLIF(layer_sizes=(1176, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    # GeneralParaLIF(layer_sizes=(400, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    # GeneralParaLIF(layer_sizes=(120, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    # GeneralParaLIF(layer_sizes=(84, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    # GeneralParaLIF(layer_sizes=(10, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn)
]

LIF_models = [
    # MNIST/FASHION
    GeneralSNN(layer_sizes=(864, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    GeneralSNN(layer_sizes=(256, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    GeneralSNN(layer_sizes=(120, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    GeneralSNN(layer_sizes=(84, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    GeneralSNN(layer_sizes=(10, 2**9, 2**8, 2**7, 10), num_steps=num_steps)
    
    # # CIFAR/SVHN
    # GeneralSNN(layer_sizes=(1176, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    # GeneralSNN(layer_sizes=(400, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    # GeneralSNN(layer_sizes=(120, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    # GeneralSNN(layer_sizes=(84, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    # GeneralSNN(layer_sizes=(10, 2**9, 2**8, 2**7, 10), num_steps=num_steps)
]

paraLIF_optimizers = [torch.optim.Adamax(m.parameters(), lr=0.001) for m in paraLIF_models]
LIF_optimizers = [torch.optim.SGD(m.parameters(), lr=0.01) for m in LIF_models]












##### Data #####

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size)




##### Load #####

pretrained_model = pretrained_model.to(original_device)
pretrained_model = load_model(pretrained_model, model_name=pretrained_load_name, device=original_device, path='Baseline Models/models/')

pretrained_model.eval()



##### Train #####

results = {
    'dataset':[],
    'model':[],
    'pretrained_model':[],
    'extraction_layer':[],
    'trial':[],
    'train_accuracy':[],
    'test_accuracy':[]
}

start_time = time.time()
print('\n---------- Training ----------\n')

for i, (model, optimizer) in enumerate(zip(paraLIF_models + LIF_models, 
                                            paraLIF_optimizers + LIF_optimizers)):
    extraction_layer = i % len(paraLIF_models)
    
    if is_leaky(model): # this actually tests if the model has LIF neurons rather than if it's leaky.
        device = torch.device('cpu')
    else:
        device = original_device
    
    for trial in range(n_trials):
        model.train()
        for epoch in range(n_epochs):
            for j, (images, labels) in enumerate(train_loader):
                images, labels = images.to(original_device), labels.to(device)
                
                with torch.no_grad():
                    representations = pretrained_model(images, extraction_layer=extraction_layer).to(device)
                    representations = (representations - representations.min()) / (representations.max() - representations.min())
                
                output = model(representations)
                loss = F.cross_entropy(output, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_accuracy = (output.argmax(dim=-1) == labels).float().mean().item()
                
                printf(f'Model [{i+1}/{len(paraLIF_models) + len(LIF_models)}]: {get_object_name(model)}, ' +
                       f'Extraction Layer: [{extraction_layer}/{len(paraLIF_models)-1}], ' +
                       f'Trial: {trial}, ' +
                       f'Epoch: [{epoch+1}/{n_epochs}], ' +
                       f'Iteration: [{j+1}/{len(train_loader)}], ' +
                       f'Training Accuracy: {train_accuracy:.4f}')

        
        results['model'].append(get_object_name(model, neat=True))
        results['dataset'].append(dataset.upper())
        results['trial'].append(trial)
        results['pretrained_model'].append(get_object_name(pretrained_model, neat=True))
        results['extraction_layer'].append(extraction_layer)
        
        with torch.no_grad():
            model.eval()
            # Train
            correct, total = 0, 0
            for images, labels in train_loader:
                images, labels = images.to(original_device), labels.to(device)
                representations = pretrained_model(images, extraction_layer=extraction_layer).to(device)
                representations = (representations - representations.min()) / (representations.max() - representations.min())
                correct += (model(representations).argmax(dim=-1) == labels).float().sum().item()
                total += len(labels)
            results['train_accuracy'].append(correct / total)
        
            correct, total = 0, 0
            for images, labels in test_loader:
                images, labels = images.to(original_device), labels.to(device)
                representations = pretrained_model(images, extraction_layer=extraction_layer).to(device)
                representations = (representations - representations.min()) / (representations.max() - representations.min())
                correct += (model(representations).argmax(dim=-1) == labels).sum().item()
                total += len(labels)
            results['test_accuracy'].append(correct / total)
        
        
        

print(f'\nTraining time: {time.time() - start_time:.1f} seconds.')



##### Save #####

results = pd.DataFrame(results)


path = 'Baseline Models/csvs/' + 'transfer_train_test_sd.csv'

if not append_results_to_csv:
    while (ans := input('Are you sure you want to OVERWRITE the existing csv? y/n: ')) not in 'yn':
        continue
    if ans == 'n':
        append_results_to_csv = False


if append_results_to_csv:
    try:
        existing = pd.read_csv(path)
        final = pd.concat((existing, results))
        final.to_csv(path, index=False)
    except:
        print('No existing CSV found!\nCreating a new CSV.')
        results.to_csv(path, index=False)
else:
    results.to_csv(path, index=False)
    
print('\n\n---------- CSV SAVED ----------\n')