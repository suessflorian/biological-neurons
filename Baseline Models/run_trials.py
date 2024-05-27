import torch
import torchvision
from models import *
import json
from scripts import train_model, test_model
from utils import load_data
import time

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu') # blank

if torch.cuda.is_available():
    torch.cuda.empty_cache()

##### Hyperparameters #####

batch_size = 256
learning_rate = 0.001 # use 0.001 for ParaLIF

### LIF/ParaLIF Hyperparameters ###

tau_mem = 0.02
tau_syn = 0.02
spike_mode = 'SB'
decay_rate = 0.

##### Options #####

train = True 
plot = False 
no_of_trials = 5 

##### Set models #####

dataset = 'mnist'
model_names = ['SimpleParaLif', 'SimpleSNN']
n_epochs_list = [5, 5]
models = [
    GeneralParaLIF(layer_sizes=(28*28, 2**9, 2**8, 2**7, 10), device=device, tau_mem=tau_mem, tau_syn=tau_syn, num_steps=20), 
    SimpleSNN(input_size=28*28, num_steps=20)]

# dataset = 'fashion'
# model_names = ['LeNet5', 'ConvAndParaFashion2', 'ConvAndParaFashion1', 'ConvAndParaFashion', 'ConvAndLifFashion1', 'ConvAndLifFashion']
# n_epochs_list = [30, 30, 30, 30, 30, 30]
# models = [LeNet5_FASHION(channels=1, para_input=5),\
#           ConvAndParaFashion2(channels=28*28, device=device, spike_mode=spike_mode, num_steps=100, tau_mem=tau_mem, tau_syn=tau_syn),\
#           ConvAndParaFashion1(channels=28*28, device=device, spike_mode=spike_mode, num_steps=100, tau_mem=tau_mem, tau_syn=tau_syn),\
#           ConvAndParaFashion(channels=28*28, device=device, spike_mode=spike_mode, num_steps=100, tau_mem=tau_mem, tau_syn=tau_syn),\
#           ConvAndLifFashion1(input_size=28*28, device=device, decay_rate=0.9, num_steps=100),\
#           ConvAndLifFashion(input_size=28*28, device=device, decay_rate=0.9, num_steps=100)]

'''dataset = 'kmnist'
model_names = ['LeNet5', 'ConvAndParaKmnist2', 'ConvAndParaKmnist1', 'ConvAndParaKmnist', 'ConvAndLifKmnist1', 'ConvAndLifKmnist']
n_epochs_list = [30, 30, 30, 50, 30, 30]
models = [LeNet5_FASHION(channels=1, para_input=5),\
          ConvAndParaKmnist2(input_size=28*28, device=device, spike_mode=spike_mode, num_steps=50, tau_mem=tau_mem, tau_syn=tau_syn),\
          ConvAndParaKmnist1(input_size=28*28, device=device, spike_mode=spike_mode, num_steps=50, tau_mem=tau_mem, tau_syn=tau_syn),\
          ConvAndParaKmnist(input_size=28*28, device=device, spike_mode=spike_mode, num_steps=50, tau_mem=tau_mem, tau_syn=tau_syn),\
          ConvAndLifKmnist1(input_size=28*28, device=device, decay_rate=0.9, num_steps=30),\
          ConvAndLifKmnist(input_size=28*28, device=device, decay_rate=0.9, num_steps=30)]'''

'''dataset = 'svhn'
model_names = ['LeNet5', 'ConvAndParaSVHN2', 'ConvAndParaSVHN1', 'ConvAndParaSVHN', 'ConvAndLifSVHN1', 'ConvAndLifSVHN']
n_epochs_list = [30, 30, 30, 30, 30, 30]
models = [LeNet5_FASHION(channels=3, para_input=6),\
          ConvAndParaSVHN2(channels=3, para_input=6, device=device, spike_mode=spike_mode, num_steps=50, tau_mem=tau_mem, tau_syn=tau_syn),\
          ConvAndParaSVHN1(channels=3, para_input=6, device=device, spike_mode=spike_mode, num_steps=50, tau_mem=tau_mem, tau_syn=tau_syn),\
          ConvAndParaSVHN(channels=3, para_input=6, device=device, spike_mode=spike_mode, num_steps=50, tau_mem=tau_mem, tau_syn=tau_syn),\
          ConvAndLifSVHN1(input_size=28*28, device=device, decay_rate=0.9, num_steps=50),\
          ConvAndLifSVHN(input_size=28*28, device=device, decay_rate=0.9, num_steps=50)]'''

##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset == 'cifar' else torchvision.transforms.Normalize(0, 1)
])

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size, transforms=transforms)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size, transforms=transforms)

print(f'\n---------- Dataset: {dataset} ----------')

##### Main Training #####
final_dict = {model_name: dict() for model_name in model_names}

for ind, model_name in enumerate(model_names):
    model = models[ind]
    n_epochs = n_epochs_list[ind]

    ##### Model #####
    model = model.to(device)
    optimizer = torch.optim.Adamax
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=decay_rate)

    ##### Training #####
    save_dict = {i:dict() for i in range(no_of_trials)}
    save_dict['epochs'] = n_epochs
    save_dict['dataset'] = dataset

    for i in range(no_of_trials):
        if train:
            start_time = time.time()
            print(f'\n---------- Training Model: {model_name}, Trial: {i} ----------\n')
            model, results = train_model(model, 
                                        loader=train_loader, 
                                        optimizer=optimizer, 
                                        n_epochs=n_epochs, 
                                        device=device)
            print(f'\nTraining time: {time.time() - start_time:.1f} seconds.')
            save_dict[i]['accuracies'] = results['train accuracies']

            if plot:
                import matplotlib.pyplot as plt
                plt.plot(results['epochs'], results['train accuracies'])
                plt.show()


        ##### Evaluation #####
        print(f'\n---------- Testing Trial: {i} ----------\n')

        train_accuracy = test_model(model, loader=train_loader, device=device)
        test_accuracy = test_model(model, loader=test_loader, device=device)
        print(f'Train Accuracy: {train_accuracy * 100:.2f}%\n' + 
            f'Test Accuracy: {test_accuracy * 100:.2f}%\n')
        
        save_dict[i]['train_accuracy'] = train_accuracy
        save_dict[i]['test_accuracy'] = test_accuracy
    
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    final_dict[model_name] = save_dict
    
    

with open(f'Baseline Models/{dataset.upper()}-trial-results.json', 'w') as fp:
    json.dump(final_dict, fp)