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




dataset = 'svhn' # [mnist, cifar, fashion, emnist, kmnist, svhn]
append_results_to_csv = True # setting this to False will replace the .csv


##### Hyperparameters #####

batch_size = 256
n_epochs = 20

### LIF/ParaLIF Hyperparameters ###

num_steps = 20
tau_mem = 0.02
tau_syn = 0.02
decay_rate = 0.
spike_mode = 'SB' # ['SB', 'TRB', 'D', 'SD', 'TD', 'TRD', 'T', 'TT', 'ST', 'TRT', 'GS']






##### Model Configuration #####

pretrained_model = LeNet5_Representations_Flexible(10)
pretrained_model = LeNet5_Representations_Flexible_CIFAR(10)
pretrained_load_name = 'SVHN-LeNet5-6-epochs-transfer' # set to None if loading not required

train = False # Set to False if model training is not required (i.e. you only want to evaluate a model)
load = True # this is for loading the post-representation models. The pretrained_model is always loaded and should NOT be trained.
save_models = True # set to False if saving not required



paraLIF_models = [
    # MNIST
    # GeneralParaLIF(layer_sizes=(864, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    # GeneralParaLIF(layer_sizes=(256, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    # GeneralParaLIF(layer_sizes=(120, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    # GeneralParaLIF(layer_sizes=(84, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    # GeneralParaLIF(layer_sizes=(10, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn)
    
    # CIFAR/SVHN
    GeneralParaLIF(layer_sizes=(1176, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    GeneralParaLIF(layer_sizes=(400, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    GeneralParaLIF(layer_sizes=(120, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    GeneralParaLIF(layer_sizes=(84, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    GeneralParaLIF(layer_sizes=(10, 2**9, 2**8, 2**7, 10), device=original_device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn)
]

LIF_models = [
    # MNIST
    # GeneralSNN(layer_sizes=(864, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    # GeneralSNN(layer_sizes=(256, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    # GeneralSNN(layer_sizes=(120, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    # GeneralSNN(layer_sizes=(84, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    # GeneralSNN(layer_sizes=(10, 2**9, 2**8, 2**7, 10), num_steps=num_steps)
    
    # CIFAR/SVHN
    GeneralSNN(layer_sizes=(1176, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    GeneralSNN(layer_sizes=(400, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    GeneralSNN(layer_sizes=(120, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    GeneralSNN(layer_sizes=(84, 2**9, 2**8, 2**7, 10), num_steps=num_steps),
    GeneralSNN(layer_sizes=(10, 2**9, 2**8, 2**7, 10), num_steps=num_steps)
]

paraLIF_optimizers = [torch.optim.Adamax(m.parameters(), lr=0.001) for m in paraLIF_models]
LIF_optimizers = [torch.optim.SGD(m.parameters(), lr=0.01) for m in LIF_models]







##### Attack Configuration #####

maximum_batches_to_run_attacks_on = ceil(1000 / batch_size) # can also be set manually
epsilons = [0.01, 0.05, 0.1]

attacks = [
    fb.attacks.LinfFastGradientAttack(),
    fb.attacks.LinfDeepFoolAttack(),
    fb.attacks.LInfFMNAttack()
]




##### Data #####

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size)




##### Load #####

pretrained_model = pretrained_model.to(original_device)
pretrained_model = load_model(pretrained_model, model_name=pretrained_load_name, device=original_device, path='Baseline Models/models/')

if load:
    for extraction_layer, model in enumerate(paraLIF_models):
        load_name = f'{dataset.upper()}-{get_object_name(model)}-Repr-{extraction_layer}-{n_epochs}.pt'
        paraLIF_models[extraction_layer] = load_model(model, load_name, device=original_device)
    for extraction_layer, model in enumerate(LIF_models):
        load_name = f'{dataset.upper()}-{get_object_name(model)}-Repr-{extraction_layer}-{n_epochs}.pt'
        LIF_models[extraction_layer] = load_model(model, load_name, device=torch.device('cpu'))
    print('Models loaded successfully.')
    if train and input('\nYou have loaded pre-trained LIF/ParaLIF models. Do you want to train further? [y/n]: ') != 'y':
        train = False

pretrained_model.eval()



##### Train #####

if train:
    start_time = time.time()
    print('\n---------- Training ----------\n')
    for i, (model, optimizer) in enumerate(zip(paraLIF_models + LIF_models, 
                                               paraLIF_optimizers + LIF_optimizers)):
        extraction_layer = i % len(paraLIF_models)
        
        if is_leaky(model): # this actually tests if the model has LIF neurons rather than if it's leaky.
            device = torch.device('cpu')
        else:
            device = original_device
        
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
                    f'Epoch: [{epoch+1}/{n_epochs}], ' +
                    f'Iteration: [{j+1}/{len(train_loader)}], ' +
                    f'Training Accuracy: {train_accuracy:.4f}')
        if save_models:
            save_name = f'{dataset.upper()}-{get_object_name(model)}-Repr-{extraction_layer}-{n_epochs}.pt'
            state_dict = model.state_dict()
            if is_leaky(model):
                state_dict = {k: v for k, v in state_dict.items() if 'mem' not in k}
            torch.save(state_dict, 'Baseline Models/models/' + save_name)

    print(f'\nTraining time: {time.time() - start_time:.1f} seconds.')



##### Evaluation #####

print('\n---------- Evaluating Accuracy ----------\n')

training_accuracies = []
test_accuracies = []

with torch.no_grad():
    model.eval()
    
    # Train accuracy
    for i, model in enumerate(paraLIF_models + LIF_models):
        extraction_layer = i % len(paraLIF_models)
        if is_leaky(model):
            device = torch.device('cpu')
        else:
            device = original_device
        
        model.eval()
        
        # Train
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(original_device), labels.to(device)
            representations = pretrained_model(images, extraction_layer=extraction_layer).to(device)
            representations = (representations - representations.min()) / (representations.max() - representations.min())
            correct += (model(representations).argmax(dim=-1) == labels).float().sum().item()
            total += len(labels)
        training_accuracies.append(correct / total)

        if i + 1 == len(paraLIF_models):
            print('\n... Halfway there ...\n')
        
        # Test
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(original_device), labels.to(device)
            representations = pretrained_model(images, extraction_layer=extraction_layer).to(device)
            representations = (representations - representations.min()) / (representations.max() - representations.min())
            correct += (model(representations).argmax(dim=-1) == labels).sum().item()
            total += len(labels)
        test_accuracies.append(correct / total)

pretrained_train_accuracy = test_model(pretrained_model, train_loader, original_device)
pretrained_test_accuracy = test_model(pretrained_model, test_loader, original_device)



##### Baseline Attack Susceptibility #####

print('\n---------- Evaluating Baseline Attack Susceptibility ----------\n')

baseline_results = {
    'attack':[],
    'epsilon':[],
    'baseline_model': [],
    'baseline_train_accuracy': [],
    'baseline_test_accuracy': [],
    'baseline_susceptibility_rate': [],
    'dataset':[]
}

baseline_name = get_object_name(pretrained_model)

for attack in attacks:
    attack_name = get_object_name(attack)
    for epsilon in epsilons:
        total_successful_attacks, total_successful_classifications = 0, 0
        for batch, (images, labels) in enumerate(test_loader):
            printf(f'Attack: {attack_name} on model: {get_object_name(pretrained_model)}, epsilon: {epsilon}, batch: [{batch}/{maximum_batches_to_run_attacks_on}]')
            images, labels = images.to(original_device), labels.to(original_device)
            
            _, _, perturbed_prediction, original_prediction = foolbox_attack(pretrained_model,
                                                                            images,
                                                                            labels,
                                                                            attack,
                                                                            epsilon,
                                                                            original_device)
            perturbed_prediction = perturbed_prediction.to(original_device)
            original_prediction = original_prediction.to(original_device)
            correct_pre_attack = (original_prediction == labels)
            correct_post_attack = (perturbed_prediction == labels)
            n_successful_attacks = (correct_pre_attack & ~correct_post_attack)
            
            total_successful_attacks += n_successful_attacks.sum().to('cpu').item()
            total_successful_classifications += correct_pre_attack.sum().to('cpu').item()
            
            if batch == maximum_batches_to_run_attacks_on:
                break
        
        baseline_results['attack'] += [attack_name]
        baseline_results['epsilon'] += [epsilon]
        baseline_results['baseline_model'] += [baseline_name]
        baseline_results['baseline_train_accuracy'] += [pretrained_train_accuracy]
        baseline_results['baseline_test_accuracy'] += [pretrained_test_accuracy]
        baseline_results['baseline_susceptibility_rate'] += [total_successful_attacks / total_successful_classifications]
        baseline_results['dataset'] += [dataset.upper()]



##### Attack Susceptibility #####

print('\n\n---------- Evaluating Attack Susceptibility ----------\n')

results = {
    'model':[],
    'epochs_trained':[],
    'extraction_layer':[],
    'attack':[],
    'epsilon':[],
    'susceptibility_rate':[],
    'train_accuracy':[],
    'test_accuracy':[]
}


for i, model in enumerate(paraLIF_models + LIF_models):
    extraction_layer = i % len(paraLIF_models) # bit of a hacky way I know :(
    model_name = get_object_name(model)
    
    if is_leaky(model):
        device = torch.device('cpu')
    else:
        device = original_device
    
    for attack in attacks:
        attack_name = get_object_name(attack)
        for epsilon in epsilons:
            total_successful_attacks, total_successful_classifications = 0, 0
            for batch, (images, labels) in enumerate(test_loader):
                printf(f'Attack: {attack_name} on model [{i+1}/{len(paraLIF_models) + len(LIF_models)}]: {model_name}, epsilon: {epsilon}, batch: [{batch}/{maximum_batches_to_run_attacks_on}]')
                images, labels = images.to(original_device), labels.to(device)
                with torch.no_grad():
                    pretrained_model.eval()
                    representations = pretrained_model(images, extraction_layer).to(device)
                    representations = (representations - representations.min()) / (representations.max() - representations.min())
                
                _, _, perturbed_prediction, original_prediction = foolbox_attack(model,
                                                                                representations,
                                                                                labels,
                                                                                attack,
                                                                                epsilon,
                                                                                device)
                perturbed_prediction = perturbed_prediction.to(device)
                original_prediction = original_prediction.to(device)
                correct_pre_attack = (original_prediction == labels)
                correct_post_attack = (perturbed_prediction == labels)
                n_successful_attacks = (correct_pre_attack & ~correct_post_attack)
                
                total_successful_attacks += n_successful_attacks.sum().to('cpu').item()
                total_successful_classifications += correct_pre_attack.sum().to('cpu').item()
                
                if batch == maximum_batches_to_run_attacks_on:
                    break
                
            results['model'] += [model_name]
            results['epochs_trained'] += [n_epochs]
            results['extraction_layer'] += [extraction_layer]
            results['attack'] += [attack_name]
            results['epsilon'] += [epsilon]
            results['susceptibility_rate'] += [total_successful_attacks / total_successful_classifications]
            results['train_accuracy'] += [training_accuracies[i]]
            results['test_accuracy'] += [test_accuracies[i]]





##### Save #####

results = pd.DataFrame(results)
baseline_results = pd.DataFrame(baseline_results)
new = results.merge(baseline_results, how='outer', on=['attack', 'epsilon'])


path = 'Baseline Models/csvs/' + 'transfer.csv'

if not append_results_to_csv:
    while (ans := input('Are you sure you want to OVERWRITE the existing csv? y/n: ')) not in 'yn':
        continue
    if ans == 'n':
        append_results_to_csv = False


if append_results_to_csv:
    try:
        existing = pd.read_csv(path)
        final = pd.concat((existing, new))
        final.to_csv(path, index=False)
    except:
        print('No existing CSV found!\nCreating a new CSV.')
        new.to_csv(path, index=False)
else:
    new.to_csv(path, index=False)
    
print('\n\n---------- CSV SAVED ----------\n')