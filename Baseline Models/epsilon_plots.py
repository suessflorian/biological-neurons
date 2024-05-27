
import torch
import torch.nn.functional as F
from models import GeneralParaLIF, GeneralSNN, LeNet5_MNIST
from scripts import test_model
from utils import load_data, get_object_name, load_model, is_leaky, printf
import time
import foolbox as fb
from math import ceil
from attacks import foolbox_attack, art_attack
import art
import pandas as pd
import copy


device = torch.device('cpu')
# device = torch.device('mps')


dataset = 'mnist'

epsilons = [0.01, 0.05, 0.1, 0.25, 0.5, 1]

##### Hyperparameters #####

batch_size = 256
n_epochs = 20

maximum_batches_to_run_attacks_on = ceil(1000 / batch_size)

### LIF/ParaLIF Hyperparameters ###

num_steps = 20
tau_mem = 0.02
tau_syn = 0.02
spike_mode = 'SB' # ['SB', 'TRB', 'D', 'SD', 'TD', 'TRD', 'T', 'TT', 'ST', 'TRT', 'GS']



model_names = [
    'MNIST-LeNet5-3-epochs',
    'MNIST-GeneralParaLIF-5-epochs',
    'MNIST-GeneralSNN-5-epochs'
]

models = [
    LeNet5_MNIST(),
    GeneralParaLIF(layer_sizes=(28*28, 2**9, 2**8, 2**7, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn),
    GeneralSNN(layer_sizes=(28*28, 2**9, 2**8, 2**7, 10), num_steps=num_steps)
]


attacks = [
    fb.attacks.LinfFastGradientAttack(),
    fb.attacks.LinfDeepFoolAttack(),
    fb.attacks.LInfFMNAttack(),
    art.attacks.evasion.SquareAttack
]

attack_functions = [
    foolbox_attack,
    foolbox_attack,
    foolbox_attack,
    art_attack
]


_, train_loader = load_data(dataset, train=True, batch_size=batch_size)
_, test_loader = load_data(dataset, train=False, batch_size=batch_size)


results = {
    'model':[],
    'attack':[],
    'epsilon':[],
    'susceptibility_rate':[],
    'dataset':[]
}

for attack, attack_function in zip(attacks, attack_functions):
    attack_name = get_object_name(attack, neat=True)
    
    for epsilon in epsilons:
        for i, (name, model) in enumerate(zip(model_names, models)):
            model_name = get_object_name(model, neat=True)
            total_successful_attacks = 0
            total_successful_classifications = 0

            for batch, (images, labels) in enumerate(test_loader):
                printf(f'Attack: {attack_name}, epsilon: {epsilon} on {model_name}, batch: [{batch}/{maximum_batches_to_run_attacks_on}]')
                images, labels = images.to(device), labels.to(device)
                
                    
                _, _, perturbed_prediction, original_prediction = attack_function(model,
                                                                                images,
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
            results['attack'] += [attack_name]
            results['epsilon'] += [epsilon]
            results['susceptibility_rate'] += [total_successful_attacks / total_successful_classifications]
            results['dataset'] += [dataset]
            
results = pd.DataFrame(results)
results.to_csv('Baseline Models/csvs/epsilon_plots.csv', index=False)
            
            