'''
Produces four plots. 
All plots are line plots for multiple models with any configuration.

The models in this script are already trained.

Plot 1: num_steps vs train/test accuracies
Plot 2: num_steps vs attack success rate (deepfool)
Plot 3: num_steps vs attack success rate (FGSM)
Plot 4: num_steps vs attack success rate (FMNA)

Warning: This script can take a long time to run.
'''


import torch
import torchvision
from models import LeNet5_CIFAR, LeNet5_MNIST, SimpleSNN, SimpleParaLif, LargerSNN , GeneralParaLIF, Frankenstein, LeNet5_Flexible
from scripts import test_model
from utils import load_data, get_object_name, is_leaky, make_noisy, load_model, printf
import time
import matplotlib.pyplot as plt
import foolbox as fb
from attacks import foolbox_attack

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

##### Configuration #####

dataset = 'mnist'
batch_size = 256

num_steps = 20
tau_mem = 0.02
tau_syn = 0.02
spike_mode = 'SB'

epsilon = 0.01

models = [
    LeNet5_MNIST(),
    # SimpleSNN(input_size=28*28, num_steps=num_steps),
    GeneralParaLIF(layer_sizes=(28*28, 2**9, 2**8, 2**7, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn)
]

model_filenames = [
    'MNIST-LeNet5-3-epochs.pt',
    # 'MNIST-SimpleSNN-5-epochs.pt',
    'MNIST-GeneralParaLIF-5-epochs.pt'
]

noise_steps = torch.arange(4).long()

attacks = [
    fb.attacks.LinfFastGradientAttack(),
    fb.attacks.LinfDeepFoolAttack(),
    fb.attacks.LInfFMNAttack()
]


##### Initialisation #####

devices = [device if not is_leaky(m) else torch.device('cpu') for m in models]
models = [load_model(m, 'Baseline Models/models/' + n, d) for m, n, d in zip(models, model_filenames, devices)]
print('Models Loaded Successfully')
models = [m.to(d) for m, d in zip(models, devices)]


##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset in ['cifar', 'svhn'] else torchvision.transforms.Normalize(0, 1)
])

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size, transforms=transforms)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size, transforms=transforms)

##### Attacks #####

all_results = []

print('\n---------- Running Attacks ----------', end='')
start_time = time.time()

results_successful_attacks = torch.zeros(len(attacks), len(models), len(noise_steps))
results_total_attacks = torch.zeros_like(results_successful_attacks)

for i, attack in enumerate(attacks):
    attack_name = get_object_name(attack)
    print(f'\n\nRunning attack: {attack_name}')
    for k, num_steps in enumerate(noise_steps):
        for batch, (images, labels) in enumerate(test_loader):
            if num_steps > 0:
                images = make_noisy(images, num_steps)
            for j, (model, device) in enumerate(zip(models, devices)):
                model_name = get_object_name(model)
                printf(f'Noise: {num_steps}, Batch: [{batch+1}/{len(test_loader)}], Model: {model_name:10}')
                images, labels = images.to(device), labels.to(device)
                _, _, perturbed_prediction, original_prediction = foolbox_attack(model,
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

                results_successful_attacks[i, j, k] += n_successful_attacks.sum().to('cpu')
                results_total_attacks[i, j, k] += correct_pre_attack.sum().to('cpu')

results = results_successful_attacks / results_total_attacks * 100

print(f'\n\nTime taken: {(time.time() - start_time) / 60:.1f} minutes.')

# Test evaluation

test_results = [test_model(m, test_loader, d) for m, d in zip(models, devices)]


##### Plots #####

model_names = [get_object_name(m, neat=True) for m in models]

fig, axs = plt.subplots(2, 2)
for i, attack in enumerate(attacks):
    plot_row = i // 2
    plot_col = i % 2
    results_for_attack = results[i]
    for attack_success_rates in results_for_attack:
        axs[plot_row, plot_col].plot(noise_steps, attack_success_rates)
    axs[plot_row, plot_col].set_xlabel('Noise Steps')
    axs[plot_row, plot_col].set_ylabel('Attack Success Rate')
    axs[plot_row, plot_col].legend(model_names, loc = 'lower right')
    axs[plot_row, plot_col].set_title(get_object_name(attack))

line_colours = [line.get_color() for line in axs[0,0].get_lines()]

axs[1, 1].bar(model_names, test_results, color=line_colours)
axs[1, 1].set_xlabel('Model')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].set_title('Test Accuracy')

     
fig.suptitle(f"Dataset: {dataset.upper()}\nEpsilon: {epsilon}")

plt.show()

