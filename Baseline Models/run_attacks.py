import torch
import torchvision
from utils import load_data, plot_attack, printf
from scripts import test_model
from models import SimpleParaLif, SimpleSNN, LeNet5_MNIST, LargerSNN, LeNet5_CIFAR
from attacks import foolbox_attack
import foolbox as fb

### GPU doesn't work for me properly - might be an easy fix though.

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')



############################## Hyperparameters ##############################

batch_size = 16

### LIF/ParaLIF Hyperparameters - Required for initialisation ###

num_steps = 20
tau_mem = 0.02
tau_syn = 0.02
decay_rate = 0.
spike_mode = 'SB'



############################## Options ##############################

use_train_data = False # Training data is used to generate attacks if True, testing data otherwise
n_successful_batches = 2 # The number of batches you want with successful attacks - set to float('Inf') to get results for the whole dataset
max_batches = 10 # The number of batches to run
epsilons = 0.01
plot = True
evaluate_model = False # set to False if you know how good this model is and only want to run attacks

### Attacks ###

# attack = fb.attacks.LinfDeepFoolAttack() # https://foolbox.readthedocs.io/en/stable/modules/attacks.html
# attack = fb.attacks.BoundaryAttack()
# attack = fb.attacks.DDNAttack()
attack = fb.attacks.LinfFastGradientAttack()

### Model Loading ###

# dataset = 'mnist'
# model_name = 'SimpleParaLIF'
# n_epochs = 5

dataset = 'mnist'
model_name = 'SimpleSNN'
n_epochs = 5


############################## Model ##############################

model = SimpleSNN(28*28, num_steps=20) # MNIST or FashionMNIST
# model = LargerSNN(3*32*32, num_steps=20) # CIFAR-10
# model = LeNet5_CIFAR()
# model = LeNet5_MNIST()
# model = SimpleParaLif(28*28, device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # MNIST
# model = testParaLIF(3*32*32, device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # CIFAR







##### ----- Nothing below here needs to be changed ----- #####


model = model.to(device)

model_name = (dataset + '-10' if dataset == 'cifar' else dataset).upper() + '-' + model_name + '-' + str(n_epochs) + '-epochs.pt'

### Loading Model ###
try:
    state_dict = torch.load('Baseline Models/models/' + model_name)
    if isinstance(model, SimpleSNN) or isinstance(model, LargerSNN):
        state_dict = {k: v for k, v in state_dict.items() if 'mem' not in k}  # Exclude memory states from loading
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
except RuntimeError:
    raise RuntimeError("Are the models you're loading and initialising the same?")
except FileNotFoundError:
    raise FileNotFoundError('Model not found. Check the directory/model name.')
else:
    print('Model loaded Successfully.\n')

############################## Data ##############################

dataset = model_name.split('-')[0].lower()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset == 'cifar' else torchvision.transforms.Normalize(0, 1)
])

dataset, loader = load_data(model_name.split('-')[0], path='data', train=use_train_data, batch_size=batch_size)

############################## Data ##############################

if evaluate_model:
    test_accuracy = test_model(model, loader=loader, device=device)
    print(f'Model\'s Accuracy: {test_accuracy * 100:.2f}%\n')

############################## Attacks ##############################


print('Running attacks...')

# We only save the successful attacks
perturbed_images = []
perturbed_predictions = []
original_predictions = []
original_images = []
original_labels = []
n_total = 0
n_total_correct = 0

batch_count = 0
for i, (images, labels) in enumerate(loader):
    
    images, labels = images.to(device), labels.to(device)
    raw_attack, perturbed_image, perturbed_prediction, original_prediction = foolbox_attack(model, 
                                                                                            images=images, 
                                                                                            labels=labels, 
                                                                                            attack=attack, 
                                                                                            epsilons=epsilons)
    if i == max_batches: 
        break
    if raw_attack is None: # no attack found
        continue
    
    correct_pre_attack = (original_prediction == labels)
    correct_post_attack = (perturbed_prediction == labels)
    
    successful_attack_indices = (correct_pre_attack & ~correct_post_attack).view(-1)
    n_successful_attacks = successful_attack_indices.sum().item()
    print(f'Batch: [{i}/{max_batches if max_batches != float('Inf') else len(loader)}], Successful Attacks: [{n_successful_attacks}/{batch_size}]')
    
    n_total_correct += correct_pre_attack.sum().item()
    
    if n_successful_attacks == 0:
        continue

    perturbed_images.append(perturbed_image[successful_attack_indices])
    perturbed_predictions.append(perturbed_prediction[successful_attack_indices])
    original_predictions.append(original_prediction[successful_attack_indices])
    original_images.append(images[successful_attack_indices])
    original_labels.append(labels[successful_attack_indices])
    
    batch_count += 1
    n_total += len(labels)
    
    if batch_count == n_successful_batches:
        break

if len(perturbed_images) == 0:
    raise ValueError('No Attacks Found.')

perturbed_images = torch.cat(perturbed_images, dim=0)
perturbed_predictions = torch.cat(perturbed_predictions, dim=0)
original_predictions = torch.cat(original_predictions, dim=0)
original_images = torch.cat(original_images, dim=0)
original_labels = torch.cat(original_labels, dim=0)

print(f"\nNumber of Successful Attacks: {perturbed_images.shape[0]}")
print(f"Number of Total Images Examined: {n_total}")
if n_total > 0:
    print(f"Attack Success Rate: {perturbed_images.shape[0] / n_total_correct*100:.2f}%\n")
else:
    print(f'No successful attacks found in {n_successful_batches} batches.\n') 

############################## Plotting ##############################

for index in range(len(original_images)):
    plot_attack(original_images, 
                perturbed_images, 
                original_labels, 
                original_predictions, 
                perturbed_predictions, 
                index=index, 
                dataset=model_name.split('-')[0].lower())

    if index < len(original_images) - 1 and input('Print More? [y, n]: ') == 'n':
        break