import torch
import torchvision
from utils import load_data, plot_attack
from models import SimpleParaLif, SimpleSNN, LeNet5_MNIST, LargerSNN
from attacks import foolbox_attack
import foolbox as fb

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

use_train_data = True
n_batches_to_run = 2
attack = fb.attacks.LinfDeepFoolAttack()
epsilons = 0.1
plot = True

### Model Loading ###

# dataset = 'mnist'
# model_name = 'SimpleParaLIF'
# n_epochs = 5

dataset = 'mnist'
model_name = 'LeNet5'
n_epochs = 3


############################## Model ##############################

# model = SimpleSNN(28*28, num_steps=20) # MNIST or FashionMNIST
# model = LargerSNN(3*32*32, num_steps=20) # CIFAR-10
# model = LeNet5_CIFAR()
model = LeNet5_MNIST()
# model = SimpleParaLif(28*28, device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # MNIST
# model = testParaLIF(3*32*32, device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # CIFAR

model = model.to(device)

model_name = dataset.upper() + '-' + model_name + '-' + str(n_epochs) + '-epochs.pt'

### Loading Model ###
try:
    state_dict = torch.load('Baseline Models/models/' + model_name)
    if isinstance(model, SimpleSNN) or isinstance(model, LargerSNN):
        state_dict = {k: v for k, v in state_dict.items() if 'mem' not in k}  # Exclude memory states from loading
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
except:
    raise FileNotFoundError('Model not found.')



############################## Data ##############################

dataset = model_name.split('-')[0].lower()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset == 'cifar' else torchvision.transforms.Normalize(0, 1)
])

dataset, loader = load_data(model_name.split('-')[0], path='data', train=use_train_data, batch_size=batch_size)



############################## Attacks ##############################


# We only save the successful attacks
raw_attacks = []
perturbed_images = []
perturbed_predictions = []
original_predictions = []
original_images = []
n_total = 0

batch_count = 0
for i, (images, labels) in enumerate(loader):
    print(i)
    images, labels = images.to(device), labels.to(device)
    n_total += len(labels)
    raw_attack, perturbed_image, perturbed_prediction, original_prediction = foolbox_attack(model, images=images, labels=labels, attack=attack, epsilons=epsilons)
    print(raw_attack.shape)
    if i == 5: 
        break
    if raw_attack is None: # no attack found
        
        continue
    
    correct_pre_attack = (original_prediction == labels)
    correct_post_attack = (perturbed_prediction == labels)
    
    successful_attack_indices = (correct_pre_attack & ~correct_pre_attack).view(-1)
    print(successful_attack_indices)
    if successful_attack_indices.sum().item() == 0:
        continue
    
    raw_attacks.append(raw_attack[successful_attack_indices])
    perturbed_images.append(perturbed_image[successful_attack_indices])
    perturbed_predictions.append(perturbed_prediction[successful_attack_indices])
    original_predictions.append(original_prediction[successful_attack_indices])
    original_images.append(images[successful_attack_indices])
    
    batch_count += 1
    
    if batch_count == n_batches_to_run:
        break

print(len(raw_attacks))
raw_attacks = torch.cat(raw_attacks, dim=0)
perturbed_images = torch.cat(perturbed_images, dim=0)
perturbed_predictions = torch.cat(perturbed_predictions, dim=0)
original_predictions = torch.cat(original_predictions, dim=0)
original_images = torch.cat(original_images, dim=0)

plot_attack(original_images, raw_attacks, perturbed_images, index=0)