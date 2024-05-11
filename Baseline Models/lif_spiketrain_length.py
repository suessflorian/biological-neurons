import torch
import torchvision
from models import GeneralParaLIF
from utils import load_data, printf, rate
import time
import foolbox as fb
from attacks import foolbox_attack
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


##### Options #####


batch_size = 256
learning_rate = 0.001 # use 0.001 for ParaLIF

steps = [1, 2, 4, 8, 16, 32, 64, 128]  # powers of 2 so we have results on the log scale

n_epochs = 5

# optimizer = torch.optim.SGD # Best for SimpleSNN
# optimizer = torch.optim.Adam # NOTE: Adam doesn't seem to perform well on CIFAR with SimpleSNN
optimizer = torch.optim.Adamax # Best for ParaLIF

tau_mem = 0.02
tau_syn = 0.02
decay_rate = 0.
spike_mode = 'SB' # ['SB', 'TRB', 'D', 'SD', 'TD', 'TRD', 'T', 'TT', 'ST', 'TRT', 'GS']

dataset = 'fashion' # ['mnist', 'cifar', 'fashion']
attack = fb.attacks.LinfFastGradientAttack()


##### Functions #####

def add_noise(features, n_steps):
    return rate(features, num_steps=n_steps).mean(dim=0)
 
def train_model(model, loader, optimizer, device, num_steps, n_epochs=20):
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        n_correct, n_total = 0, 0
        for i, (features, labels) in enumerate(loader):
            #features = add_noise(features, num_steps)
            
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct = (output.argmax(dim = -1) == labels).sum().item()
            n_correct += correct
            n_total += len(labels)
            if (i + 1) % 20 == 0:
                printf(f'Epoch: [{epoch+1}/{n_epochs}], Batch: [{i+1}/{len(loader)}], Loss: {loss.item():.4f}, Training Accuracy: {correct/len(labels):.4f}, num_steps={num_steps}')
    return model

def test_model(model, loader, device, num_steps):
    with torch.no_grad():
        model.eval()
        n_correct, n_total = 0, 0
        
        for features, labels in loader:
            features = add_noise(features, num_steps)
            
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            output = output.argmax(dim=-1)
            n_correct += (output == labels).sum().item()
            n_total += len(labels)
    
        return n_correct / n_total

def adv_attack(model_for_attack, attack):
    total = 0
    total_successful = 0

    for i, (images, labels) in enumerate(test_loader):
        
        images, labels = images.to(device), labels.to(device)
        raw_attack, perturbed_image, perturbed_prediction, original_prediction = foolbox_attack(model_for_attack, 
                                                                                                images=images, 
                                                                                                labels=labels, 
                                                                                                attack=attack, 
                                                                                                epsilons=0.01,
                                                                                                device=device)

        
        correct_pre_attack = (original_prediction == labels)
        correct_post_attack = (perturbed_prediction == labels)
        
        total += correct_pre_attack.sum().item()
        
        if raw_attack is None: # no attack found    
            continue
        
        successful_attack_indices = (correct_pre_attack & ~correct_post_attack).view(-1)
        n_successful_attacks = successful_attack_indices.sum().item()
        printf(f'Batch: [{i}/{len(test_loader)}], ' 
            + f'Successful Attacks: [{n_successful_attacks}/{correct_pre_attack.sum().item()}],'
            + f'Batch Model Accuracy: [{correct_pre_attack.sum().item()}/{len(labels)}]')
        
        total_successful += n_successful_attacks
    return total_successful / total




##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset == 'cifar' else torchvision.transforms.Normalize(0, 1)
])

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size, transforms=transforms)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size, transforms=transforms)


##### Model #####

train_accuracy = []
test_accuracy = []
attack_susceptibility = []


for n_steps in steps:
    model = GeneralParaLIF(layer_sizes=(28*28, 1024, 768, 512, 256, 128, 10), device=device, spike_mode=spike_mode, num_steps=n_steps, tau_mem=tau_mem, tau_syn=tau_syn).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    start_time = time.time()
    model = train_model(model, loader=train_loader, optimizer=optimizer,device=device,num_steps=n_steps, n_epochs=n_epochs)
    train_accuracy.append(test_model(model, loader=train_loader, device=device, num_steps=n_steps))
    test_accuracy.append(test_model(model, loader=test_loader, device=device, num_steps=n_steps))
    attack_susceptibility.append(adv_attack(model, attack=attack))


##### Plotting #####

steps = torch.log2(torch.tensor(steps))

fig, axs = plt.subplots(1, 2)

plt.plot(results['epochs'], results['train accuracies'])
plt.plot(results['epochs'], results['val accuracies'])
plt.legend(('Train', 'Test'))
plt.show()

axs[0].plot(steps, train_accuracy, label = 'Train Accuracy')
axs[0].plot(steps, test_accuracy, label='Test Accuracy')
axs[0].legend(loc='lower right')
axs[1].plot(steps, attack_susceptibility, label='Attack Susceptibility')
axs[1].legend(loc='lower right')
plt.suptitle('Impact of Number of Steps')
# axs[0].set_xticks(torch.arange(min(steps), max(steps)))
# axs[1].set_xticks(torch.arange(min(steps), max(steps)))
# axs[0].set_yticks(torch.arange(min(attack_susceptibility) - 0.1, max(attack_susceptibility) + 0.1))
# axs[1].set_yticks(torch.arange(min(train_accuracy + test_accuracy), min(train_accuracy + test_accuracy)))
plt.show()