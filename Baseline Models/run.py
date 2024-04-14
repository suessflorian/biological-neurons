import torch
import torchvision
from models import LeNet5_CIFAR, LeNet5_MNIST, SimpleSNN, SimpleParaLif
from scripts import train_model, test_model
from utils import load_data
import time

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

##### Hyperparameters #####

batch_size = 256
learning_rate = 0.01
n_epochs = 10

# optimizer = torch.optim.SGD
# optimizer = torch.optim.Adam # NOTE: Adam doesn't seem to perform well on CIFAR with SimpleSNN
optimizer = torch.optim.Adamax # Best for ParaLIF



### LIF/ParaLIF Hyperparameters ###

num_steps = 10
tau_mem = 2e-2
tau_syn = 2e-2
decay_rate = 0.9
spike_mode = 'SB'


##### Options #####

dataset = 'mnist' # ['mnist', 'cifar', 'fashion']
train = True # Set to False if model training is not required

# model = SimpleSNN(28*28, decay_rate=decay_rate, num_steps=num_steps) # MNIST
# model = SimpleSNN(3*32*32, decay_rate=decay_rate, num_steps=num_steps) # CIFAR-10
# model = LeNet5_CIFAR()
# model = LeNet5_MNIST()
model = SimpleParaLif(28*28, device=device, spike_mode=spike_mode)

load_name = None #'MNIST-SimpleSNN-5-epochs' # set to None if loading not required
save_name = None #'MNIST-SimpleSNN-10-epochs' # set to None if saving not required




##### ----- Nothing below here needs to be changed unless you're using a new dataset ----- #####



##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset == 'cifar' else torchvision.transforms.Normalize(0, 1)
])

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size, transforms=transforms)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size, transforms=transforms)


##### Model #####

model = model.to(device)

if load_name:
    try:
        state_dict = torch.load('Baseline Models/models/' + load_name + '.pt')
        if isinstance(model, SimpleSNN):
            state_dict = {k: v for k, v in state_dict.items() if 'mem' not in k}  # Exclude memory states from loading
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        raise RuntimeError('SNNTorch Models cannot loaded properly yet.')
    except:
        print('Model Not Found. Using Untrained Model.')

optimizer = optimizer(model.parameters(), lr=learning_rate)

##### Training #####

if train:
    start_time = time.time()
    print('\n---------- Training ----------\n')
    model, results = train_model(model, 
                                loader=train_loader, 
                                optimizer=optimizer, 
                                n_epochs=n_epochs, 
                                device=device)
    print(f'\nTraining time: {time.time() - start_time:.1f} seconds.')


##### Evaluation #####
print('\n---------- Testing ----------\n')

train_accuracy = test_model(model, loader=train_loader, device=device)
test_accuracy = test_model(model, loader=test_loader, device=device)
print(f'Train Accuracy: {train_accuracy * 100:.2f}%\n' + 
      f'Test Accuracy: {test_accuracy * 100:.2f}%\n')


##### Saving #####

if save_name:
    state_dict = model.state_dict()
    if isinstance(model, SimpleSNN):
        state_dict = {k: v for k, v in state_dict.items() if 'mem' not in k}
    torch.save(state_dict, 'Baseline Models/models/' + save_name + '.pt')