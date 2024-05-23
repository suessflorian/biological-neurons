import torch
import torchvision
from models import LeNet5_CIFAR, LeNet5_MNIST, SimpleSNN, SimpleParaLif, LargerSNN , GeneralParaLIF, Frankenstein, LeNet5_Flexible, GeneralSNN
from scripts import train_model, test_model
from utils import load_data, get_object_name, load_model
import time

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

##### Hyperparameters #####

batch_size = 256
learning_rate = 0.001 # use 0.001 for ParaLIF (0.001 is possibly best for LIF too?)
n_epochs = 5

# optimizer = torch.optim.SGD # Best for SimpleSNN
# optimizer = torch.optim.Adam # NOTE: Adam doesn't seem to perform well on CIFAR with SimpleSNN
optimizer = torch.optim.Adamax # Best for ParaLIF


### LIF/ParaLIF Hyperparameters ###

num_steps = 20
tau_mem = 0.02
tau_syn = 0.02
decay_rate = 0.
spike_mode = 'SB' # ['SB', 'TRB', 'D', 'SD', 'TD', 'TRD', 'T', 'TT', 'ST', 'TRT', 'GS']


##### Options #####

dataset = 'mnist' # [mnist, cifar, fashion, emnist, kmnist, svhn]
train = True # Set to False if model training is not required (i.e. you only want to evaluate a model)
plot = False

# model = SimpleSNN(28*28, num_steps=20) # MNIST or FashionMNIST
# model = LargerSNN(3*32*32, num_steps=20) # CIFAR-10
model = GeneralSNN(layer_sizes=(28*28, 10), num_steps=num_steps)
# model = LeNet5_CIFAR()
# model = LeNet5_MNIST()
# model = LeNet5_Flexible(n_classes=47) # EMNIST
# model = SimpleParaLif(28*28, device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # MNIST
# model = GeneralParaLIF(layer_sizes=(28*28, 2**9, 2**8, 2**7, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # MNIST
# model = GeneralParaLIF(layer_sizes=(28*28, 2**9, 2**8, 2**7, 47), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # EMNIST
# model = GeneralParaLIF(layer_sizes=(28*28, 1024, 768, 512, 256, 128, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # MNIST
# model = GeneralParaLIF(layer_sizes=(28*28, 5000, 64, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # MNIST
# model = GeneralParaLIF(layer_sizes=(3*32*32, 1024, 512, 256, 128, 64, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # CIFAR
# model = GeneralParaLIF(layer_sizes=(3*32*32, 6144, 512, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # CIFAR
# model = Frankenstein(layer_sizes=(28*28, 2**9, 2**8, 2**7, 10), device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn)

load_name = None #'MNIST-SimpleParaLIF-10-epochs' # set to None if loading not required
save_name = None #'MNIST-GeneralParaLIF-5-epochs' # set to None if saving not required





##### ----- Nothing below here needs to be changed unless you're using a new dataset ----- #####



##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset in ['cifar', 'svhn'] else torchvision.transforms.Normalize(0, 1)
])

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size, transforms=transforms)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size, transforms=transforms)


##### Model #####

model = model.to(device)

if load_name:
    model = load_model(model, model_name='./models/' + load_name + '.pt', device=device)
    print('Model loaded successfully.')

optimizer = optimizer(model.parameters(), lr=learning_rate)


##### Training #####

if train:
    start_time = time.time()
    print('\n---------- Training ----------\n')
    model, results = train_model(model, 
                                loader=train_loader, 
                                optimizer=optimizer,
                                n_epochs=n_epochs, 
                                device=device,
                                val_loader=test_loader)
    print(f'\nTraining time: {time.time() - start_time:.1f} seconds.')
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(results['epochs'], results['train accuracies'])
        plt.plot(results['epochs'], results['val accuracies'])
        plt.legend(('Train', 'Test'))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        model_name = get_object_name(model, neat=True)
        optimizer_name = get_object_name(optimizer)
        plt.title(f'{model_name} on {dataset.upper()}, ' +
                  f'lr={learning_rate}, optim={optimizer_name}')
        plt.show()


##### Evaluation #####
print('\n---------- Testing ----------\n')

train_accuracy = test_model(model, loader=train_loader, device=device)
test_accuracy = test_model(model, loader=test_loader, device=device)
print(f'Train Accuracy: {train_accuracy * 100:.2f}%\n' + 
      f'Test Accuracy: {test_accuracy * 100:.2f}%\n')


##### Saving #####

if save_name and train and input('Do you want to save??: ').lower() in ['y', 'yes']:
    state_dict = model.state_dict()
    if isinstance(model, SimpleSNN) or isinstance(model, LargerSNN):
        state_dict = {k: v for k, v in state_dict.items() if 'mem' not in k}
    torch.save(state_dict, './models/' + save_name + '.pt')