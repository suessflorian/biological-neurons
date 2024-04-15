import torch
import torchvision
from models import LeNet5_CIFAR, LeNet5_MNIST, SimpleSNN, SimpleParaLif, LargerSNN #,testParaLIF
from scripts import train_model, test_model
from utils import load_data
import time

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')



##### Hyperparameters #####

batch_size = 256
learning_rate = 0.01 # use 0.001 for ParaLIF
n_epochs = 50

optimizer = torch.optim.SGD # Best for SimpleSNN
# optimizer = torch.optim.Adam # NOTE: Adam doesn't seem to perform well on CIFAR with SimpleSNN
# optimizer = torch.optim.Adamax # Best for ParaLIF



### LIF/ParaLIF Hyperparameters ###

num_steps = 20
tau_mem = 0.02
tau_syn = 0.02
decay_rate = 0.
spike_mode = 'SB'


##### Options #####

dataset = 'fashion' # ['mnist', 'cifar', 'fashion']
train = False # Set to False if model training is not required
plot = True

model = SimpleSNN(28*28, num_steps=20) # MNIST or FashionMNIST
# model = LargerSNN(3*32*32, num_steps=20) # CIFAR-10
# model = LeNet5_CIFAR()
# model = LeNet5_MNIST()
# model = SimpleParaLif(28*28, device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # MNIST
# model = testParaLIF(3*32*32, device=device, spike_mode=spike_mode, num_steps=num_steps, tau_mem=tau_mem, tau_syn=tau_syn) # CIFAR

load_name = 'FASHION-SimpleSNN-5-epochs' # set to None if loading not required
save_name = None #'CIFAR-10-LargerSNN-100-epochs' # set to None if saving not required


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
        if isinstance(model, SimpleSNN) or isinstance(model, LargerSNN):
            state_dict = {k: v for k, v in state_dict.items() if 'mem' not in k}  # Exclude memory states from loading
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
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
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(results['epochs'], results['train accuracies'])
        plt.show()


##### Evaluation #####
print('\n---------- Testing ----------\n')

train_accuracy = test_model(model, loader=train_loader, device=device)
test_accuracy = test_model(model, loader=test_loader, device=device)
print(f'Train Accuracy: {train_accuracy * 100:.2f}%\n' + 
      f'Test Accuracy: {test_accuracy * 100:.2f}%\n')


##### Saving #####

if save_name and train and input('SAVE??: ') == 'y':
    state_dict = model.state_dict()
    if isinstance(model, SimpleSNN) or isinstance(model, LargerSNN):
        state_dict = {k: v for k, v in state_dict.items() if 'mem' not in k}
    torch.save(state_dict, 'Baseline Models/models/' + save_name + '.pt')