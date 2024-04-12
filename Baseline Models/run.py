import torch
import torchvision
from models import LeNet5_CIFAR, LeNet5_MNIST
from scripts import train_model, test_model
from utils import load_data

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

##### Options #####

dataset = 'fashion'
train = True # Set to False if model training is not required

model = LeNet5_MNIST().to(device)

load_name = None #'FASHION-LeNet5-20-epochs' # set to None if loading not required
save_name = None #'FASHION-LeNet5-50-epochs' # set to None if saving not required

##### Hyperparameters #####

batch_size = 256
learning_rate = 0.01
n_epochs = 1


##### ----- Nothing below here needs to be changed unless you're using a new dataset ----- #####



##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1)) if dataset == 'cifar' else torchvision.transforms.Normalize(0, 1)
])

train_dataset, train_loader = load_data(dataset=dataset, path='data', train=True, batch_size=batch_size, transforms=transforms)
test_dataset, test_loader = load_data(dataset=dataset, path='data', train=False, batch_size=batch_size, transforms=transforms)


##### Model #####

if load_name:
    try:
        model.load_state_dict(torch.load('Baseline Models/models/' + load_name + '.pt'))
    except:
        print('Model Not Found. Using Untrained Model.')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##### Training #####

if train:
    print('\n---------- Training ----------\n')
    model, results = train_model(model, 
                                loader=train_loader, 
                                optimizer=optimizer, 
                                n_epochs=n_epochs, 
                                device=device)


##### Evaluation #####
print('\n---------- Testing ----------\n')

train_accuracy = test_model(model, loader=train_loader, device=device)
test_accuracy = test_model(model, loader=test_loader, device=device)
print(f'Train Accuracy: {train_accuracy * 100:.2f}%\n' + 
      f'Test Accuracy: {test_accuracy * 100:.2f}%\n')


##### Saving #####

if save_name:
    torch.save(model.state_dict(), 'Baseline Models/models/' + save_name + '.pt')