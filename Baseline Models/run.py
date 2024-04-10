import torch
import torchvision
from models import LeNet5
from scripts import train_model, test_model

##### Options #####

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
load_name = 'CIFAR-10-LeNet5-150-epochs' # set to None if loading not required
save_name = 'CIFAR-10-LeNet5-200-epochs' # set to None if saving not required
train = True


##### Hyperparameters #####

batch_size = 256
learning_rate = 0.01
n_epochs = 50


##### Data #####

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,0,0), (1,1,1))
])

train_dataset = torchvision.datasets.CIFAR10('data', train = True, transform=transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10('data', train = False, transform=transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


##### Model #####

model = LeNet5().to(device)

if load_name:
    try:
        model.load_state_dict(torch.load('Baseline Models/models/' + load_name + '.pt'))
    except:
        print('Model Not Found. Using Untrained Model.')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##### Training #####

if train:
    model, results = train_model(model, 
                                loader=train_loader, 
                                optimizer=optimizer, 
                                n_epochs=n_epochs, 
                                device=device)


##### Evaluation #####

train_accuracy = test_model(model, loader=train_loader, device=device)
test_accuracy = test_model(model, loader=test_loader, device=device)
print(f'Train Accuracy: {train_accuracy * 100:.4f}%\n' + 
      f'Test Accuracy: {test_accuracy * 100:.4f}%')


##### Saving #####

if save_name:
    torch.save(model.state_dict(), 'Baseline Models/models/' + save_name + '.pt')