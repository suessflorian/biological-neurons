import torch
import torchvision

def printf(string):
    # function for printing stuff that gets removed from the output each iteration
    import sys
    from IPython.display import clear_output
    clear_output(wait = True)
    print(string, end = '')
    sys.stdout.flush()
    
def load_data(dataset = "mnist", 
              path = 'data', 
              train = True, 
              batch_size = 256, 
              transforms = torchvision.transforms.ToTensor()):
    if dataset.lower() == 'mnist':
        dataset = torchvision.datasets.MNIST('data', train=train, transform=transforms, download=True)
    elif dataset.lower() == 'fashion':
        dataset = torchvision.datasets.FashionMNIST('data', train=train, transform=transforms, download=True)
    elif dataset.lower() == 'cifar':
        dataset = torchvision.datasets.CIFAR10('data', train=train, transform=transforms, download=True)
    else:
        raise ValueError('Invalid dataset. Options: [mnist, cifar, fashion]')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader