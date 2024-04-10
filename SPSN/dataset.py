import torch
import torchvision
from snntorch import spikegen

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder='data', dataset='mnist', num_steps=1, train=True, transform=None):
        self.folder = folder
        self.train = train      
        self.num_steps = num_steps
        self.transform = transform or torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        if dataset == 'mnist':
            self.dataset = torchvision.datasets.MNIST(self.folder, train=self.train, transform=self.transform)
        elif dataset == 'cifar':
            self.dataset = torchvision.datasets.CIFAR10(self.folder, train=self.train, transform=self.transform)
    
    def __getitem__(self, idx):
        x, label = self.dataset[idx]
        
        image = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        image = image.view(-1)

        image = spikegen.rate(image, num_steps=self.num_steps)

        return image, label
    
    def __len__(self):
        return len(self.dataset)