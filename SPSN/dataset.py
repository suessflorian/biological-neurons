import torch
import torchvision
from snntorch import spikegen

class MNISTCustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder='data', num_steps=1, train=True, transform=None):
        self.folder = folder
        self.train = train      
        self.num_steps = num_steps
        self.transform = transform or torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        self.dataset = torchvision.datasets.MNIST(self.folder, train=self.train, transform=self.transform)
    
    def __getitem__(self, idx):
        x, label = self.dataset[idx]
        
        image = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        image = image.view(-1)

        image = spikegen.rate(image, num_steps=self.num_steps)

        return image, label
    
    def __len__(self):
        return len(self.dataset)