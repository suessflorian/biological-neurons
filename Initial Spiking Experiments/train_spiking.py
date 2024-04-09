import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import autograd

from spiking_modules import spikingNeuron

device = torch.device('cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu')

num_classes = 10
num_epochs = 50
batch_size = 100
learning_rate = 0.01

input_size = 28*28  # per row
t = 4

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Create feed forward layer


class SpikingFeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.l1 = spikingNeuron(in_features, 1024, device)
        self.l2 = spikingNeuron(1024, 512, device)
        self.l3 = spikingNeuron(512, out_features, device)

    def forward(self, x, t):
        init_x = x

        for i in range(t):
            # with torch.autograd.detect_anomaly():
            x = self.l1(init_x, i)
            x = self.l2(x, i)
            x = self.l3(x, i)

        return x


# Training method
model = SpikingFeedForward(input_size, num_classes, device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [N, 1, 28*28]
        # resized: [N, 28*28]
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images, t)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (i+1) % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images, t)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
