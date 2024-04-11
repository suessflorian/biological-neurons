import torch.nn.functional as F
import torch.nn as nn

class LeNet5(nn.Module):
    # CIFAR-10
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor for the fully connected layer
        x = self.bn3(self.fc1(x))
        x = F.relu(x)
        x = self.bn4(self.fc2(x))
        x = F.relu(x)
        x = self.fc3(x)
        return x

