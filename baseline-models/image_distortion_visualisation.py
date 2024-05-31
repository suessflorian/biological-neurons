import torch
import matplotlib.pyplot as plt
from utils import rate, load_data
import torchvision


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0, 1)
])

train_dataset, train_loader = load_data(dataset='fashion', path='data', train=True, batch_size=16, transforms=transforms)

features, labels = next(iter(train_loader))

steps = [1, 2, 4, 8, 16, 32, 64, 128]

steps = [x.long().item() for x in torch.log2(torch.tensor(steps))]
image = features[0].permute(1,2,0)

fig, axs = plt.subplots(4, 2)
for i, n_steps in enumerate(steps):
    r, c = i // 2, i % 2
    image_with_noise = rate(image, num_steps=2**n_steps).mean(dim=0)
    axs[r, c].imshow(image_with_noise)

plt.show()
