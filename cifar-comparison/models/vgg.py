import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_LIF(nn.Module):
    def __init__(self, vgg_name, decay_rate=0.9, window=7):
        super(VGG_LIF, self).__init__()
        self.window = window
        self.features = self._make_layers(cfg[vgg_name], decay_rate)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        mem = {l: l.init_leaky() for l in self.modules() if isinstance(l, snn.Leaky)}

        spike_train = spikegen.rate(x, num_steps=self.window)  # Convert input to spike train
        output_spikes = []
        for step in range(self.window):
            out = spike_train[step]
            for layer in self.features:
                if isinstance(layer, snn.Leaky):
                    out, mem[layer] = layer(out, mem[layer])
                else:
                    out = layer(out)

            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            output_spikes.append(out)

        return torch.stack(output_spikes, dim=0).sum(0)

    def _make_layers(self, cfg, decay_rate):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           snn.Leaky(beta=decay_rate, spike_grad=surrogate.atan())]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
