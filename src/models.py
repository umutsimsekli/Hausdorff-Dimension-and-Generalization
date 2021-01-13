# Colleciton of tools which creates a random CNN with various different properties
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Configuration of VGGs with different number of layers
cfg = {
    'VGG4': [512, 'M'],
    'VGG5': [256, 'M', 512, 'M'],
    'VGG6': [256, 'M', 512, 'M', 512, 'M'],
    'VGG7': [128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

filter_nums = {
        'VGG4': 131072,
        'VGG5': 32768,
        'VGG6': 8192,
        'VGG7': 2048
        }

class VGG(nn.Module):
    def __init__(self, vgg_name, with_bias):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], with_bias)
        if vgg_name in filter_nums:
            self.classifier = nn.Linear(filter_nums[vgg_name], 10, bias=with_bias)
        else:
             self.classifier = nn.Linear(512, 10, bias=with_bias)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, with_bias):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=with_bias),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def count_parameters(self):
        param_count = 0
        for p in self.parameters():
            param_count += np.prod(list(p.shape))
        return param_count


class LeNetFamily(nn.Module):
    def __init__(self, num_convs, num_linear):
        super(LeNetFamily, self).__init__()
        self._make_layers(num_convs, num_linear)

    def _make_layers(self, num_convs, num_linear):
        if num_convs not in [0, 1,2]:
            raise ValueError("Number of convolutional layers is either 1 or 2")

        self.num_convs = num_convs
        if num_convs == 2:
            self.conv1 = nn.Conv2d(3,6,5)
            self.conv2 = nn.Conv2d(6,16,5)
            out_dim = 16*5*5

        if num_convs == 1:
            self.conv1 = nn.Conv2d(3,8,5)
            out_dim = 8*14*14
        
        if num_convs == 0:
            out_dim = 32*32*3
        
        if num_linear == 1:
            self.linears = []
            final_dim = out_dim
        else:
            self.linears = nn.ModuleList([nn.Linear(out_dim, 120)])
            if num_linear == 2:
                final_dim = 120
            else:
                self.linears.append(nn.Linear(120,84))
                final_dim = 84

        if num_linear > 2:
            for i in range(num_linear-3):
                self.linears.append(nn.Linear(84,84))
        self.final_linear = nn.Linear(final_dim,10)
            
    def forward(self, x):
        if self.num_convs > 0:
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out,2)
        if self.num_convs == 2:
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out,2)
        if self.num_convs == 0:
            out = x
        out = out.view(out.size(0), -1)
        for linear in self.linears:
            out = F.relu(linear(out))
        out = self.final_linear(out)
        return out

    def count_parameters(self):
        param_count = 0
        for p in self.parameters():
            param_count += np.prod(list(p.shape))
        return param_count


def test():
    for netname in cfg.keys():
        print(netname)
        for bias in [True, False]:
            net = VGG(netname, bias)
            x = torch.randn(2,3,32,32)
            y = net(x)
            print(bias)
            print(net.count_parameters())
            print(y.size())

#test()


class WideMLPFamily(nn.Module):
    def __init__(self, num_layers, num_parameters):
        super(WideMLPFamily, self).__init__()
        self._make_layers(num_layers, num_parameters)

    def _solve_for_l3(self, i,o,p):
        f = lambda h: i*h*h + h*h*h + o*h - p
        fp = lambda h:  2*i*h + 3*h*h + o
        h = int(np.ceil(p/(i+o)))
        itera = 0
        while f(h) > 10 or itera<100:
            h = h - (f(h))/(fp(h))
            itera+=1
        return int(np.ceil(h)), f(int(np.ceil(h)))

    def _parameter_count_to_layers(self, num_parameters, num_layers, num_input, num_output):
        if num_layers > 3 or num_layers<2:
            raise ValueError("Num Layers is either 2 or 3")
        if num_layers == 2:
            # i->h->o
            hidden_count = int(np.ceil(num_parameters/(num_input+num_output)))
            return [num_input, hidden_count, num_output]
        if num_layers == 3:
            # i->h^2->h->o
            h, err = self._solve_for_l3(num_input, num_output, num_parameters)
            upd = int(np.floor(err / (num_input + num_output + h + h*h)))
            return [num_input, h*h, h, num_output]

    def _make_layers(self, num_layers, num_parameters):
        layers = self._parameter_count_to_layers(num_parameters, num_layers, 32*32, 10)
        self.linears = nn.ModuleList([nn.Linear(layers[0], layers[1])])
        for i in range(1, len(layers)-1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        out = x.mean(dim=1)
        out = out.view(out.size(0), -1)
        for linear in self.linears:
            out = F.relu(linear(out))
        return out

    def count_parameters(self):
        param_count = 0
        for p in self.parameters():
            param_count += np.prod(list(p.shape))
        return param_count
