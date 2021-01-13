import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from random_parameter import *

from models import VGG

import torch.optim as optim


import torch.nn as nn
import click
import pickle

def train_model_vgg(vgg_name, l_rate, batch_size, device):
    transform_train = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = VGG(vgg_name, True)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=l_rate, momentum=0.9)
    model_name = "vgg_{}__bs_{}__lr_{}".format(vgg_name, batch_size, l_rate)
    path = "{}{}".format("./models/", model_name)
    
    
    EITER = 5
    for epoch in range(81):
        model_path = "{}__epoch_{}.pth".format(path, epoch)
        running_loss = 0.0
        for i,data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            last_path = "{}__last_epoch__step_{}".format(path, i)
            if i % EITER == 1:
                print("Loss @ [Epoch:{} Inter:{}] is {}".format(epoch, i, running_loss/EITER))
                running_loss = 0.0
            if epoch > 79:
                torch.save(net.state_dict(), last_path)
        if epoch % 20 == 0:
            torch.save(net.state_dict(), model_path)

@click.command()
@click.option('--lr', help="Learning Rate")
@click.option('--bs', help="Batch Size")
def train_lenet(lr, bs):
    # This code will run on GPU, set to CPU if no GPU available
    device = torch.device("cuda:0")
    
    for lay_nums in ['VGG4', 'VGG5', 'VGG6', 'VGG7', 'VGG8', 'VGG11', 'VGG13', 'VGG16', 'VGG19']: 
        print("Running with {} {} {}".format(lay_nums, float(lr), int(bs)))
        train_model_vgg(lay_nums, float(lr), int(bs), device)
    
if __name__ == '__main__':
    train_lenet()

    
    
