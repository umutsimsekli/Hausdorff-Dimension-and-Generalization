import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from models import LeNetFamily

import torch.optim as optim

import torch.nn as nn
import click
import pickle

def train_model_lenet(num_convs, num_linear, l_rate, batch_size, device):
    transform_train = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = LeNetFamily(num_convs, num_linear)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=l_rate, momentum=0.9)
    
    model_name = "nc_{}__nl_{}__bs_{}__lr_{}".format(num_convs, num_linear, batch_size, l_rate)
    path = "{}{}".format("./models/", model_name)
    
    
    EITER = 500
    for epoch in range(51):
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
            if epoch > 49:
                torch.save(net.state_dict(), last_path)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), model_path)

@click.command()
@click.option('--lr', help="Learning Rate")
@click.option('--bs', help="Batch Size")
def train_lenet(lr, bs):
    # This code will run on GPU, set to CPU if no GPU available
    device = torch.device("cuda:0")
    
    for lay_nums in [(0, 8), (0, 4), (0, 2), (2, 8), (2, 4), (2, 2), (2, 1)]:
        nc = lay_nums[0]
        nl = lay_nums[1]
        print("Training with {} {} {} {}".format(nc, nl, float(lr), int(bs)))
        train_model_lenet(nc, nl, float(lr), int(bs), device)
    

if __name__ == '__main__':
    train_lenet()



    
    
