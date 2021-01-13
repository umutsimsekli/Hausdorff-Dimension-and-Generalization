import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from random_parameter import *

from models import LeNetFamily, VGG

import glob

import torch.nn as nn
import click
import pickle


def compute_loss_accuracy(net, loader, device):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    correct_acc = 0.0
    num = 0
    tot_loss = 0.0
    for i, data in tqdm(enumerate(loader, 0)):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).squeeze()
        correct_acc += torch.sum(acc).item()
        num += labels.size()[0]
        loss = criterion(outputs, labels)
        tot_loss += loss.item()
    return correct_acc/num, tot_loss/num


def test_on_cifar(net, device):
    transform_train = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    train_acc, train_loss = compute_loss_accuracy(net, trainloader, device)
    test_acc, test_loss = compute_loss_accuracy(net, testloader, device)
    return (train_acc, train_loss), (test_acc, test_loss)

def test_vgg(current_exp, mod_name, device):
    # Initialize the network
    # Sample random parameters from an alpha-stable rv.
    results = []

    deep_model = VGG(mod_name, True)
    deep_model = deep_model.to(device)
    deep_model.load_state_dict(torch.load(current_exp))

    r = test_on_cifar(deep_model, device)
    results.append(r)
    return results 
    
@click.command()
@click.option("--model_folder")
def test_models(model_folder):
    exps = glob.glob('{}/*'.format(model_folder))
    exps = sorted(exps)
    res = {}
    for exp in exps:
        mod_name = exp.split('/')[-1].split('__')[0].split('_')[1]
        # This code will run on GPU, set to CPU if no GPU available
        device = torch.device("cuda:0")
        res[exp] = test_vgg(current_exp, mod_name, device)
    with open("test_accuracy_{}.bn".format(model_folder), "wb") as f:
        pickle.dump(res, f)

if __name__ == '__main__':
    test_models()


