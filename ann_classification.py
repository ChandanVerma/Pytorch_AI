import torch
import torch.nn as nn 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import warnings
warnings.filterwarnings('ignore')
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm.notebook import tqdm

train_dataset = MNIST(root= '/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/', 
                      train= True,
                      transform= transforms.ToTensor(),
                      download= True)

test_dataset = MNIST(root= '/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/', 
                      train= False,
                      transform= transforms.ToTensor(),
                      download= True)

train_dataset.data.max()
train_dataset.data.shape
train_dataset.targets

test_dataset.data.max()
test_dataset.data.shape
test_dataset.targets

## Build the model
model = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

batch_size = 128

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size= batch_size,
                                            shuffle= True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size= batch_size,
                                            shuffle= False)

## testing out dataloader
tmp_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=1, shuffle= True)

for x, y in tmp_loader:
    print(x)
    print(x.shape)
    print(y.shape)
    break

## Training the model
n_epochs = 10

train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)

for it in tqdm(range(n_epochs)):
    train_loss = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs.view(-1, 784)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        train_loss.append(loss.item())

    train_loss = np.mean(train_loss)

    test_loss = []

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs.view(-1, 784)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss.append(loss.item())

    test_loss = np.mean(test_loss)

    train_losses[it] = train_loss
    test_losses[it] = test_loss

    print(f'Epoch {it+1}/{n_epochs}: Train_loss: {train_loss}, Test_loss: {test_loss}')


plt.plot(train_losses, label = 'training loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()

## Computing train accuracy

n_correct = 0
n_total = 0
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    inputs = inputs.view(-1, 784)

    outputs = model(inputs)

    _, predictions = torch.max(outputs, 1)

    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

train_acc = n_correct/n_total
print(train_acc)

## Computing test accuracy

n_correct = 0
n_total = 0

for inputs, targets in test_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    inputs = inputs.view(-1, 784)

    outputs = model(inputs)

    _, predictions = torch.max(outputs, 1)

    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

test_accu = n_correct/n_total
print(test_accu)
