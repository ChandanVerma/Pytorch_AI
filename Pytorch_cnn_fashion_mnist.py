import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime

train_dataset = torchvision.datasets.FashionMNIST(root= '/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/',
                                                  train= True,
                                                  download  = True,
                                                  transform= transforms.ToTensor())

test_dataset = torchvision.datasets.FashionMNIST(root= '/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/',
                                                  train= False,
                                                  download  = True,
                                                  transform= transforms.ToTensor())

k = len(set(train_dataset.targets.numpy()))

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, stride = 2, kernel_size = 3),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2),
            nn.ReLU()
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, k)
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


model = CNN(k)
model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs = 1000):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        train_losses[it] = train_loss

        test_loss = []

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)   
            test_loss.append(loss.item())

        test_loss = np.mean(test_loss)

        test_losses[it] = test_loss

        dt = datetime.now() - t0

        print(f'Epoch {it + 1}/ {epochs} , train_loss: {train_loss}, test_loss: {test_loss}, duration: {dt}')

    return train_losses, test_losses
        
train_losses, test_losses = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs = 15)


plt.plot(train_losses, label = 'training loss')
plt.plot(test_losses, label = 'test_loss')
plt.legend()
plt.show()

## Calculating accuracy
n_correct = 0
n_total = 0
for inputs, targets in train_loader:

    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)

    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

train_acc = n_correct/ n_total
print(train_acc)

n_correct = 0
n_total = 0
for inputs, targets in test_loader:
    inputs , targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)

    _, predictions = torch.max(outputs, 1)

    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

test_acc = n_correct/ n_total
print(test_acc)


