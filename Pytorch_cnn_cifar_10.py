import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime
import torch.nn.functional as F

train_dataset = torchvision.datasets.CIFAR10(root='/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/', 
                                             train = True,
                                             download= True,
                                             transform= transforms.ToTensor())

test_dataset = torchvision.datasets.CIFAR10(root = '/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/',
                                             train = False,
                                             download = True,
                                             transform = transforms.ToTensor())


k = len(set(train_dataset.targets))

class CIFARCNN(nn.Module):
    def __init__(self, k):
        super(CIFARCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels= 32, kernel_size= 3, stride= 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size= 3, stride= 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size= 3, stride= 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, k)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = X.view(-1, 128 * 3 * 3)
        X = F.dropout(X, p = 0.5)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p = 0.2)
        X = self.fc2(X)
        return X

model = CIFARCNN(k)

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle = False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs = 15):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    train_accu = np.zeros(epochs)
    test_accu = np.zeros(epochs)

    for it in range(epochs):
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        n_correct = 0
        n_total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            _, predictions = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

        train_acc = n_correct / n_total
        train_loss = np.mean(train_loss)

        n_correct = 0
        n_total = 0

        for inputs, targets in test_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            _, predictions = torch.max(outputs, 1)

            test_loss.append(loss.item())
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

        test_acc = n_correct / n_total
        test_loss = np.mean(test_loss)

        print(f'Epoch {it + 1}/{epochs}, train_loss: {train_loss}, test_loss: {test_loss}, train_acc: {train_acc}, test_acc: {test_acc}')
    
    train_losses[it] = train_loss
    test_losses[it] = test_loss
    train_accu[it] = train_acc
    test_accu[it] = test_acc

    return train_losses, test_losses, train_accu, test_accu


train_losses, test_losses, train_accu, test_accu = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs = 15)

plt.plot(train_losses, label = 'training loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()

plt.plot(train_accu, label = 'training accuracy')
plt.plot(test_accu, label = 'test accuracy')
plt.legend()
plt.show()

## Confusion matrix

