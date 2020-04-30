import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime
import torch.nn.functional as F

train_transforms = torchvision.transforms.Compose([
    transforms.RandomCrop(32, padding =4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(0, translate= (0.1, 0.1)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(root='/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/', 
                                             train = True,
                                             download= True,
                                             transform= train_transforms)

test_dataset = torchvision.datasets.CIFAR10(root = '/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/',
                                             train = False,
                                             download = True,
                                             transform = transforms.ToTensor())


k = len(set(train_dataset.targets))

class CIFARCNN(nn.Module):
    def __init__(self, k):
        super(CIFARCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels= 32, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size= 3, stride = 2, padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size= 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size= 3, stride = 2, padding =1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, k)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = X.view(X.size(0), -1)
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

def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs = 80):

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


train_losses, test_losses, train_accu, test_accu = batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs = 80)

plt.plot(train_losses, label = 'training loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()

plt.plot(train_accu, label = 'training accuracy')
plt.plot(test_accu, label = 'test accuracy')
plt.legend()
plt.show()

## Confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize = False, title = 'confusion matrix', cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation= 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment = 'center',
        color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()

from torchsummary import summary

summary(model, (3, 32, 32))



