import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from torchvision import datasets, transforms, models
import numpy as np 
import matplotlib.pyplot as plt 
import sys, os
import glob
from datetime import datetime
import imageio
from tqdm.notebook import tqdm
import shutil
from torchsummary import summary
       

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size= 256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees= 15),
    transforms.ColorJitter(),
    transforms.CenterCrop(size= 224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size = 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    '/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/Food-5K/train/',
    transform= train_transforms
)

test_dataset = datasets.ImageFolder(
    '/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/Food-5K/test/',
    transform= test_transforms
)

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle= True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle = False)

model = models.vgg16(pretrained= True)

for param in model.parameters():
    param.requires_grad = False

model.classifier

n_features = model.classifier[0].in_features
n_features

model.classifier = nn.Linear(n_features, 2)

print(model)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def full_gd(model, criterion, optimizer, train_loader, test_loader, epochs = 5):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    for it in range(epochs):
        train_loss = []
        test_loss = []
        t0 = datetime.now()
        for inputs, targets in train_loader:
            optimizer.zero_grad()

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss_ = criterion(outputs, targets)

            test_loss.append(loss_.item())

        test_loss = np.mean(test_loss)
    
    duration = datetime.now() - t0

    train_losses[it] = train_loss
    test_losses[it] = test_loss

        print(f'Epoch {it+1}/{epochs}, train_loss: {train_loss}, test_loss: {test_loss}, duration: {duration}')

    return train_losses, test_losses

train_losses, test_losses = full_gd(model, criterion, optimizer, train_loader, test_loader, epochs = 15)

plt.plot(train_losses)
plt.show()