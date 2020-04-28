import torch
import torch.nn as nn
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
from pathlib import Path

data = pd.read_csv('data/moore.csv', header = None).values

X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

plt.scatter(X, Y)
plt.show()

Y = np.log(Y)

plt.scatter(X, Y)
plt.show()

mx = X.mean()
sx = X.std()
my = Y.mean()
sy = Y.std()

X = (X- mx)/ sx
Y = (Y- my)/ sy

plt.scatter(X, Y)
plt.show()

X = X.astype(np.float32)
Y = Y.astype(np.float32)

model = nn.Linear(1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.1, momentum= 0.7)

inputs = torch.from_numpy(X)
targets = torch.from_numpy(Y)

n_epochs = 100
losses = []

for it in range(n_epochs):
    
    optimizer.zero_grad()

    outputs = model(input= inputs)
    loss = criterion(outputs, targets)

    losses.append(loss.item())

    loss.backward()
    optimizer.step()

    print(f'Epoch {it+1}/{n_epochs}, Loss: {loss}')


plt.plot(losses)
plt.show()

predicted = model(inputs).detach().numpy()
plt.scatter(X, Y, label = 'original data')
plt.plot(X, predicted, label = 'predicted line')
plt.legend()
plt.show()

print(model.weight.data.numpy())