import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt 

torch.cuda.get_device_name(0)

## Generate 20 data points
N = 20

X = np.random.random(N)*10 -5

Y = 0.5 * X -1 + np.random.randn(N)

#plt.scatter(X,Y)

model = nn.Linear(1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.1)

X = X.reshape(N, 1)
Y = Y.reshape(N, 1)

inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

## Train the model 
n_epochs = 50
losses = []

for it in range(n_epochs):

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    losses.append(loss.item())

    loss.backward()
    optimizer.step()

    print(f'Epoch {it+1}/ {n_epochs}, Loss: {loss}')


plt.plot(losses)
plt.show()

predicted = model(inputs).detach().numpy()
plt.scatter(X,Y, label = 'Original data')
plt.plot(X, predicted, label = 'Fitted line')
plt.legend()
plt.show()

w = model.weight.data.numpy()
b = model.bias.data.numpy()
print(w, b)