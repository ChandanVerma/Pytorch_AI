import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D

## Making the dataset
N = 1000
X = np.random.random((N, 2)) * 6 -3
Y = np.cos(2*X[:,0]) + np.cos(3*X[:,1])

print(Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

model = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr= 0.01)

def full_gd(model, criterion, optimizer, X_train, y_train, epochs = 1000):
    train_losses = np.zeros(epochs)

    for it in range(epochs):

        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        train_losses[it] = loss.item()
        if (it + 1) % 50 == 0:
            print(f'Epoch {it + 1}/{epochs}, Train loss: {loss}')

    return train_losses

X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(Y.astype(np.float32).reshape(-1, 1))

train_losses = full_gd(model=model, criterion=criterion, optimizer= optimizer, X_train = X_train, y_train = y_train, epochs= 1000)

plt.plot(train_losses)
plt.show()

with torch.no_grad():
    line = np.linspace(-3, 3, 50)
    xx, yy = np.meshgrid(line, line)
    Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
    Xgrid_torch = torch.from_numpy(Xgrid.astype(np.float32))
    Yhat = model(Xgrid_torch).numpy().flatten()
    ax.plot_trisurf(Xgrid[:,0], Xgrid[:, 1], Yhat, linewidth = 0.2, antialiased = True)
    plt.show()