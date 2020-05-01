import torch
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 

N = 1000
series = np.sin( 0.1 * np.arange(N))

plt.plot(series)
plt.show()

## Build the dataset
T = 10
X = []
Y = []

for t in range(len(series) - T):
    x = series[t: t + T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T)
Y = np.array(Y).reshape(-1, 1)
N = len(X)
X.shape, Y.shape

## building autoregressive linear model
model = nn.Linear(T, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

X_train = torch.from_numpy(X[:-N//2].astype(np.float32))
y_train = torch.from_numpy(Y[:-N//2].astype(np.float32))
X_test = torch.from_numpy(X[-N//2:].astype(np.float32))
y_test = torch.from_numpy(Y[-N//2:].astype(np.float32))

## Training the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs = 200):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):

        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
    

        train_losses[it] = loss.item()
        test_losses[it] = test_loss.item()

        if (it + 1) % 5 == 0:
            print(f'Epoch {it + 1}/{epochs}, train_loss: {loss}, test_loss: {test_loss}')

    return train_losses, test_losses


train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs = 200)

plt.plot(train_losses, label = 'training loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()

validation_target = Y[-N//2:]
validation_predictions = []

last_x = torch.from_numpy(X[-N//2].astype(np.float32))

while len(validation_predictions) < len(validation_target):
    input_ = last_x.view(1, -1)

    p = model(input_)
    validation_predictions.append(p[0, 0].item())

    last_x = torch.cat((last_x[1:], p[0]))


plt.plot(validation_target, label = 'target')
plt.plot(validation_predictions, label = 'prediction')
plt.legend()
plt.show()