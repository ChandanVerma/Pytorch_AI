import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
import numpy as np 
import matplotlib.pyplot as plt 
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
data.keys()
X, Y = data.data, data.target

X.shape
Y.shape

data.target_names
data.feature_names

## Performing train test split
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size = 0.2, random_state = 123)
N, D = X_train.shape

## Scaling the dataset
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

## Building the model
model = nn.Sequential(
                nn.Linear(D, 1),
                nn.Sigmoid()
                    )

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32).reshape(-1, 1))
Y_test = torch.from_numpy(Y_test.astype(np.float32).reshape(-1, 1))

n_epochs = 1000
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)
train_accu = np.zeros(n_epochs)
test_accu = np.zeros(n_epochs)

for it in range(n_epochs):
    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, Y_train)
    loss.backward()

    optimizer.step()

    output_test = model(X_test)
    loss_test = criterion(output_test, Y_test)

    train_accu[it] = np.mean(np.round(outputs.detach().numpy()) == Y_train.detach().numpy())
    test_accu[it] = np.mean(np.round(output_test.detach().numpy()) == Y_test.detach().numpy())

    train_losses[it] = loss.item()
    test_losses[it] = loss_test.item()

    if (it+1) % 50 == 0:
        print(f'Epoch: {it+1}/{n_epochs}, train_loss:{loss.item()}, test_loss: {loss_test.item()}')


plt.plot(train_losses, label = 'train_loss')
plt.plot(test_losses, label = 'test_loss')
plt.legend()
plt.show()


plt.plot(train_accu, label = 'train_acc')
plt.plot(test_accu, label = 'test_acc')
plt.legend()
plt.show()


with torch.no_grad():
    p_train = model(X_train)
    p_train = np.round(p_train.numpy())
    train_acc = np.mean(Y_train.numpy() == p_train)

    p_test = model(X_test)
    p_test = np.round(p_test.numpy())
    test_acc = np.mean(Y_test.numpy() == p_test)
print(f'train_acc : {train_acc}, test_acc: {test_acc}')

model.state_dict()
torch.save(model.state_dict(), 'pytorch_classification_1.pt')

model2 = nn.Sequential(
                nn.Linear(D, 1),
                nn.Sigmoid()
)

model2.load_state_dict(torch.load('pytorch_classification_1.pt'))

with torch.no_grad():

    p_train = model2(X_train)
    p_train = np.round(p_train.numpy())
    train_acc = np.mean(Y_train.numpy() == p_train)

    p_test = model2(X_test)
    p_test = np.round(p_test.numpy())
    test_acc = np.mean(Y_test.numpy() == p_test)

print(f'Model 2: train_acc: {train_acc}, test_acc: {test_acc}')