import pandas as pd
import torch
import numpy as np
import sys

def accuracy(z, y):
    with torch.no_grad():
        return (z*(y - 0.5) >= 0).float().mean()

np.random.seed(42)

iris = pd.read_csv("iris.csv")
train = iris.groupby("variety", group_keys=False).apply(
    lambda x: x.sample(int(0.9*len(x)))
)
test = iris.loc[iris.index.difference(train.index.values),:]

X_train = train[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
y_train = np.asarray([{"Virginica": 1, "Setosa": 0, "Versicolor": 0}[y] for y in train["variety"]])

X_test = test[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
y_test = np.asarray([{"Virginica": 1, "Setosa": 0, "Versicolor": 0}[y] for y in test["variety"]])

dtype = torch.float
torch.set_default_device("cpu")

X_train_t = torch.tensor(X_train, dtype=dtype)
y_train_t = torch.tensor(y_train, dtype=dtype)

lin = torch.nn.Linear(4, 1, bias=True, dtype=dtype)

learning_rate = 0.01

for e in range(20000):
    z = lin(X_train_t)
    lax = torch.logaddexp(torch.zeros(X_train_t.shape[0]), z)
    yz = y_train_t * z
    loss = (lax - yz).sum()
    if (e % 100 == 1):
        print(loss.item(), accuracy(z, y_train_t))
        sys.stdout.flush()
    lin.zero_grad()
    loss.backward(retain_graph=True)
    with torch.no_grad():
        for p in lin.parameters():
            p_new = p - learning_rate*p.grad
            p.copy_(p_new)
