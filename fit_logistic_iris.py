print("Here")
import pandas as pd
import torch
import numpy as np
import sys

print("Here")

def accuracy(z, y):
    with torch.no_grad():
        return ((((z >= 0).float() - 0.5) *(y - 0.5) + 0.5)).mean()

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
print("Here")

dtype = torch.float
torch.set_default_device("cpu")

print("Here")
X_train_t = torch.tensor(X_train, dtype=dtype)
y_train_t = torch.tensor(y_train, dtype=dtype)

b0 = torch.randn((), dtype=dtype, requires_grad=True)
b1 = torch.randn((), dtype=dtype, requires_grad=True)
b2 = torch.randn((), dtype=dtype, requires_grad=True)
b3 = torch.randn((), dtype=dtype, requires_grad=True)
b4 = torch.randn((), dtype=dtype, requires_grad=True)

learning_rate = 0.001

print("Here")
for e in range(5):
    print("Here now")
    z = b0 + b1*X_train_t[:,0] + b2*X_train_t[:,1] + b3*X_train_t[:,2] + b4*X_train_t[:,2]
    lax = torch.logaddexp(torch.zeros(X_train_t.shape[0]), z)
    yz = y_train_t * z
    loss = (lax - yz).sum()
    if (e % 100 == 1):
        print(loss.item(), accuracy(z, y_train_t))
        sys.stdout.flush()
    loss.backward(retain_graph=True)
    with torch.no_grad():
        b0 -= learning_rate*b0.grad
        b1 -= learning_rate*b1.grad
        b2 -= learning_rate*b2.grad
        b3 -= learning_rate*b3.grad
        b4 -= learning_rate*b4.grad
        b0.grad = None
        b1.grad = None
        b2.grad = None
        b3.grad = None
        b4.grad = None

