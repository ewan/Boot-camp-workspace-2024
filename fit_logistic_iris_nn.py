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
y_train = np.asarray([{"Virginica": 1, "Setosa": 1, "Versicolor": 0}[y] for y in train["variety"]])

X_test = test[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
y_test = np.asarray([{"Virginica": 1, "Setosa": 1, "Versicolor": 0}[y] for y in test["variety"]])

dtype = torch.float
torch.set_default_device("cpu")

X_train_t = torch.tensor(X_train, dtype=dtype)
y_train_t = torch.tensor(y_train, dtype=dtype)

lin1 = torch.nn.Linear(4, 2, bias=True, dtype=dtype)
lin2 = torch.nn.Linear(2, 1, bias=True, dtype=dtype)

loss_fn = torch.nn.BCEWithLogitsLoss()

learning_rate = 0.1

optimizer = torch.optim.Adam(list(lin1.parameters()) + 
                              list(lin2.parameters()), lr=learning_rate)

for e in range(20000):
    optimizer.zero_grad()
    hidden = lin1(X_train_t)
    hidden_s = torch.nn.functional.sigmoid(hidden)
    z = lin2(hidden_s).squeeze()
    loss = loss_fn(z, y_train_t)
    if (e % 100 == 1):
        print(loss.item(), accuracy(z, y_train_t))
        sys.stdout.flush()
    loss.backward(retain_graph=True)
    optimizer.step()
