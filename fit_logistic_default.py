import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def nll(theta, X, y):
    z = np.insert(X, 0, 1, 1).dot(theta)
    yz = z*y
    log1pexpz = np.log1p(np.exp(z))
    return np.sum(log1pexpz - yz)

datafile = sys.argv[1]
df = pd.read_csv(datafile)
X = default[["balance", "income"]].values
y = np.asarray([{"Yes": 1., "No": 0.}[x] for x in default["default"]])

m = minimize(nll, np.random.rand(3,), args=(X, y), method="L-BFGS-B")

