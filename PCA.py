import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

X = np.random.multivariate_normal(np.array([0,0]), np.array([[1,0],[0,1]]), 500).T
X = (X.T - np.mean(X, axis = 1)).T

C = (1 / X.shape[0]) * np.matmul(X, X.T)
eigenValues, eigenVectors = LA.eig(C)


Y = X / np.sqrt(X.shape[0])
U, s, Vt = LA.svd(Y)
pc = U @ np.diag(s)
pc = pc[:,::-1]

explained_variance = np.var(pc, axis=0)
explained_variance.cumsum()

plt.scatter([0,1], explained_variance)
plt.scatter([0,1], explained_variance.cumsum())
plt.scatter(X[0,:], X[1,:])
