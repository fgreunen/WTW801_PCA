from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

mean = [0, 0]
cov = [[2, 0.854], [0.854, 1]]
X = np.random.multivariate_normal(mean, cov, 500)
plt.scatter(X[:,0], X[:,1])
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pca.fit_transform(X)

eachexplained = pca.explained_variance_ratio_
cumexplained = pca.explained_variance_ratio_.cumsum()

print(cumexplained)