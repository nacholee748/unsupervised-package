
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Cargar los datos
iris = load_iris()
X = iris.data

# Sklearn
from sklearn.decomposition import PCA as pca_sk
pca_sklearn = pca_sk(n_components=2)
X_pca_sk = pca_sklearn.fit_transform(X)

# Visualizar los datos transformados
plt.subplot(141)
plt.scatter(X_pca_sk[:, 0], X_pca_sk[:, 1], c=iris.target)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')

# Unsupervised JIM PCA
from unsupervised_jim import PCA as pca_unsupervised
pca_jim = pca_unsupervised(n_components=2)
pca_jim.fit(X)
X_pca_jim = pca_jim.fit_transform(X)

# Visualizar los datos transformados
plt.subplot(142)
plt.scatter(X_pca_jim[:, 0], X_pca_jim[:, 1], c=iris.target)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')

# Unsupervised JIM PCA
from unsupervised_jim import PCA2 as pca_unsupervised2
pca_jim2 = pca_unsupervised2(n_components=2)
pca_jim2.fit(X)
X_pca2 = pca_jim2.transform(X)

# Visualizar los datos transformados
plt.subplot(143)
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=iris.target)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')


# Visualizar los datos transformados
plt.subplot(144)
plt.scatter(X[:, 0], X[:, 1], c=iris.target)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')

plt.show()
