import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE as tsne_sklearn

# Cargar el conjunto de datos de iris
iris = load_iris()

# sklearn
tsne_sk = tsne_sklearn(n_components=2, perplexity=30, learning_rate=200)
embedded_data_sk = tsne_sk.fit_transform(iris.data)

# Visualizar los resultados
plt.subplot(1,2,1)
plt.scatter(embedded_data_sk[:, 0], embedded_data_sk[:, 1], c=iris.target)


# unsupervised jim
from unsupervised_jim import TSNE as tsne_unsupervised
tsne_jim = tsne_unsupervised(n_components=2, perplexity=30, learning_rate=200)
embedded_data_jim = tsne_jim.fit_transform(iris.data)

# Visualizar los resultados
plt.subplot(1,2,2)
plt.scatter(embedded_data_jim[:, 0], embedded_data_jim[:, 1], c=iris.target)
plt.show()
