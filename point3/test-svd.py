from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD as svd_sklearn

import matplotlib.pyplot as plt

# Cargar el conjunto de datos Iris
iris = load_iris()
iris_data = iris.data

# Crear un objeto TruncatedSVD y ajustarlo a los datos
svd_sk = svd_sklearn(n_components=2)
svd_transfor_sk = svd_sk.fit_transform(iris_data)

# Mostrar los resultados
print("Datos originales (shape):", svd_transfor_sk.shape)
print("Datos transformados (shape):", svd_transfor_sk.shape)

# Graficar los datos transformados
plt.scatter(svd_transfor_sk[:, 0], svd_transfor_sk[:, 1], c=iris.target)
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title("Datos de Iris transformados con SVD")

# unsupervised jim
from unsupervised_jim import SVD as svd_unsupervised
svd_jim = svd_unsupervised(n_components=2)
svd_transfor_jim = svd_jim.fit_transform(iris_data)

# Mostrar los resultados
print("Datos originales (shape):", svd_transfor_jim.shape)
print("Datos transformados (shape):", svd_transfor_jim.shape)

# Graficar los datos transformados
plt.scatter(svd_transfor_jim[:, 0], svd_transfor_jim[:, 1], c=iris.target)
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title("Datos de Iris transformados con SVD")

plt.show()
