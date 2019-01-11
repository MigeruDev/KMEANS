
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

from Clustering.Kmeans import Kmeans

df = pd.read_csv("../bank-data2.csv", delimiter=';') #load the dataset
df.head()

#______________________________________________
# Numero de clusters
k = 2
# DataSet
data = df.values[:, 0:10]
# Classifications
y = df['pep']
# Clustering (para graficar)
C = df.loc[:,:'mortgage']
# Classification (para graficar)
X = df.loc[:,:'pep']
# Colores
colors = ['blue', 'green']
# Categoria
category = ['NO', 'YES']
#_______________________________________________

kmeans = Kmeans(data, k)
cluster = kmeans.k_means()

# Obtenemos matriz de confusión
kmeans.confusion_matrix(y,y,category)
kmeans.confusion_matrix(y,cluster,category)

# Graficamos el DataSet con Clasificacion
# Normalizamos los datos
X_norm = (X - X.min())/(X.max() - X.min())
# Convertimos N-Dimensiones en 2-D
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))

plt.subplots()
plt.title('Classification')
plt.scatter(transformed[y==0][0], transformed[y==0][1],
            label='NO', c='red', s=3)
plt.scatter(transformed[y==1][0], transformed[y==1][1],
            label='YES', c='blue', s=3)

# Graficamos el DataSet con Clustering
# Formamos un nuevo data set con nuestro Cluster
T =np.array(cluster)
C['cluster'] = pd.Series(T.T, index=C.index)
print(C)
print(X)

# Normalizamos los datos
X_norm = (C - C.min())\
         /(C.max() - C.min())
# Convertimos N-Dimensiones en 2-D
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))

plt.subplots()
plt.title('Clustering')
plt.scatter(transformed[y==0][0], transformed[y==0][1],
            label='NO', c='red', s=3)
plt.scatter(transformed[y==1][0], transformed[y==1][1],
            label='YES', c='blue', s=3)

plt.legend()
plt.show()