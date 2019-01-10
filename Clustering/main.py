from copy import deepcopy

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt


df = pd.read_csv("../bank-data2.csv", delimiter=';') #load the dataset
df.head()
data = df.values[:, 0:10]
#category = df.values[:, 10]

#______________________________________________
#Numero de clusters
k = 2
#Datos de entrenamiento
n = data.shape[0]
#Numero de entradas del DataSet
c = data.shape[1]
#Colores
colors = ['blue', 'green']
#_______________________________________________

centroides = np.array([data[0],data[400]])  # Almacena los centroides

centers_old = np.zeros(centroides.shape)    # Almacena centroides anteriores
centers_new = deepcopy(centroides)          # Almacena centroides actuales
clusters = np.zeros(n)                      # Clasifica cada registro del DataSet
distances = np.zeros((n,k))                 # Almacena las distancias de cada registro con su centroide

error = np.linalg.norm(centers_new-centers_old) # Cuando las distancias son iguales (converge)
                                                # el error es 0

while error!=0:                             # Itera hasta que error es 0

    # Para cada registro se calcula la distancia con su centroide
    for i in range(k):
        distances[:,i] = np.linalg.norm(data - centers_new[i],axis = 1)

    clusters = np.argmin(distances,axis=1)

    centers_old = deepcopy(centers_new)

    # Se actualizan los centroides con la media
    for i in range(k):
        centers_new[i] = np.mean(data[clusters==i], axis=0)

    # Calcula el error
    error = np.linalg.norm(centers_new-centers_old)

    #print("clusters\n",clusters)
    #print("distances\n",distances)
    #print("error = ",error)
    #print("new centers\n",centers_new)


#plt.show()