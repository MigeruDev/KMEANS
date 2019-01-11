import numpy as np # linear algebra
from copy import deepcopy
from sklearn.metrics import confusion_matrix

class Kmeans:

    def __init__(self, data, k):
        #_______________________________________________
        # Número de clusters
        self.k = k
        # DataSet
        self.data = data
        # Datos de entrenamiento
        self.n = data.shape[0]
        # Número de entradas del DataSet
        self.c = data.shape[1]
        # Almacena los centroides
        self.centers = np.array([data[0],data[100]])
        # Almacena los centroides anteriores
        self.centers_old = np.zeros(self.centers.shape)
        # Clasifica cada registro del DataSet dentro de un Cluster
        self.clusters = np.zeros(self.n)
        # Almacena las distancais de cada registro con su centroide
        self.distances = np.zeros((self.n,k))
        #_______________________________________________


    def k_means(self):
        error = np.linalg.norm(self.centers-self.centers_old)

        while error != 0:
            # Para cada registro se calcula la distancia con su centroide
            for i in range(self.k):
                self.distances[:, i] = np.linalg.norm(
                    self.data - self.centers[i], axis=1)

            self.clusters = np.argmin(self.distances, axis=1)

            self.centers_old = deepcopy(self.centers)

            # Se actualizan los centroides con la media
            for i in range(self.k):
                self.centers[i] = np.mean(
                    self.data[self.clusters == i], axis=0)

            # Calcula el error
            error = np.linalg.norm(self.centers - self.centers_old)

            # print("clusters\n",clusters)
            # print("distances\n",distances)
            # print("error = ",error)
            # print("new centers\n",centers_new)

        return self.clusters

    def confusion_matrix(self, y, c, category):
        print("*******MATRIZ DE CONFUSION********")
        cm = confusion_matrix(y, c)
        print("","  ".join(category))
        for i in range(len(cm)):
            print("%s %s"%(cm[i],category[i]))
        print("\n")