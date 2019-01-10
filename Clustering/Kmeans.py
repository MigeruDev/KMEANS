import numpy as np # linear algebra

class Kmeans:

    def __init__(self, data, k):
        #_______________________________________________
        # Número de clusters
        self.k = k
        # DataSet
        self.data = data
        # 
        self.n = data.shape[0]
        self.c = data.shape[1]

        self.centers = np.array([data[0],data[400]])
        self.centers_old = np.zeros(self.centers.shape)
        self.clusters = np.zeros(self.n)
        self.distances = np.zeros((self.n,k))
        #_______________________________________________


    def k_means(self):
        pass
