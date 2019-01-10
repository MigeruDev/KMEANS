import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt


def average(cluster_points, numdata):
    sums = []
    centroids = []
    for cluster in cluster_points:
        for i in range(numdata):
            sums.append(0)
        n = len(cluster)
        for points in cluster:
            for i in range(numdata):
                sums[i] += points[i]
        centroids.append([x/n for x in sums])
    return centroids

df = pd.read_csv("bank-data2.csv", delimiter=';') #load the dataset
df.head()
data = df.values[:, 0:10]

centroids = []
centroids.append(data[0])
centroids.append(data[50])

converge = False
it_counter = 0
iterations = 10
while (not converge) and (it_counter < iterations):
    cluster_points = [[],[]]
    # Assign points in nearest centroid
    for p in data:
        print("old centroids= ",centroids)
        d1 = np.linalg.norm(centroids[0]-p)
        d2 = np.linalg.norm(centroids[1]-p)
        distances = [d1,d2]
        point = min(distances)
        cluster_points[distances.index(point)].append(p)
    #print("STEP -")
    #print(cluster_points)

    centroids = average(cluster_points,10)
    print("centroids = \n",centroids)
    # Check that converge all Clusters
    #converge = [c.converge for c in centroids].count(False) == 0

    # Increment counter and delete lists of clusters points
    it_counter += 1
