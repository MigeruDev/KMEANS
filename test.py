
from pandas.plotting import scatter_matrix
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA


df = pd.read_csv("bank-data2.csv", delimiter=';') #load the dataset
df.head()
data = df.values[:, 0:10]
category = df.values[:, 10]
'''
scatter_matrix(df)
plt.show()
'''
y = df['pep']          # Split off classifications
X = df.loc[:,:'mortgage']          # Split off features
X_norm = (X - X.min())/(X.max() - X.min())
#print(X)
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))

print("transformed = \n",transformed)
plt.subplots()
plt.title('Clustering')
plt.scatter(transformed[y==0][0], transformed[y==0][1], label='NO', c='red')
plt.scatter(transformed[y==1][0], transformed[y==1][1], label='YES', c='blue')
print("transformed[y==0][0]\n",transformed[y==0][0])
print('Y\n',y)


X = df.loc[:,:'pep']          # Non split off features
X_norm = (X - X.min())/(X.max() - X.min())
#print(X)
transformed = pd.DataFrame(pca.fit_transform(X_norm))

plt.subplots()
plt.title('Classification')
plt.scatter(transformed[y==0][0], transformed[y==0][1], label='NO', c='red')
plt.scatter(transformed[y==1][0], transformed[y==1][1], label='YES', c='blue')

plt.legend()
plt.show()
