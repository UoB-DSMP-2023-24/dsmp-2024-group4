from sklearn.cluster import KMeans
from encoders.bio_matrices_encode import encode_cdr3
import pandas as pd
import numpy as np


df = pd.read_csv('vdjdb.csv', header=None)
cdr3=df[2].tolist()
epitope=df[9].tolist()
# delete the first row('cdr3')
cdr3.pop(0)
epitope.pop(0)
encode_output= []
for i in cdr3:
    encode_output.append(encode_cdr3(i,'BLOSUM90',20))

cdr3_flattened = [np.array(seq).flatten() for seq in encode_output]

#from sklearn.decomposition import PCA
#pca = PCA(n_components=20)
#pca_output = pca.fit_transform(cdr3_flattened)

kmeans = KMeans(n_clusters=1169) # 1169 is the number of unique epitopes
kmeans.fit(cdr3_flattened)

from sklearn.metrics import adjusted_rand_score,silhouette_score
print(adjusted_rand_score(epitope, kmeans.labels_)) # 0.00311186860990052 this result implies that the clustering is not better than random
print(silhouette_score(cdr3_flattened,kmeans.labels_)) # 0.13751468062377964

