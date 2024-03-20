from encoders.ordinal_encode import seqs2mat
from tcr_sampler import sampler
from CDR3distance import distance_cal,dist_to_matrix
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import itertools

df = pd.read_csv('../vdjdb.csv')
df=sampler(df, n_samples=2000, n_epitopes=5)
# head = None

cdr3=df['cdr3'].tolist()
epitope=df['antigen.epitope'].tolist()
seqs_mat, seqs_L = seqs2mat(cdr3) # seqs_mat is a matrix of the sequences, seqs_L is a vector of the lengths of the sequences
dist,indices = distance_cal(seqs_mat, seqs_L)
dist_matrix=dist_to_matrix(dist, indices,len(cdr3))
epitope_num=len(set(epitope))
cluster=AgglomerativeClustering(n_clusters=5,affinity='precomputed', linkage='complete')
cluster.fit(dist_matrix)
from sklearn.metrics import adjusted_rand_score,silhouette_score
print(adjusted_rand_score(epitope, cluster.labels_))
print(silhouette_score(seqs_mat,cluster.labels_))