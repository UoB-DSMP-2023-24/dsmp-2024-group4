from sklearn.cluster import KMeans
from encoders.bio_matrices_encode import encode_cdr3
from TCRs_distance import distance_cal,dist_to_matrix,TCR
import pandas as pd
import numpy as np


df = pd.read_csv('../pre_processing_final/cdr3_alpha_beta_df.csv')
df = df[df['species'] == 'HomoSapiens']
# df = remove_imbalance(df, threshold=10)
# df = sampler(df, n_samples=2000, n_epitopes=10)
# head = None
# seqs = ['CAVSLDSNYQLIW','CILRVGATGGNNKLTL','CAMREPSGTYQRF']
# complex.id,cdr3_a_aa,v_a_gene,j_a_gene,species,mhc.a,mhc.b,mhc.class,epitope,vdjdb.score,cdr3_b_aa,v_b_gene,j_b_gene
cdr3_alpha = df['cdr3_a_aa'].tolist()
cdr3_beta = df['cdr3_b_aa'].tolist()
v_segm_alpha = df['v_a_gene'].tolist()
v_segm_beta = df['v_b_gene'].tolist()
j_segm_alpha = df['j_a_gene'].tolist()
j_segm_beta = df['j_b_gene'].tolist()
mhc_a = df['mhc.a'].tolist()
mhc_b = df['mhc.b'].tolist()
epitope = df['epitope'].tolist()
n_epitopes = len(set(epitope))

num_tcrs = len(cdr3_alpha)
TCRs = [TCR(cdr3_alpha[i], cdr3_beta[i], v_segm_alpha[i], v_segm_beta[i], j_segm_alpha[i], j_segm_beta[i], mhc_a[i], mhc_b[i], epitope[i]) for i in range(len(cdr3_alpha))]
dist, indices = distance_cal(TCRs)

dist=dist_to_matrix(dist, indices,len(cdr3_alpha)).astype(np.float64)

#from sklearn.decomposition import PCA
#pca = PCA(n_components=20)
#pca_output = pca.fit_transform(cdr3_flattened)

kmeans = KMeans(n_clusters=5) # 1169 is the number of unique epitopes
kmeans.fit(dist)

from sklearn.metrics import adjusted_rand_score,silhouette_score
# print(adjusted_rand_score(epitope, kmeans.labels_)) # 0.00311186860990052 this result implies that the clustering is not better than random
print(silhouette_score(dist,kmeans.labels_,metric='precomputed'))

