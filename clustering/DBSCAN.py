import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from encoders.ordinal_encode import seqs2mat
from tcr_sampler import sampler,remove_imbalance,transform_imbalance
from TCRs_distance import distance_cal,dist_to_matrix,TCR
from sklearn.cluster import DBSCAN
import pandas as pd
import itertools
from sklearn.metrics import adjusted_rand_score,silhouette_score,normalized_mutual_info_score
from cluster_tools import pure_clusters_fraction,pure_cluster_retention


df = pd.read_csv('../pre-processing final/cdr3_alpha_beta_df.csv')
# df = df[df['species'] == 'HomoSapiens']
df = df[df['species'] == 'MusMusculus']
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

'''
cdr3_alpha = df[1].tolist()
cdr3_beta = df[4].tolist()
v_segm_alpha = df[2].tolist()
v_segm_beta = df[5].tolist()
j_segm_alpha = df[3].tolist()
j_segm_beta = df[6].tolist()
mhc_a = df[8].tolist()
mhc_b = df[9].tolist()
epitope = df[11].tolist()
cdr3_alpha.pop(0)
cdr3_beta.pop(0)
v_segm_alpha.pop(0)
v_segm_beta.pop(0)
j_segm_alpha.pop(0)
j_segm_beta.pop(0)
mhc_a.pop(0)
mhc_b.pop(0)
epitope.pop(0)
n_epitopes=len(set(epitope))
'''

# TCRs = [TCR(cdr3_alpha[i], cdr3_beta[i], v_segm_alpha[i], v_segm_beta[i], j_segm_alpha[i], j_segm_beta[i], mhc_a[i], mhc_b[i], epitope[i]) for i in range(len(cdr3_alpha))]
# TCRs = [TCR(cdr3_alpha[i], None, v_segm_alpha[i], None, j_segm_alpha[i], None, mhc_a[i], None, epitope[i]) for i in range(num_tcrs)]
TCRs = [TCR(None,cdr3_beta[i],None,v_segm_beta[i],None,j_segm_beta[i],None,mhc_b[i],epitope[i]) for i in range(num_tcrs)]


dist, indices = distance_cal(TCRs)

dist=dist_to_matrix(dist, indices,len(cdr3_alpha)).astype(np.float64)

# save the distance matrix
# np.save('distance_matrix.npy', dist)
# dist = np.load('distance_matrix.npy')

epitope_num=len(set(epitope))

cluster=DBSCAN(eps=22, min_samples=4, metric='precomputed')
cluster.fit(dist)

# print(adjusted_rand_score(epitope, cluster.labels_))
print(cluster.labels_.tolist())
print(pure_clusters_fraction(cluster.labels_, epitope))
print(pure_cluster_retention(cluster.labels_, epitope))
print(normalized_mutual_info_score(epitope, cluster.labels_))

'''
def df2dict(df):
    cluster_id = df.index
    score = df.values
    return dict(zip(cluster_id, score))
# load the distance matrix

# dist = np.load('distance_matrix.npy')
eps_list = [i for i in range(10, 120, 2)]

purity_fraction = []
purity_retention = []
nmi=[]

for eps in eps_list:
    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed')
    cluster.fit(dist)

    purity_fraction.append(pure_clusters_fraction(cluster.labels_, epitope))
    purity_retention.append(pure_cluster_retention(cluster.labels_, epitope))
    nmi.append(normalized_mutual_info_score(epitope, cluster.labels_))

import matplotlib.pyplot as plt
plt.plot(eps_list, purity_fraction, label='Purity Fraction')
plt.plot(eps_list, purity_retention, label='Purity Retention')
plt.plot(eps_list, nmi, label='NMI')
plt.legend()
plt.show()
'''




# from left to right: purity fraction, purity retention, NMI
# human combined (eps=70, min_samples=4) 0.42857142857142855 0.11241970021413276 0.4069359548003655
# human alpha (eps=22, min_samples=4) 0.1951219512195122 0.043897216274089934 0.426454793081235
# human beta (eps=22, min_samples=4) 0.39655172413793105 0.08029978586723768 0.37061544815295017

# mouse combined (eps=70, min_samples=4) 0.2777777777777778 0.04212860310421286 0.2627689472275712
# mouse alpha (eps=22, min_samples=4) 0.22580645161290322 0.06319290465631928 0.411312103709991
# mouse beta (eps=22, min_samples=4) 0.2631578947368421 0.038802660753880266 0.4108547695686911



