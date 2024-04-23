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

df = pd.read_csv('../cdr3_alpha_beta.csv')

df=remove_imbalance(df,threshold=10)
df=sampler(df, n_samples=2000, n_epitopes=10)
# head = None
# seqs = ['CAVSLDSNYQLIW','CILRVGATGGNNKLTL','CAMREPSGTYQRF']
# complex.id,cdr3_alpha,v.segm_alpha,j.segm_alpha,cdr3_beta,v.segm_beta,j.segm_beta,species,mhc.a,mhc.b,mhc.class,antigen.epitope,vdjdb.score
cdr3_alpha = df['cdr3_alpha'].tolist()
cdr3_beta = df['cdr3_beta'].tolist()
v_segm_alpha = df['v.segm_alpha'].tolist()
v_segm_beta = df['v.segm_beta'].tolist()
j_segm_alpha = df['j.segm_alpha'].tolist()
j_segm_beta = df['j.segm_beta'].tolist()
mhc_a = df['mhc.a'].tolist()
mhc_b = df['mhc.b'].tolist()
epitope = df['antigen.epitope'].tolist()
n_epitopes=len(set(epitope))

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
TCRs = [TCR(cdr3_alpha[i], cdr3_beta[i], v_segm_alpha[i], v_segm_beta[i], j_segm_alpha[i], j_segm_beta[i], mhc_a[i], mhc_b[i], epitope[i]) for i in range(len(cdr3_alpha))]
dist, indices = distance_cal(TCRs)

dist=dist_to_matrix(dist, indices,len(cdr3_alpha))
epitope_num=len(set(epitope))
cluster=DBSCAN(eps=8, min_samples=n_epitopes, metric='precomputed')
cluster.fit(dist)
from sklearn.metrics import adjusted_rand_score,silhouette_score
print(adjusted_rand_score(epitope, cluster.labels_))
# print(silhouette_score(seqs_mat,cluster.labels_))

# I think we should select a total of 2000-3000 TCRs corresponding to epitope in 5-10 for clustering.
# We do not need to use all the data in the dataset and all types of epitope in the dataset.