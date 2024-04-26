import numpy as np
import matplotlib.pyplot as plt
from encoders.GIANA_encoder import GIANA_encoder
import pandas as pd
from TCRs_distance import TCR
import seaborn as sns
from umap import UMAP
from tcr_sampler import sampler,remove_imbalance,transform_imbalance
import pandas as pd
from tcrdist.repertoire import TCRrep
from sklearn.metrics import silhouette_score


df = pd.read_csv('pre-processing final/cdr3_alpha_beta_df.csv')

human = df[df['species'] == 'HomoSapiens']
human = remove_imbalance(human, threshold=10)
# 输出每个抗原的数量
print(human['epitope'].value_counts())

mouse = df[df['species'] == 'MusMusculus']
# mouse = remove_imbalance(mouse, threshold=10)


epitope_human=human['epitope'].tolist()
cdr3_alpha_human = human['cdr3_a_aa'].tolist()
cdr3_beta_human = human['cdr3_b_aa'].tolist()
TCRs_human = [TCR(cdr3_alpha_human[i], cdr3_beta_human[i]) for i in range(len(cdr3_beta_human))]
print(len(set(epitope_human)))
'''
epitope_mouse=mouse['epitope'].tolist()
cdr3_alpha_mouse = mouse['cdr3_a_aa'].tolist()
cdr3_beta_mouse = mouse['cdr3_b_aa'].tolist()
TCRs_mouse = [TCR(cdr3_alpha_mouse[i], cdr3_beta_mouse[i]) for i in range(len(cdr3_beta_mouse))]
print(len(set(epitope_mouse)))
'''
CDR3_encoded=GIANA_encoder(TCRs_human)
print(silhouette_score(CDR3_encoded, epitope_human))
# save the encoded data
# np.save('pre-processing final/human_CDR3_encoded.npy', CDR3_encoded)
# load the encoded data
# CDR3_encoded = np.load('pre-processing final/human_CDR3_encoded.npy')

reducer = UMAP(random_state=42)
human_cdr3_reduced_origin = reducer.fit_transform(CDR3_encoded)
human_cdr3_reduced_origin = pd.DataFrame(human_cdr3_reduced_origin, columns=['UMAP1', 'UMAP2'])
# human_cdr3_reduced_origin['epitope'] = epitope_human

# silhouette_score
print(silhouette_score(human_cdr3_reduced_origin, epitope_human))

# plt.figure(figsize=(15, 10))
# sns.scatterplot(x='UMAP1', y='UMAP2', data=human_cdr3_reduced_origin, hue='epitope', s=20, legend=False)
# plt.show()

'''
CDR3_encoded=GIANA_encoder(TCRs_mouse)
reducer = UMAP(random_state=42)
mouse_cdr3_reduced = reducer.fit_transform(CDR3_encoded)
mouse_cdr3_reduced = pd.DataFrame(mouse_cdr3_reduced, columns=['UMAP1', 'UMAP2'])
mouse_cdr3_reduced['epitope'] = epitope_mouse
plt.figure(figsize=(10, 10))
sns.scatterplot(x='UMAP1', y='UMAP2', data=mouse_cdr3_reduced, hue='epitope',s=30, legend=False)
plt.show()
'''
'''
tr = TCRrep(cell_df = mouse,
            organism = 'mouse',
            chains = ['alpha','beta'],
            db_file = 'alphabeta_gammadelta_db.tsv')
mouse_matrix = tr.pw_cdr3_b_aa
reducer = UMAP(random_state=42)
mouse_cdr3_reduced = reducer.fit_transform(mouse_matrix)
mouse_cdr3_reduced = pd.DataFrame(mouse_cdr3_reduced, columns=['UMAP1', 'UMAP2'])
mouse_cdr3_reduced['epitope'] = epitope_mouse
plt.figure(figsize=(10, 10))
sns.scatterplot(x='UMAP1', y='UMAP2', data=mouse_cdr3_reduced, hue='epitope',s=30, legend=False)
plt.show()
'''
tr=TCRrep(cell_df=human,
              organism='human',
              chains=['alpha','beta'],
              db_file='alphabeta_gammadelta_db.tsv')
human_matrix=tr.pw_cdr3_b_aa
reducer = UMAP(random_state=42)
human_cdr3_reduced_dist = reducer.fit_transform(human_matrix)
human_cdr3_reduced_dist = pd.DataFrame(human_cdr3_reduced_dist, columns=['UMAP1', 'UMAP2'])
# human_cdr3_reduced_dist['epitope'] = epitope_human
# silhouette_score
print(silhouette_score(human_cdr3_reduced_dist, epitope_human))
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x='UMAP1', y='UMAP2', data=human_cdr3_reduced_dist, hue='epitope',s=20, legend=False)
# plt.show()