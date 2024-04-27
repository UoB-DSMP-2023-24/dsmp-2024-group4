import pandas as pd
from TCRs_distance import distance_cal, dist_to_matrix, TCR,matrix_position
from tcr_sampler import remove_imbalance, sampler
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score


def dbscan(length, eps, min_samples, dist_matrix):
    labels = [-1] * length  # -1 表示未分类，初始化所有点为未分类
    cluster_id = 0

    def find_neighbors(i):
        return [j for j in range(length) if i != j and dist_matrix[matrix_position(length,i,j)] <= eps]

    for i in range(length):
        if labels[i] != -1:
            continue  # 如果已经访问过，跳过

        neighbors = find_neighbors(i)
        if len(neighbors) < min_samples:
            labels[i] = -2  # 标记为噪声点
            continue

        # 创建新的聚类
        labels[i] = cluster_id
        seeds = set(neighbors)
        while seeds:
            current = seeds.pop()
            if labels[current] == -2:
                labels[current] = cluster_id
            if labels[current] != -1:
                continue

            labels[current] = cluster_id
            new_neighbors = find_neighbors(current)
            if len(new_neighbors) >= min_samples:
                seeds.update(new_neighbors)

        cluster_id += 1  # 递增聚类ID

    return labels


df = pd.read_csv('../pre-processing final/cdr3_alpha_beta_df.csv')
df = df[df['species'] == 'HomoSapiens']
df = remove_imbalance(df, threshold=10)
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

# combine alpha and beta chain
TCRs = [TCR(cdr3_alpha[i], cdr3_beta[i], v_segm_alpha[i], v_segm_beta[i], j_segm_alpha[i], j_segm_beta[i], mhc_a[i], mhc_b[i], epitope[i]) for i in range(num_tcrs)]
dist,indice = distance_cal(TCRs)
# 归一化
dist = (dist - dist.min()) / (dist.max() - dist.min())
'''
# save dist
with open('dist_sample.txt', 'w') as f:
    for item in dist:
        f.write("%s\n" % item)

# load dist
dist = []
with open('dist_sample.txt', 'r') as f:
    for line in f:
        dist.append(float(line.strip()))
'''
# dist_matrix = dist_to_matrix(dist, indices, num_tcrs)
# print(dist_matrix)


import matplotlib.pyplot as plt
k = 4
k_distances = []

for i in range(num_tcrs):
    point_distances = [dist[matrix_position(num_tcrs, i, j)] for j in range(num_tcrs) if i != j]
    sorted_point_distances = sorted(point_distances)
    k_distances.append(sorted_point_distances[k-1])  # k-1 因为索引从0开始

# 对第15近邻距离进行排序
k_distances_sorted = sorted(k_distances, reverse=True)

# 绘制k-distance图
plt.figure(figsize=(10, 5))
plt.plot(k_distances_sorted)
plt.ylabel('Distance')
plt.xlabel('Points')
plt.title('K-Distance Graph (k=4)')
plt.grid(True)
# save
plt.savefig('elbow_method.png')
# save k_distances_sorted
with open('k_distances_sorted.txt', 'w') as f:
    for item in k_distances_sorted:
        f.write("%s\n" % item)
'''

# 使用自定义的DBSCAN
cluster_labels = dbscan(len(epitope), eps=100, min_samples=4, dist_matrix=dist)
print("Cluster labels:", cluster_labels)
print("Silhouette score:", silhouette_score(dist_to_matrix(dist,indice,len(cdr3_alpha)).astype(float), cluster_labels, metric='precomputed'))
print("Davies-Bouldin score:", davies_bouldin_score(dist_to_matrix(dist,indice,len(cdr3_alpha)).astype(float), cluster_labels))
print("Calinski-Harabasz score:", calinski_harabasz_score(dist_to_matrix(dist,indice,len(cdr3_alpha)).astype(float), cluster_labels))
'''
