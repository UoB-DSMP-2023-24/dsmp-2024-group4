import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Calculate the purity of each cluster
def calculate_purity(df):
    cluster_purity = df.groupby('cluster_id').apply(lambda x: x['epitope'].value_counts().max() / len(x))
    return cluster_purity

# Define pure clusters: clusters with a purity of 1
def pure_clusters_fraction(cluster_labels, target_labels):
    df=pd.DataFrame({'cluster_id': cluster_labels, 'epitope': target_labels})
    purity = calculate_purity(df)
    pure_cluster_fraction = (purity == 1).sum() / len(purity)
    return pure_cluster_fraction


# Calculate Pure Cluster Retention
def pure_cluster_retention(cluster_labels, target_labels):
    df = pd.DataFrame({'cluster_id': cluster_labels, 'epitope': target_labels})
    purity = calculate_purity(df)
    pure_clusters = purity[purity == 1].index
    total_pure_tcrs = df[df['cluster_id'].isin(pure_clusters)].shape[0]
    total_tcrs = df.shape[0]
    pure_cluster_retention_rate = total_pure_tcrs / total_tcrs

    return pure_cluster_retention_rate