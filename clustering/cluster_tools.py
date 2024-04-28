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
