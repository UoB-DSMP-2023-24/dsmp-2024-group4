from Bio.Align import substitution_matrices
from collections import OrderedDict
import numpy as np
import pandas as pd

def BLOSUM90():
    # load matrix as symmetric pandas dataframe
    blosum100 = substitution_matrices.load("BLOSUM90")
    # Extract row and column names (unique amino acids)
    amino_acids = sorted(set(k[0] for k in blosum100.keys()).union(set(k[1] for k in blosum100.keys())))

    # Create DataFrame with proper row and column names
    aa_df = pd.DataFrame(index=amino_acids, columns=amino_acids)
    # Fill DataFrame with blosum100 values
    for (aa1, aa2), score in blosum100.items():
        aa_df.at[aa1, aa2] = score
        aa_df.at[aa2, aa1] = score  # Ensure symmetry
    # only keep standard protein letters
    standard_protein_letters = "IRQCYMLVAFNESHKWGDTP"
    aa_df = aa_df.loc[list(standard_protein_letters), list(standard_protein_letters)]

    # add NULL '-' with max loss (min value) & median for match
    aa_df.loc['-', :] = aa_df.values.min()
    aa_df.loc[:, '-'] = aa_df.values.min()
    aa_df.loc['-', '-'] = np.median(aa_df.values.diagonal())

    return aa_df

def BLOSUM62():
    blosum100 = substitution_matrices.load("BLOSUM62")
    amino_acids = sorted(set(k[0] for k in blosum100.keys()).union(set(k[1] for k in blosum100.keys())))

    aa_df = pd.DataFrame(index=amino_acids, columns=amino_acids)
    for (aa1, aa2), score in blosum100.items():
        aa_df.at[aa1, aa2] = score
        aa_df.at[aa2, aa1] = score  # Ensure symmetry
    standard_protein_letters = "IRQCYMLVAFNESHKWGDTP"
    aa_df = aa_df.loc[list(standard_protein_letters), list(standard_protein_letters)]

    # add NULL '-' with max loss (min value) & median for match
    aa_df.loc['-', :] = aa_df.values.min()
    aa_df.loc[:, '-'] = aa_df.values.min()
    aa_df.loc['-', '-'] = np.median(aa_df.values.diagonal())

    return aa_df

def BLOSUM80():
    blosum100 = substitution_matrices.load("BLOSUM80")
    amino_acids = sorted(set(k[0] for k in blosum100.keys()).union(set(k[1] for k in blosum100.keys())))

    aa_df = pd.DataFrame(index=amino_acids, columns=amino_acids)
    for (aa1, aa2), score in blosum100.items():
        aa_df.at[aa1, aa2] = score
        aa_df.at[aa2, aa1] = score  # Ensure symmetry
    standard_protein_letters = "IRQCYMLVAFNESHKWGDTP"
    aa_df = aa_df.loc[list(standard_protein_letters), list(standard_protein_letters)]

    # add NULL '-' with max loss (min value) & median for match
    aa_df.loc['-', :] = aa_df.values.min()
    aa_df.loc[:, '-'] = aa_df.values.min()
    aa_df.loc['-', '-'] = np.median(aa_df.values.diagonal())

    return aa_df


def encode_cdr3(seq,bio_matrix_name,maxlength):
    if bio_matrix_name == 'BLOSUM90':
        bio_matrix = BLOSUM90()
    elif bio_matrix_name == 'BLOSUM62':
        bio_matrix = BLOSUM62()
    elif bio_matrix_name == 'BLOSUM80':
        bio_matrix = BLOSUM80()
    else:
        raise ValueError('No such matrix')
    seq_embed = [bio_matrix[aa].tolist() for aa in seq]
    if len(seq_embed) < maxlength:
        pad = [bio_matrix['-'].tolist() for _ in range(maxlength - len(seq_embed))]
        seq_embed.extend(pad)
    elif len(seq_embed) > maxlength:
        seq_embed = seq_embed[:maxlength]
    return seq_embed

# read csv
df = pd.read_csv('../vdjdb.csv', header=None)
cdr3=df[2].tolist()
epitope=df[9].tolist()
# delete the first row('cdr3')
cdr3.pop(0)
epitope.pop(0)
aa_idx = BLOSUM90()
encode_output= []
for i in cdr3:
    print(encode_cdr3(i,aa_idx,20))
    encode_output.append(encode_cdr3(i,aa_idx,20))

# USE PCA to reduce the dimension of the data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
flatten_output = []
for i in encode_output:
    flatten_output.append(np.array(i).flatten())
pca_output = pca.fit_transform(flatten_output)
# show with a plot
import matplotlib.pyplot as plt
plt.scatter(pca_output[:, 0], pca_output[:, 1])
plt.show()

'''
unique_epitopes = list(set(epitope))
epitope_colors = {epi: i / len(unique_epitopes) for i, epi in enumerate(unique_epitopes)}
color_values = [epitope_colors[epi] for epi in epitope]

plt.scatter(pca_output[:, 0], pca_output[:, 1], c=color_values, cmap='viridis')
plt.show()
'''