import numpy as np
import pandas as pd
def one_hot_encode_cdr3(cdr3_sequence):
    standard_protein_letters = "IRQCYMLVAFNESHKWGDTP"
    encoding = np.zeros((len(cdr3_sequence), len(standard_protein_letters)))

    for i, amino_acid in enumerate(cdr3_sequence):
        if amino_acid in standard_protein_letters:
            encoding[i, standard_protein_letters.index(amino_acid)] = 1

    return encoding

# Example usage
standard_protein_letters = "IRQCYMLVAFNESHKWGDTP"
# read csv
df = pd.read_csv('../vdjdb.csv', header=None)
cdr3=df[2].tolist()
cdr3.pop(0)
encode_output= []
for cdr3_sequence in cdr3:
    encoded_cdr3 = one_hot_encode_cdr3(cdr3_sequence)
    encode_output.append(encoded_cdr3)
