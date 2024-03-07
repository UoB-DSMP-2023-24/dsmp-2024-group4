import numpy as np
import pandas as pd
def one_hot_encode_cdr3(cdr3_sequence, max_length=20):
    standard_protein_letters = "IRQCYMLVAFNESHKWGDTP"
    encoding = np.zeros((len(cdr3_sequence), len(standard_protein_letters)))
    for i, amino_acid in enumerate(cdr3_sequence):
        if amino_acid in standard_protein_letters:
            encoding[i, standard_protein_letters.index(amino_acid)] = 1
    if len(cdr3_sequence) < max_length:
        pad = np.zeros((max_length - len(cdr3_sequence), len(standard_protein_letters)))
        encoding = np.concatenate((encoding, pad), axis=0)
    elif len(cdr3_sequence) > max_length:
        encoding = encoding[:max_length, :]
    return encoding
def main():
    standard_protein_letters = "IRQCYMLVAFNESHKWGDTP"
    df = pd.read_csv('../vdjdb.csv', header=None)
    cdr3=df[2].tolist()
    cdr3.pop(0)
    encode_output= []
    for cdr3_sequence in cdr3:
        encoded_cdr3 = one_hot_encode_cdr3(cdr3_sequence)
        encode_output.append(encoded_cdr3)

if __name__ == "__main__":
    main()
