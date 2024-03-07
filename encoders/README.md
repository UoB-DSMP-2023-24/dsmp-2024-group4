# Encoding Methods
This folder contains various encoding methods we have experimented with for encoding CDR3 sequences. The Complementarity-Determining Region 3 (CDR3) is a critical part of the T-cell receptor (TCR) and B-cell receptor (BCR) that is responsible for recognizing and binding to antigens. Encoding these sequences effectively is crucial for various immunoinformatics applications, including TCR/BCR repertoire analysis, antigen specificity prediction, and vaccine design.
## Biological Encoding
- File: encoders/bio_matrices_encode.py
-  This script utilizes the substitution matrices to encode CDR3 sequences. The BLOSUM90 substitution matrix provides scores for different amino acid pair substitutions. These scores are based on the frequency of amino acid substitutions in the protein sequence comparison. For each amino acid in the CDR3 sequence, the corresponding score in the BLOSUM90 matrix is used for encoding. Each amino acid was converted to a score vector that represents the substitution score of that amino acid with all other amino acids.
## One-Hot Encoding
- File: encoders/one_hot_encode.py
- This script encodes CDR3 sequences using one-hot encoding. In this method, each amino acid in the CDR3 sequence is represented as a binary vector. The length of the vector is equal to the number of unique amino acids in the sequence. The vector contains all zeros except for the position corresponding to the amino acid, which is set to 1.
## Comparing
![Comparison](knn_accuracy.png)