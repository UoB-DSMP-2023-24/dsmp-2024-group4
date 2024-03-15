import numpy as np
def seqs2mat(seqs, alphabet='IRQCYMLVAFNESHKWGDTP', max_len=None):
    if max_len is None:
        max_len = np.max([len(s) for s in seqs])
    mat = -1 * np.ones((len(seqs), max_len), dtype=np.int8)
    L = np.zeros(len(seqs), dtype=np.int8)
    for si, s in enumerate(seqs):
        L[si] = len(s)
        for aai in range(max_len):
            if aai >= len(s):
                break
            mat[si, aai] = alphabet.index(s[aai])
    return mat, L

