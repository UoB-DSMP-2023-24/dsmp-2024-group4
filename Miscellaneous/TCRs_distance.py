import numpy as np
import itertools
from Bio.Align import substitution_matrices
import pandas as pd
import numba as nb
from encoders.ordinal_encode import seqs2mat
from tqdm import tqdm
from tcrdist.repertoire import TCRrep

'''
ctrim - number of amino acids to trim of c-terminal end of a CDR. (i.e. ctrim = 3 CASSQDFEQ-YF consider only positions after CAS-SQDFEQ-YF)
ntrim - number of amino acids to trim of n-terminal end of a CDR. (i.e. ntrim = 2 CASSQDFEQ-YF consider only positions before “YF”, SQDFEQ-YF)
gap_penalty - the penalty accrued for gaps in the pairwise alignment.
‘fixed_gappos’ - When sequences are of different length, there will be a gap position. If ‘fixed_gappos’ is False, then the metric inserts a single gap at an optimal position based on a BLOSUM62 scoring matrix. This is recommended for the CDR3, but it is not necessary when the CDR1, 2, and 2.5 are already imgt_aligned and of a fixed length.
'''

class TCR:
    def __init__(self, cdr3_alpha=None, cdr3_beta=None,v_segm_alpha=None,v_segm_beta=None,j_segm_alpha=None,j_segm_beta=None,mhc_a=None,mhc_b=None, epitope=None,weight=None):
        self.cdr3_alpha = cdr3_alpha
        self.cdr3_beta = cdr3_beta
        self.v_segm_alpha=v_segm_alpha
        self.v_segm_beta=v_segm_beta
        self.j_segm_alpha=j_segm_alpha
        self.j_segm_beta=j_segm_beta
        self.mhc_a = mhc_a
        self.mhc_b = mhc_b
        self.epitope = epitope


def matrix_position(size, i, j):
    small = min(i, j)+1
    big = max(i, j)+1
    if small == 1:
        return int(big-2)
    return int(size * (small-1) - small * (small - 1) / 2 + big - small - 1)

def distance_cal(TCRs,alphabet='IRQCYMLVAFNESHKWGDTP'):
    tcr_dict_distance_matrix = {('A', 'A'): 0, ('A', 'C'): 4, ('A', 'D'): 4, ('A', 'E'): 4, ('A', 'F'): 4,
                                ('A', 'G'): 4, ('A', 'H'): 4, ('A', 'I'): 4, ('A', 'K'): 4, ('A', 'L'): 4,
                                ('A', 'M'): 4, ('A', 'N'): 4, ('A', 'P'): 4, ('A', 'Q'): 4, ('A', 'R'): 4,
                                ('A', 'S'): 3, ('A', 'T'): 4, ('A', 'V'): 4, ('A', 'W'): 4, ('A', 'Y'): 4,
                                ('C', 'A'): 4, ('C', 'C'): 0, ('C', 'D'): 4, ('C', 'E'): 4, ('C', 'F'): 4,
                                ('C', 'G'): 4, ('C', 'H'): 4, ('C', 'I'): 4, ('C', 'K'): 4, ('C', 'L'): 4,
                                ('C', 'M'): 4, ('C', 'N'): 4, ('C', 'P'): 4, ('C', 'Q'): 4, ('C', 'R'): 4,
                                ('C', 'S'): 4, ('C', 'T'): 4, ('C', 'V'): 4, ('C', 'W'): 4, ('C', 'Y'): 4,
                                ('D', 'A'): 4, ('D', 'C'): 4, ('D', 'D'): 0, ('D', 'E'): 2, ('D', 'F'): 4,
                                ('D', 'G'): 4, ('D', 'H'): 4, ('D', 'I'): 4, ('D', 'K'): 4, ('D', 'L'): 4,
                                ('D', 'M'): 4, ('D', 'N'): 3, ('D', 'P'): 4, ('D', 'Q'): 4, ('D', 'R'): 4,
                                ('D', 'S'): 4, ('D', 'T'): 4, ('D', 'V'): 4, ('D', 'W'): 4, ('D', 'Y'): 4,
                                ('E', 'A'): 4, ('E', 'C'): 4, ('E', 'D'): 2, ('E', 'E'): 0, ('E', 'F'): 4,
                                ('E', 'G'): 4, ('E', 'H'): 4, ('E', 'I'): 4, ('E', 'K'): 3, ('E', 'L'): 4,
                                ('E', 'M'): 4, ('E', 'N'): 4, ('E', 'P'): 4, ('E', 'Q'): 2, ('E', 'R'): 4,
                                ('E', 'S'): 4, ('E', 'T'): 4, ('E', 'V'): 4, ('E', 'W'): 4, ('E', 'Y'): 4,
                                ('F', 'A'): 4, ('F', 'C'): 4, ('F', 'D'): 4, ('F', 'E'): 4, ('F', 'F'): 0,
                                ('F', 'G'): 4, ('F', 'H'): 4, ('F', 'I'): 4, ('F', 'K'): 4, ('F', 'L'): 4,
                                ('F', 'M'): 4, ('F', 'N'): 4, ('F', 'P'): 4, ('F', 'Q'): 4, ('F', 'R'): 4,
                                ('F', 'S'): 4, ('F', 'T'): 4, ('F', 'V'): 4, ('F', 'W'): 3, ('F', 'Y'): 1,
                                ('G', 'A'): 4, ('G', 'C'): 4, ('G', 'D'): 4, ('G', 'E'): 4, ('G', 'F'): 4,
                                ('G', 'G'): 0, ('G', 'H'): 4, ('G', 'I'): 4, ('G', 'K'): 4, ('G', 'L'): 4,
                                ('G', 'M'): 4, ('G', 'N'): 4, ('G', 'P'): 4, ('G', 'Q'): 4, ('G', 'R'): 4,
                                ('G', 'S'): 4, ('G', 'T'): 4, ('G', 'V'): 4, ('G', 'W'): 4, ('G', 'Y'): 4,
                                ('H', 'A'): 4, ('H', 'C'): 4, ('H', 'D'): 4, ('H', 'E'): 4, ('H', 'F'): 4,
                                ('H', 'G'): 4, ('H', 'H'): 0, ('H', 'I'): 4, ('H', 'K'): 4, ('H', 'L'): 4,
                                ('H', 'M'): 4, ('H', 'N'): 3, ('H', 'P'): 4, ('H', 'Q'): 4, ('H', 'R'): 4,
                                ('H', 'S'): 4, ('H', 'T'): 4, ('H', 'V'): 4, ('H', 'W'): 4, ('H', 'Y'): 2,
                                ('I', 'A'): 4, ('I', 'C'): 4, ('I', 'D'): 4, ('I', 'E'): 4, ('I', 'F'): 4,
                                ('I', 'G'): 4, ('I', 'H'): 4, ('I', 'I'): 0, ('I', 'K'): 4, ('I', 'L'): 2,
                                ('I', 'M'): 3, ('I', 'N'): 4, ('I', 'P'): 4, ('I', 'Q'): 4, ('I', 'R'): 4,
                                ('I', 'S'): 4, ('I', 'T'): 4, ('I', 'V'): 1, ('I', 'W'): 4, ('I', 'Y'): 4,
                                ('K', 'A'): 4, ('K', 'C'): 4, ('K', 'D'): 4, ('K', 'E'): 3, ('K', 'F'): 4,
                                ('K', 'G'): 4, ('K', 'H'): 4, ('K', 'I'): 4, ('K', 'K'): 0, ('K', 'L'): 4,
                                ('K', 'M'): 4, ('K', 'N'): 4, ('K', 'P'): 4, ('K', 'Q'): 3, ('K', 'R'): 2,
                                ('K', 'S'): 4, ('K', 'T'): 4, ('K', 'V'): 4, ('K', 'W'): 4, ('K', 'Y'): 4,
                                ('L', 'A'): 4, ('L', 'C'): 4, ('L', 'D'): 4, ('L', 'E'): 4, ('L', 'F'): 4,
                                ('L', 'G'): 4, ('L', 'H'): 4, ('L', 'I'): 2, ('L', 'K'): 4, ('L', 'L'): 0,
                                ('L', 'M'): 2, ('L', 'N'): 4, ('L', 'P'): 4, ('L', 'Q'): 4, ('L', 'R'): 4,
                                ('L', 'S'): 4, ('L', 'T'): 4, ('L', 'V'): 3, ('L', 'W'): 4, ('L', 'Y'): 4,
                                ('M', 'A'): 4, ('M', 'C'): 4, ('M', 'D'): 4, ('M', 'E'): 4, ('M', 'F'): 4,
                                ('M', 'G'): 4, ('M', 'H'): 4, ('M', 'I'): 3, ('M', 'K'): 4, ('M', 'L'): 2,
                                ('M', 'M'): 0, ('M', 'N'): 4, ('M', 'P'): 4, ('M', 'Q'): 4, ('M', 'R'): 4,
                                ('M', 'S'): 4, ('M', 'T'): 4, ('M', 'V'): 3, ('M', 'W'): 4, ('M', 'Y'): 4,
                                ('N', 'A'): 4, ('N', 'C'): 4, ('N', 'D'): 3, ('N', 'E'): 4, ('N', 'F'): 4,
                                ('N', 'G'): 4, ('N', 'H'): 3, ('N', 'I'): 4, ('N', 'K'): 4, ('N', 'L'): 4,
                                ('N', 'M'): 4, ('N', 'N'): 0, ('N', 'P'): 4, ('N', 'Q'): 4, ('N', 'R'): 4,
                                ('N', 'S'): 3, ('N', 'T'): 4, ('N', 'V'): 4, ('N', 'W'): 4, ('N', 'Y'): 4,
                                ('P', 'A'): 4, ('P', 'C'): 4, ('P', 'D'): 4, ('P', 'E'): 4, ('P', 'F'): 4,
                                ('P', 'G'): 4, ('P', 'H'): 4, ('P', 'I'): 4, ('P', 'K'): 4, ('P', 'L'): 4,
                                ('P', 'M'): 4, ('P', 'N'): 4, ('P', 'P'): 0, ('P', 'Q'): 4, ('P', 'R'): 4,
                                ('P', 'S'): 4, ('P', 'T'): 4, ('P', 'V'): 4, ('P', 'W'): 4, ('P', 'Y'): 4,
                                ('Q', 'A'): 4, ('Q', 'C'): 4, ('Q', 'D'): 4, ('Q', 'E'): 2, ('Q', 'F'): 4,
                                ('Q', 'G'): 4, ('Q', 'H'): 4, ('Q', 'I'): 4, ('Q', 'K'): 3, ('Q', 'L'): 4,
                                ('Q', 'M'): 4, ('Q', 'N'): 4, ('Q', 'P'): 4, ('Q', 'Q'): 0, ('Q', 'R'): 3,
                                ('Q', 'S'): 4, ('Q', 'T'): 4, ('Q', 'V'): 4, ('Q', 'W'): 4, ('Q', 'Y'): 4,
                                ('R', 'A'): 4, ('R', 'C'): 4, ('R', 'D'): 4, ('R', 'E'): 4, ('R', 'F'): 4,
                                ('R', 'G'): 4, ('R', 'H'): 4, ('R', 'I'): 4, ('R', 'K'): 2, ('R', 'L'): 4,
                                ('R', 'M'): 4, ('R', 'N'): 4, ('R', 'P'): 4, ('R', 'Q'): 3, ('R', 'R'): 0,
                                ('R', 'S'): 4, ('R', 'T'): 4, ('R', 'V'): 4, ('R', 'W'): 4, ('R', 'Y'): 4,
                                ('S', 'A'): 3, ('S', 'C'): 4, ('S', 'D'): 4, ('S', 'E'): 4, ('S', 'F'): 4,
                                ('S', 'G'): 4, ('S', 'H'): 4, ('S', 'I'): 4, ('S', 'K'): 4, ('S', 'L'): 4,
                                ('S', 'M'): 4, ('S', 'N'): 3, ('S', 'P'): 4, ('S', 'Q'): 4, ('S', 'R'): 4,
                                ('S', 'S'): 0, ('S', 'T'): 3, ('S', 'V'): 4, ('S', 'W'): 4, ('S', 'Y'): 4,
                                ('T', 'A'): 4, ('T', 'C'): 4, ('T', 'D'): 4, ('T', 'E'): 4, ('T', 'F'): 4,
                                ('T', 'G'): 4, ('T', 'H'): 4, ('T', 'I'): 4, ('T', 'K'): 4, ('T', 'L'): 4,
                                ('T', 'M'): 4, ('T', 'N'): 4, ('T', 'P'): 4, ('T', 'Q'): 4, ('T', 'R'): 4,
                                ('T', 'S'): 3, ('T', 'T'): 0, ('T', 'V'): 4, ('T', 'W'): 4, ('T', 'Y'): 4,
                                ('V', 'A'): 4, ('V', 'C'): 4, ('V', 'D'): 4, ('V', 'E'): 4, ('V', 'F'): 4,
                                ('V', 'G'): 4, ('V', 'H'): 4, ('V', 'I'): 1, ('V', 'K'): 4, ('V', 'L'): 3,
                                ('V', 'M'): 3, ('V', 'N'): 4, ('V', 'P'): 4, ('V', 'Q'): 4, ('V', 'R'): 4,
                                ('V', 'S'): 4, ('V', 'T'): 4, ('V', 'V'): 0, ('V', 'W'): 4, ('V', 'Y'): 4,
                                ('W', 'A'): 4, ('W', 'C'): 4, ('W', 'D'): 4, ('W', 'E'): 4, ('W', 'F'): 3,
                                ('W', 'G'): 4, ('W', 'H'): 4, ('W', 'I'): 4, ('W', 'K'): 4, ('W', 'L'): 4,
                                ('W', 'M'): 4, ('W', 'N'): 4, ('W', 'P'): 4, ('W', 'Q'): 4, ('W', 'R'): 4,
                                ('W', 'S'): 4, ('W', 'T'): 4, ('W', 'V'): 4, ('W', 'W'): 0, ('W', 'Y'): 2,
                                ('Y', 'A'): 4, ('Y', 'C'): 4, ('Y', 'D'): 4, ('Y', 'E'): 4, ('Y', 'F'): 1,
                                ('Y', 'G'): 4, ('Y', 'H'): 2, ('Y', 'I'): 4, ('Y', 'K'): 4, ('Y', 'L'): 4,
                                ('Y', 'M'): 4, ('Y', 'N'): 4, ('Y', 'P'): 4, ('Y', 'Q'): 4, ('Y', 'R'): 4,
                                ('Y', 'S'): 4, ('Y', 'T'): 4, ('Y', 'V'): 4, ('Y', 'W'): 2, ('Y', 'Y'): 0}
    indices = list(itertools.combinations(range(len(TCRs)), 2))
    dm = np.zeros((len(alphabet), len(alphabet)), dtype=np.int32)
    for (aa1, aa2), d in tcr_dict_distance_matrix.items():
        dm[alphabet.index(aa1), alphabet.index(aa2)] = d
        dm[alphabet.index(aa2), alphabet.index(aa1)] = d
    indices = np.array(indices, dtype=np.int64)
    distance=np.zeros(indices.shape[0], dtype=np.int16)
    if TCRs[0].cdr3_alpha is not None:
        cdr3_alpha = [tcr.cdr3_alpha for tcr in TCRs]
        seqs_mat, seqs_L = seqs2mat(cdr3_alpha)
        dist_alpha = _distance_cal(indices, seqs_mat, seqs_L, dm, dist_weight=3, gap_penalty=4, ntrim=3, ctrim=2,
                                   fixed_gappos=True)
        distance=distance+dist_alpha
    if TCRs[0].cdr3_beta is not None:
        cdr3_beta = [tcr.cdr3_beta for tcr in TCRs]
        seqs_mat, seqs_L = seqs2mat(cdr3_beta)
        dist_beta = _distance_cal(indices, seqs_mat, seqs_L, dm, dist_weight=3, gap_penalty=4, ntrim=3, ctrim=2,
                                  fixed_gappos=True)
        distance=distance+dist_beta
    if TCRs[0].v_segm_alpha is not None:
        v_segm_alpha = [tcr.v_segm_alpha for tcr in TCRs]
        dist_vsegm_alpha = distance_categorical(v_segm_alpha, 3)
        distance = distance + dist_vsegm_alpha
    if TCRs[0].v_segm_beta is not None:
        v_segm_beta = [tcr.v_segm_beta for tcr in TCRs]
        dist_vsegm_beta = distance_categorical(v_segm_beta, 3)
        distance = distance + dist_vsegm_beta
    if TCRs[0].j_segm_alpha is not None:
        j_segm_alpha = [tcr.j_segm_alpha for tcr in TCRs]
        dist_jsegm_alpha = distance_categorical(j_segm_alpha, 3)
        distance = distance + dist_jsegm_alpha
    if TCRs[0].j_segm_beta is not None:
        j_segm_beta = [tcr.j_segm_beta for tcr in TCRs]
        dist_jsegm_beta = distance_categorical(j_segm_beta, 3)
        distance = distance + dist_jsegm_beta
    if TCRs[0].mhc_a is not None:
        mhc_a = [tcr.mhc_a for tcr in TCRs]
        dist_mhc_a = distance_categorical(mhc_a, 3)
        distance = distance + dist_mhc_a
    if TCRs[0].mhc_b is not None:
        mhc_b = [tcr.mhc_b for tcr in TCRs]
        dist_mhc_b = distance_categorical(mhc_b, 3)
        distance = distance + dist_mhc_b


    return distance,indices

def distance_categorical(TCRs,weight):
    indices = list(itertools.combinations(range(len(TCRs)), 2))
    dist = np.zeros(len(indices), dtype=np.int16)
    for i, (i1, i2) in enumerate(indices):
        dist[i] = weight * (TCRs[i1]!= TCRs[i2])
    return dist
def _distance_cal(indices, seqs_mat, seqs_L, distance_matrix, dist_weight=3, gap_penalty=4, ntrim=3, ctrim=2, fixed_gappos=True):
    """
    indices : np.ndarray [nseqs, 2]
        Indices into seqs_mat indicating pairs of sequences to compare.
    seqs_mat : np.ndarray dtype=int16 [nseqs, seq_length]
        Created by pwsd.seqs2mat with padding to accomodate
        sequences of different lengths (-1 padding)
    seqs_L : np.ndarray [nseqs]
        A vector containing the length of each sequence,
        without the padding in seqs_mat
    distance_matrix : np.ndarray [alphabet, alphabet] dtype=int32
        A square distance matrix (NOT a similarity matrix).
        Matrix must match the alphabet that was used to create
        seqs_mat, where each AA is represented by an index into the alphabet.
    dist_weight : int
        Weight applied to the mismatch distances before summing with the gap penalties
    gap_penalty : int
        Distance penalty for the difference in the length of the two sequences
    ntrim/ctrim : int
        Positions trimmed off the N-terminus (0) and C-terminus (L-1) ends of the peptide sequence. These symbols will be ignored
        in the distance calculation.
    fixed_gappos : bool
        If True, insert gaps at a fixed position after the cysteine residue statring the CDR3 (typically position 6).
        If False, find the "optimal" position for inserting the gaps to make up the difference in length
    """

    dist = np.zeros(indices.shape[0], dtype=np.int16)
    for ind_i in tqdm(nb.prange(indices.shape[0])):
        query_i = indices[ind_i, 0]
        seq_i = indices[ind_i, 1]
        q_L = seqs_L[query_i]
        s_L = seqs_L[seq_i]
        if q_L == s_L:
            """No gaps: substitution distance"""
            for i in range(ntrim, q_L - ctrim):
                dist[ind_i] += distance_matrix[seqs_mat[query_i, i], seqs_mat[seq_i, i]] * dist_weight
            continue

        short_len = min(q_L, s_L)
        len_diff = abs(q_L - s_L)
        if fixed_gappos:
            min_gappos = min(6, 3 + (short_len - 5) // 2)
            max_gappos = min_gappos
        else:
            min_gappos = 5
            max_gappos = short_len - 1 - 4
            while min_gappos > max_gappos:
                min_gappos -= 1
                max_gappos += 1
        min_dist = -1
        # min_count = -1
        # A "gap" is a blank position introduced when comparing two or more biological sequences (e.g. DNA, RNA or protein sequences) in order to maximise alignment and similarity between sequences.
        # When sequences are not of the same length, the similarity of two sequences can be compared more accurately by introducing gaps in the shorter sequence. The gappos variable in the function seems to be used to determine the best place to introduce gaps in the sequences.

        for gappos in range(min_gappos, max_gappos + 1):
            tmp_dist = 0
            # tmp_count = 0
            remainder = short_len - gappos
            for n_i in range(ntrim, gappos):
                """n_i refers to position relative to N term"""
                # print (n_i, shortseq[i], longseq[i], distance_matrix[shortseq[i]+longseq[i]])
                tmp_dist += distance_matrix[seqs_mat[query_i, n_i], seqs_mat[seq_i, n_i]]
                # tmp_count += 1
            #print('sequence_distance_with_gappos1:', gappos, remainder, dist[seq_i])
            for c_i in range(ctrim, remainder):
                """c_i refers to position relative to C term, counting upwards from C term"""
                tmp_dist += distance_matrix[seqs_mat[query_i, q_L - 1 - c_i], seqs_mat[seq_i, s_L - 1 - c_i]]
                # tmp_count += 1
            #print('sequence_distance_with_gappos2:', gappos, remainder, dist[seq_i])
            # Find the gap position that minimises the distance
            if tmp_dist < min_dist or min_dist == -1:
                min_dist = tmp_dist
                # min_count = tmp_count
            if min_dist == 0:
                break
        dist[ind_i] = min_dist * dist_weight + len_diff * gap_penalty
    return dist

def dist_to_matrix(dist, indices, nseqs):
    """Convert a distance vector to a distance matrix"""
    dist_matrix = np.zeros((nseqs, nseqs), dtype=np.int16)
    for i in range(indices.shape[0]):
        dist_matrix[indices[i, 0], indices[i, 1]] = dist[i]
        dist_matrix[indices[i, 1], indices[i, 0]] = dist[i]
    return dist_matrix



def main():

    df = pd.read_csv('pre-processing final/cdr3_alpha_beta_df.csv')
    human_df = df[df['species'] == 'HomoSapiens']
    mouse_df = df[df['species'] == 'MusMusculus']
    # TCRs = [TCR(cdr3_alpha[i], cdr3_beta[i], v_segm_alpha[i], v_segm_beta[i], j_segm_alpha[i], j_segm_beta[i], mhc_a[i], mhc_b[i], epitope[i]) for i in range(len(cdr3_alpha))]
    # TCRs = [TCR(cdr3_alpha[i], None, v_segm_alpha[i], None, j_segm_alpha[i], None, mhc_a[i], None, epitope[i]) for i in range(num_tcrs)]
    # TCRs = [TCR(None,cdr3_beta[i],None,v_segm_beta[i],None,j_segm_beta[i],None,mhc_b[i],epitope[i]) for i in range(num_tcrs)
    cdr3_alpha_human = human_df['cdr3_a_aa'].tolist()
    cdr3_beta_human = human_df['cdr3_b_aa'].tolist()
    v_segm_alpha_human = human_df['v_a_gene'].tolist()
    v_segm_beta_human = human_df['v_b_gene'].tolist()
    j_segm_alpha_human = human_df['j_a_gene'].tolist()
    j_segm_beta_human = human_df['j_b_gene'].tolist()
    mhc_a_human = human_df['mhc.a'].tolist()
    mhc_b_human = human_df['mhc.b'].tolist()
    epitope_human = human_df['epitope'].tolist()
    cdr3_alpha_mouse = mouse_df['cdr3_a_aa'].tolist()
    cdr3_beta_mouse = mouse_df['cdr3_b_aa'].tolist()
    v_segm_alpha_mouse = mouse_df['v_a_gene'].tolist()
    v_segm_beta_mouse = mouse_df['v_b_gene'].tolist()
    j_segm_alpha_mouse = mouse_df['j_a_gene'].tolist()
    j_segm_beta_mouse = mouse_df['j_b_gene'].tolist()
    mhc_a_mouse = mouse_df['mhc.a'].tolist()
    mhc_b_mouse = mouse_df['mhc.b'].tolist()
    epitope_mouse = mouse_df['epitope'].tolist()

    human_df_alpha_beta_TCRs = [TCR(cdr3_alpha_human[i], cdr3_beta_human[i], v_segm_alpha_human[i], v_segm_beta_human[i], j_segm_alpha_human[i], j_segm_beta_human[i], mhc_a_human[i], mhc_b_human[i], epitope_human[i]) for i in range(len(cdr3_alpha_human))]
    mouse_df_alpha_beta_TCRs = [TCR(cdr3_alpha_mouse[i], cdr3_beta_mouse[i], v_segm_alpha_mouse[i], v_segm_beta_mouse[i], j_segm_alpha_mouse[i], j_segm_beta_mouse[i], mhc_a_mouse[i], mhc_b_mouse[i], epitope_mouse[i]) for i in range(len(cdr3_alpha_mouse))]
    human_df_alpha_TCRs = [TCR(cdr3_alpha_human[i], None, v_segm_alpha_human[i], None, j_segm_alpha_human[i], None, mhc_a_human[i], None, epitope_human[i]) for i in range(len(cdr3_alpha_human))]
    mouse_df_alpha_TCRs = [TCR(cdr3_alpha_mouse[i], None, v_segm_alpha_mouse[i], None, j_segm_alpha_mouse[i], None, mhc_a_mouse[i], None, epitope_mouse[i]) for i in range(len(cdr3_alpha_mouse))]
    human_df_beta_TCRs = [TCR(None, cdr3_beta_human[i], None, v_segm_beta_human[i], None, j_segm_beta_human[i], None, mhc_b_human[i], epitope_human[i]) for i in range(len(cdr3_beta_human))]
    mouse_df_beta_TCRs = [TCR(None, cdr3_beta_mouse[i], None, v_segm_beta_mouse[i], None, j_segm_beta_mouse[i], None, mhc_b_mouse[i], epitope_mouse[i]) for i in range(len(cdr3_beta_mouse))]
    dist, indices = distance_cal(human_df_alpha_beta_TCRs)
    dist_matrix = dist_to_matrix(dist, indices, len(human_df_alpha_beta_TCRs))
    np.save('human_df_alpha_beta_TCRs_distance_matrix.npy', dist_matrix)
    dist, indices = distance_cal(mouse_df_alpha_beta_TCRs)
    dist_matrix = dist_to_matrix(dist, indices, len(mouse_df_alpha_beta_TCRs))
    np.save('mouse_df_alpha_beta_TCRs_distance_matrix.npy', dist_matrix)
    dist, indices = distance_cal(human_df_alpha_TCRs)
    dist_matrix = dist_to_matrix(dist, indices, len(human_df_alpha_TCRs))
    np.save('human_df_alpha_TCRs_distance_matrix.npy', dist_matrix)
    dist, indices = distance_cal(mouse_df_alpha_TCRs)
    dist_matrix = dist_to_matrix(dist, indices, len(mouse_df_alpha_TCRs))
    np.save('mouse_df_alpha_TCRs_distance_matrix.npy', dist_matrix)
    dist, indices = distance_cal(human_df_beta_TCRs)
    dist_matrix = dist_to_matrix(dist, indices, len(human_df_beta_TCRs))
    np.save('human_df_beta_TCRs_distance_matrix.npy', dist_matrix)
    dist, indices = distance_cal(mouse_df_beta_TCRs)
    dist_matrix = dist_to_matrix(dist, indices, len(mouse_df_beta_TCRs))
    np.save('mouse_df_beta_TCRs_distance_matrix.npy', dist_matrix)



    '''
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
    n_epitopes = len(set(epitope))

    # select 10 TCRs
    num_tcrs = 10

    # combine alpha and beta chain
    TCRs = [TCR(cdr3_alpha[i], cdr3_beta[i], v_segm_alpha[i], v_segm_beta[i], j_segm_alpha[i], j_segm_beta[i], mhc_a[i], mhc_b[i], epitope[i]) for i in range(num_tcrs)]
    dist, indices = distance_cal(TCRs)
    dist_matrix = dist_to_matrix(dist, indices, num_tcrs)
    print(dist_matrix)

    # only alpha chain
    TCRs = [TCR(cdr3_alpha[i], None, v_segm_alpha[i], None, j_segm_alpha[i], None, mhc_a[i], mhc_b[i], epitope[i]) for i in range(num_tcrs)]
    dist, indices = distance_cal(TCRs)
    dist_matrix = dist_to_matrix(dist, indices, num_tcrs)
    print(dist_matrix)

    # only beta chain
    TCRs = [TCR(None, cdr3_beta[i], None, v_segm_beta[i], None, j_segm_beta[i], mhc_a[i], mhc_b[i], epitope[i]) for i in range(num_tcrs)]
    dist, indices = distance_cal(TCRs)
    dist_matrix = dist_to_matrix(dist, indices, num_tcrs)
    print(dist_matrix)
    '''
if __name__ == "__main__":
    main()