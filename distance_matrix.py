from scipy.sparse import csr_matrix
import numpy as np
from encoders.one_hot_encode import one_hot_encode_cdr3
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('vdjdb.csv', header=None)
cdr3 = df[2].tolist()
epitope = df[9].tolist()

cdr3.pop(0)
epitope.pop(0)

encode_output = []
for i in cdr3:
    encode_output.append(one_hot_encode_cdr3(i,20))


cdr3_flattened = [np.array(seq).flatten() for seq in encode_output]

num_samples = len(cdr3_flattened)
cdr3_flattened = np.array(cdr3_flattened)


filename = 'distance_matrix.memmap'
fp = np.memmap(filename, dtype=np.float32, mode='w+', shape=(num_samples,num_samples))

fp[:,:]= np.sqrt(np.sum((cdr3_flattened[:, np.newaxis, :] - cdr3_flattened[np.newaxis, :, :]) ** 2, axis=2))

fp.flush()

