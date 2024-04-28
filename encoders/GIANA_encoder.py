from encoders.GIANA.GIANA4_1 import EncodingCDR3,M6,n0
import numpy as np

def GIANA_encoder(TCRs, ST=3):
    cdr3_alpha_encoded = None
    cdr3_beta_encoded = None
    if TCRs[0].cdr3_alpha is not None:
        cdr3_alpha = [tcr.cdr3_alpha for tcr in TCRs]
        cdr3_alpha_encoded = [EncodingCDR3(seq[ST:-2], M6, n0) for seq in cdr3_alpha]
        cdr3_alpha_array = np.array(cdr3_alpha_encoded)

    if TCRs[0].cdr3_beta is not None:
        cdr3_beta = [tcr.cdr3_beta for tcr in TCRs]
        cdr3_beta_encoded = [EncodingCDR3(seq[ST:-2], M6, n0) for seq in cdr3_beta]
        cdr3_beta_array = np.array(cdr3_beta_encoded)
    if cdr3_alpha_encoded is not None and cdr3_beta_encoded is not None:
        return np.concatenate((cdr3_alpha_array, cdr3_beta_array), axis=1) # concatenate alpha and beta chain encodings
    elif cdr3_alpha_encoded is not None:
        return cdr3_alpha_encoded
    elif cdr3_beta_encoded is not None:
        return cdr3_beta_encoded

def GIANA_encoder_pd(df, chains=['alpha', 'beta'],ST=3):
    if chains== ['alpha', 'beta']:
        cdr3_alpha = df['cdr3_a_aa'].tolist()
        cdr3_beta = df['cdr3_b_aa'].tolist()
        cdr3_alpha_encoded = [EncodingCDR3(seq[ST:-2], M6, n0) for seq in cdr3_alpha]
        cdr3_beta_encoded = [EncodingCDR3(seq[ST:-2], M6, n0) for seq in cdr3_beta]
        cdr3_alpha_array = np.array(cdr3_alpha_encoded)
        cdr3_beta_array = np.array(cdr3_beta_encoded)
        cdr3_output=np.concatenate((cdr3_alpha_array, cdr3_beta_array), axis=1).tolist()
        df.loc[:, 'encoded_cdr3'] = cdr3_output
    elif chains== ['alpha']:
        cdr3_alpha = df['cdr3_a_aa'].tolist()
        cdr3_alpha_encoded = [EncodingCDR3(seq[ST:-2], M6, n0) for seq in cdr3_alpha]
        df.loc[:, 'encoded_cdr3'] = cdr3_alpha_encoded
    elif chains== ['beta']:
        cdr3_beta = df['cdr3_b_aa'].tolist()
        cdr3_beta_encoded = [EncodingCDR3(seq[ST:-2], M6, n0) for seq in cdr3_beta]
        df.loc[:, 'encoded_cdr3'] = cdr3_beta_encoded
    return df

# from encoders.GIANA_encoder import GIANA_encoder_pd
# human = GIADA_encoder_pd(human, chains=['alpha', 'beta'])
# it will add a new column 'encoded_cdr3' to the dataframe, which contains the encoded cdr3 sequences.


import numpy as np
import pandas as pd
df = pd.read_csv('../pre-processing final/cdr3_alpha_beta_df.csv')
# df = df[df['species'] == 'HomoSapiens']
df = df[df['species'] == 'MusMusculus']
df = GIANA_encoder_pd(df, chains=['alpha', 'beta'])
print(df)

