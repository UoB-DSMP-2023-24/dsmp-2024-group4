file_path = 'vdjdb.txt'

import pandas as pd

df = pd.read_csv(file_path, sep='\t')

csv_file_path = 'vdjdb.csv'
df.to_csv(csv_file_path, index=False)



