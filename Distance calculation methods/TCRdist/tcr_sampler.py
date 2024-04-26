import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sampler(df, n_samples=2000, n_epitopes=10,random_state=42):

    selected_epitopes = df['antigen.epitope'].drop_duplicates().sample(n=n_epitopes, random_state=random_state)
    filtered_data = df[df['antigen.epitope'].isin(selected_epitopes)]
    samples_per_epitope = n_samples // n_epitopes
    sampled_data = filtered_data.groupby('antigen.epitope').sample(n=samples_per_epitope, replace=True, random_state=random_state)

    return sampled_data

def remove_imbalance(df,threshold=10):
    epitope_num = df['antigen.epitope'].value_counts()
    epitope_num = epitope_num[epitope_num > threshold]
    df = df[df['antigen.epitope'].isin(epitope_num.index)]
    return df

def transform_imbalance(df,threshold=10):
    epitope_num = df['antigen.epitope'].value_counts()
    epitope_num = epitope_num[epitope_num > threshold]
    df['antigen.epitope'] = df['antigen.epitope'].apply(lambda x: x if x in epitope_num.index else 'other')
    print('Number of other: ',len(df[df['antigen.epitope'] == 'other']))
    return df

def main():
    df = pd.read_csv('vdjdb.csv')
    sampled_df = sampler(df)
    print(sampled_df)

if __name__ == "__main__":
    main()