import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sampler(df, n_samples=2000, n_epitopes=10,random_state=42):

    selected_epitopes = df['antigen.epitope'].drop_duplicates().sample(n=n_epitopes, random_state=random_state)
    filtered_data = df[df['antigen.epitope'].isin(selected_epitopes)]
    samples_per_epitope = n_samples // n_epitopes
    sampled_data = filtered_data.groupby('antigen.epitope').sample(n=samples_per_epitope, replace=True, random_state=random_state)

    return sampled_data

def main():
    df = pd.read_csv('vdjdb.csv')
    sampled_df = sampler(df)
    print(sampled_df)

if __name__ == "__main__":
    main()