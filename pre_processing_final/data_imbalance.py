import pandas as pd
import numpy as np
# from tcr_sampler import remove_imbalance
import matplotlib.pyplot as plt
def bar_plot(df, column='antigen.epitope', title='Distribution of Epitopes', xlabel='Epitope', ylabel='Number of TCRs',show_x=False):
    plt.figure(figsize=(12, 6))
    data = df[column].value_counts()
    plt.bar(range(len(data)), data)
    plt.yscale('log')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlabel(xlabel)
    if show_x:
        plt.xticks(range(len(data)), data.index, rotation=90)
    plt.show()
# df=pd.read_csv('cdr3_alpha_beta_without_0score.csv')

# df=remove_imbalance(df,threshold=10)
# epitope_num=df['antigen.epitope'].value_counts()
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
'''
plt.bar(range(len(epitope_num)), epitope_num, color='skyblue')
plt.yscale('log')  # Continue using a logarithmic scale
plt.ylabel('Number of Entries (log scale)')
plt.title('Distribution of Entries by Type (Log Scale)')'''

'''
cumulative_percentage = np.cumsum(epitope_num) / epitope_num.sum()
plt.plot(cumulative_percentage, color='green')
plt.xlabel('Number of Types (Ranked)')
plt.ylabel('Cumulative Percentage of Entries')
plt.title('Cumulative Percentage of Entries by Type')

plt.tight_layout()
plt.show()
'''

# plt.pie(epitope_num)
# plt.show()
