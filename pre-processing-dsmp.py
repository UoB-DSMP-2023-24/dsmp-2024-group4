#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

vdjdb= pd.read_csv('/Users/shalomifernandes/Desktop/UOB/TB2/DSMP/vdjdb.csv')


# In[2]:


vdjdb


# In[3]:


#Creating a copy of the original vdjdb dataset for modifications
filtered_data = vdjdb.copy()


# In[4]:


# Assuming the dataset is already loaded into the variable `vdjdb`

# Filter the dataset to keep relevant columns
filtered_data = vdjdb[['complex.id','gene', 'cdr3', 'v.segm', 'j.segm', 'species', 'mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope']]
print(filtered_data)


# In[5]:


# Assuming the dataset is already loaded into the variable `vdjdb`

# Filter the dataset to keep relevant columns
filtered_data = vdjdb[['complex.id','gene', 'cdr3', 'v.segm', 'j.segm', 'species', 'mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope']]
print(filtered_data)


# In[6]:


filtered_data


# In[7]:


# Check for missing values in each column
missing_values = filtered_data.isnull().sum()
print("Missing values in each column:\n", missing_values)


# In[8]:


# Drop rows with any missing values
filtered_data.dropna(subset=['v.segm','j.segm'], inplace=True)


# In[9]:


# Check for missing values in each column
cleaned_data = filtered_data.isnull().sum()
cleaned_data


# In[10]:


# Check for duplicate rows
duplicates = filtered_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")


# In[11]:


# Dropping all duplicate rows
filtered_data.drop_duplicates(inplace=True)


# In[12]:


# Checking for duplicate rows
duplicates = filtered_data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")


# In[13]:


filtered_data


# In[14]:


# Creating two new DataFrames based on the 'gene' column values
cdr3_alpha_df = filtered_data[filtered_data['gene'] == 'TRA'].copy()
cdr3_beta_df = filtered_data[filtered_data['gene'] == 'TRB'].copy()

# Now you have:
# cdr3_alpha_df: containing rows where 'gene' is 'TRA'
# cdr3_beta_df: containing rows where 'gene' is 'TRB'

# You can inspect the first few rows of each DataFrame to ensure it's what you expected
print("Alpha Chains:")
print(cdr3_alpha_df.head())

print("\nBeta Chains:")
print(cdr3_beta_df.head())


# In[15]:


filtered_data


# In[ ]:





# In[17]:


# Dropping rows where 'complex.id' is 0
#cdr3_alpha_beta_df = cdr3_alpha_beta_df[cdr3_alpha_beta_df['complex.id'] != 0]


# In[ ]:


# Ensure that the DataFrame does not contain rows where 'complex.id' is 0
#cdr3_alpha_beta_df = filtered_data.copy()
cdr3_alpha_beta_df = filtered_data[filtered_data['complex.id'] != 0]

# Rename columns in preparation for merging
cdr3_alpha_df_renamed = cdr3_alpha_df.rename(columns={
    'cdr3': 'cdr3_alpha',
    'v.segm': 'v.segm_alpha',
    'j.segm': 'j.segm_alpha'
})

cdr3_beta_df_renamed = cdr3_beta_df.rename(columns={
    'cdr3': 'cdr3_beta',
    'v.segm': 'v.segm_beta',
    'j.segm': 'j.segm_beta'
})

# Drop the 'gene' column as it's no longer necessary
cdr3_alpha_df_renamed.drop('gene', axis=1, inplace=True)
cdr3_beta_df_renamed.drop('gene', axis=1, inplace=True)

# Merge the dataframes on 'complex.id'
cdr3_alpha_beta_df = pd.merge(cdr3_alpha_df_renamed, cdr3_beta_df_renamed, on='complex.id')

# Keep only the required columns
required_columns = [
    'complex.id', 'cdr3_alpha', 'cdr3_beta', 'v.segm_alpha', 'v.segm_beta', 
    'j.segm_alpha', 'j.segm_beta', 'species', 'mhc.a', 'mhc.b', 'mhc.class', 'antigen.epitope'
]
cdr3_alpha_beta_df = cdr3_alpha_beta_df[required_columns]


# In[ ]:


# Output the head of the resulting dataframe
cdr3_alpha_beta_df.head()


# In[ ]:




