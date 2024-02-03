import pandas as pd

# Load the provided data file
file_path = 'vdjdb.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()

# Analyze the data for each column
data_info = data.info()
data_description = data.describe(include='all')

# Check for missing values
missing_values = data.isnull().sum()

data_info, data_description, missing_values
