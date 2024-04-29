import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from tcr_sampler import sampler,remove_imbalance,transform_imbalance

def split_cdr3(cdr3_sequence):
    return list(cdr3_sequence)

df = pd.read_csv('../cdr3_alpha_beta.csv')
# df=sampler(df, n_samples=10000, n_epitopes=10)
# df=remove_imbalance(df,threshold=10)
# df=transform_imbalance(df,threshold=10)
# print(len(df['antigen.epitope'].value_counts()))
df=df[['cdr3_alpha','antigen.epitope']]

split_sequences = df['cdr3_alpha'].apply(split_cdr3)

split_df = pd.DataFrame(split_sequences.tolist())
split_df = split_df.fillna('NA')
alphabet='IRQCYMLVAFNESHKWGDTP'
split_df = split_df.applymap(lambda x: alphabet.index(x) if x in alphabet else 20)
X=split_df
y=df['antigen.epitope']
# Uniform division according to the type of epitope
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rf_classifier = RandomForestClassifier(n_estimators=80, random_state=42) 


rf_classifier.fit(X_train, y_train)


y_pred = rf_classifier.predict(X_test)

print(classification_report(y_test, y_pred))
