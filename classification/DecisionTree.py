from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report
import pandas as pd
from tcr_sampler import sampler
import matplotlib.pyplot as plt

def split_cdr3(cdr3_sequence):
    return list(cdr3_sequence)

df = pd.read_csv('../vdjdb.csv')
# df = sampler(df, n_samples=30000, n_epitopes=500)
df = df[['cdr3', 'antigen.epitope']]
split_sequences = df['cdr3'].apply(split_cdr3)

split_df = pd.DataFrame(split_sequences.tolist())
split_df = split_df.fillna('NA')
alphabet = 'IRQCYMLVAFNESHKWGDTP'
split_df = split_df.applymap(lambda x: alphabet.index(x) if x in alphabet else 20)
X = split_df
y = df['antigen.epitope']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

print(classification_report(y_test, y_pred))

'''
plot_tree(dt_classifier, filled=True)
plt.savefig('decision_tree.png')
'''