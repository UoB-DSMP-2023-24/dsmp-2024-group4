import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report
import pandas as pd
from tcr_sampler import sampler,remove_imbalance,transform_imbalance
import matplotlib.pyplot as plt


def split_cdr3(cdr3_sequence):
    return list(cdr3_sequence)

df = pd.read_csv('../vdjdb.csv')
# df = sampler(df, n_samples=30000, n_epitopes=500)
df=remove_imbalance(df,threshold=10)
# df=transform_imbalance(df,threshold=10)
df = df[['cdr3', 'antigen.epitope']]
split_sequences = df['cdr3'].apply(split_cdr3)

split_df = pd.DataFrame(split_sequences.tolist())
split_df = split_df.fillna('NA')
alphabet = 'IRQCYMLVAFNESHKWGDTP'
split_df = split_df.applymap(lambda x: alphabet.index(x) if x in alphabet else 20)
X = split_df
y = df['antigen.epitope']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 多次实验求平均
accuracy = []
# max_depths= [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# min_samples_splits = [2, 4, 6, 8, 10]

for i in range(10):
    dt_classifier = DecisionTreeClassifier(class_weight='balanced')
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
    accuracy.append(dt_classifier.score(X_test, y_test))
print('mean accuracy:', sum(accuracy) / len(accuracy))



# entropy 0.3808139534883722 gini 0.3810368526636452
# class_weight='balanced' 0.3303737276172078
# When dealing with unbalanced datasets, 'balanced' weights increase the relative importance of less common categories.
# This may cause the model to focus more on these minority categories and thus may make more errors on majority categories, especially if these majority categories dominate the dataset.
# In addition, the metric of accuracy may not always capture the effects of category imbalance.


'''
plot_tree(dt_classifier, filled=True)
plt.savefig('decision_tree.png')
'''