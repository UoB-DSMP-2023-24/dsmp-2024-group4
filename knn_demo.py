from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from encoders.bio_matrices_encode import encode_cdr3_list
from encoders.one_hot_encode import one_hot_encode_cdr3

df = pd.read_csv('vdjdb.csv', header=None)
cdr3 = df[2].tolist()
epitope = df[9].tolist()

cdr3.pop(0)
epitope.pop(0)

# encode_output = []
# for i in cdr3:
    # encode_output.append(one_hot_encode_cdr3(i,20))

encode_output= encode_cdr3_list(cdr3, 'BLOSUM62', 25)
cdr3_flattened = [np.array(seq).flatten() for seq in encode_output]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cdr3_flattened, epitope, test_size=0.2, random_state=42)

# create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# BLOSUM 90: 0.4068445163028833(n_neighbors100) 0.4230126650498518 (n_neighbors=50) 0.43470762597682566 (n_neighbors=20)
# One hot: 0.4390191323093506 (n_neighbors=20)
# I believe we can consider these results as a baseline.

'''
matrices_list = ['BENNER22', 'BENNER6', 'BENNER74', 'BLASTP', 'BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 
                 'BLOSUM90', 'DAYHOFF', 'FENG', 'GENETIC', 'GONNET1992', 'JOHNSON', 'JONES', 'LEVIN', 'MCLACHLAN', 
                 'MDM78', 'PAM250', 'PAM30', 'PAM70', 'RAO', 'RISLER', 'STR']
accuracies=[]
if len(accuracies) != len(matrices_list):
    # 用0填充不足的accuracies
    accuracies = accuracies + [0] * (len(matrices_list) - len(accuracies))
for i in range(len(matrices_list)):
    if accuracies[i]!=0:
        continue
    try:
        encode_output= encode_cdr3_list(cdr3, matrices_list[i], 25)
    except:
        print(matrices_list[i] + " failed")
        continue
    cdr3_flattened = [np.array(seq).flatten() for seq in encode_output]

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(cdr3_flattened, epitope, test_size=0.2, random_state=42)

    # create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=20)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    accuracies[i] = accuracy

import matplotlib.pyplot as plt
# 绘制条形图，轴范围为0.4到0.6，在条上显示数值
plt.barh(matrices_list, accuracies)
plt.xlim(0.4, 0.450)
for x, y in zip(accuracies, matrices_list):
    plt.text(x, y, '%.3f' % x, ha='left', va='center')
plt.savefig('knn_accuracy.png')
'''




'''
matrices_list = ['BENNER22', 'BENNER6', 'BENNER74', 'BLASTP', 'BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90', 'DAYHOFF', 'FENG', 'GENETIC', 'GONNET1992', 'JOHNSON', 'JONES', 'LEVIN', 'MCLACHLAN', 'MDM78', 'PAM250', 'PAM30', 'PAM70', 'RAO', 'RISLER', 'STR','One hot']

accuracies = [0.4291565615736998,0.4279170035030989,0.42840204796550796,0.43384532471032067,0.4357316087308003,
              0.4342225815144166,0.43384532471032067,0.43411479385610346,0.43470762597682566,0.42753974669900296,
              0.4322824036647804,0.43627054702236595,0.4294799245486392,0.43249797898140663,0.42813257881972516,
              0.42910266774454325,0.4337914308811641,0.4279170035030989,0.4275936405281595,0.436755591484775,
              0.4347615198059822,0.43179735920237133,0.4194556723255187,0.4330369172729722,0.4390191323093506]
'''