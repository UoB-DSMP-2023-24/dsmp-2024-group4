from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from encoders.bio_matrices_encode import encode_cdr3

df = pd.read_csv('vdjdb.csv', header=None)
cdr3 = df[2].tolist()
epitope = df[9].tolist()

cdr3.pop(0)
epitope.pop(0)

encode_output = []
for i in cdr3:
    encode_output.append(encode_cdr3(i, 'BLOSUM90', 20))

cdr3_flattened = [np.array(seq).flatten() for seq in encode_output]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cdr3_flattened, epitope, test_size=0.2, random_state=42)

# create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=1000)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) # Accuracy: 0.4068445163028833(n_neighbors100)
print("Accuracy:", accuracy)
