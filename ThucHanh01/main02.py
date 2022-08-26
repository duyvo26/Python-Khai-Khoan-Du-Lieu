# 26d8m2022

from sklearn.datasets import load_digits
import numpy as np

data = load_digits()
X, y = data['data'], data['target']
# # # # # # # #

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, shuffle=True)
print("Tap du lieu goc\t", X.shape)
print("Tap du lieu train\t", X_train.shape)
print("Tap du lieu test\t", X_test.shape)

# # # # # # # #

from sklearn.svm import SVC

model = SVC(C=1, gamma='auto', kernel='linear')  # linear - sigmoid
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_pred, y_test) * 100
precision = precision_score(y_pred, y_test, average="macro") * 100
recall = recall_score(y_pred, y_test, average="macro") * 100
f1 = f1_score(y_pred, y_test, average="macro") * 100
print("Accuracy:\t", accuracy)
print("Precision:\t", precision)
print("Recall:\t", recall)
print("F1:\t", f1)
