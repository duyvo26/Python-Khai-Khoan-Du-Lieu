from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1)
mnist.keys()
X, y = mnist['data'], mnist['target']
print(X.shape)
print(y.shape)



# import matplotlib.pyplot as plt
# some_digit = X.to_numpy()[6]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap="binary")
# plt.axis("off")
# plt.show()


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=True) 
print(X_train.shape, y_train.shape) 
print(X_test.shape, y_test.shape) 


# from sklearn.neighbors import KNeighborsClassifier 
# model = KNeighborsClassifier(n_neighbors=1) 
# model.fit(X_train, y_train) 
# from sklearn.metrics import accuracy_score 
# print("ACC tren tap train") 
# y_pred_train = model.predict(X_train) 
# print("ACC Train:", accuracy_score(y_pred_train, y_train))
# y_pred_test = model.predict(X_test) 
# print("ACC Test:", accuracy_score(y_pred_test, y_test))



from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train) 
from sklearn.metrics import accuracy_score 
print("ACC tren tap train") 
y_pred_train = model.predict(X_train) 
print("ACC Train:", accuracy_score(y_pred_train, y_train))
y_pred_test = model.predict(X_test) 
print("ACC Test:", accuracy_score(y_pred_test, y_test))



from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train) 
from sklearn.metrics import accuracy_score 
print("ACC tren tap train") 
y_pred_train = model.predict(X_train) 
print("ACC Train:", accuracy_score(y_pred_train, y_train))
y_pred_test = model.predict(X_test) 
print("ACC Test:", accuracy_score(y_pred_test, y_test))




from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier(criterion='entropy', max_features=0.5)
model.fit(X_train, y_train) 
from sklearn.metrics import accuracy_score 
print("ACC tren tap train") 
y_pred_train = model.predict(X_train) 
print("ACC Train:", accuracy_score(y_pred_train, y_train))
y_pred_test = model.predict(X_test) 
print("ACC Test:", accuracy_score(y_pred_test, y_test))





from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier(criterion='entropy', max_depth=2)
model.fit(X_train, y_train) 
from sklearn.metrics import accuracy_score 
print("ACC tren tap train") 
y_pred_train = model.predict(X_train) 
print("ACC Train:", accuracy_score(y_pred_train, y_train))
y_pred_test = model.predict(X_test) 
print("ACC Test:", accuracy_score(y_pred_test, y_test))




from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=2)
model.fit(X_train, y_train) 
from sklearn.metrics import accuracy_score 
print("ACC tren tap train") 
y_pred_train = model.predict(X_train) 
print("ACC Train:", accuracy_score(y_pred_train, y_train))
y_pred_test = model.predict(X_test) 
print("ACC Test:", accuracy_score(y_pred_test, y_test))




















