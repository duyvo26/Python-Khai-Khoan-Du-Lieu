import mahotas as mahotas

train_path = "demo\\train"
output_path = "\\output"

# lay label
import os
import numpy as np
import cv2

train_labels = os.listdir(train_path)
train_labels.sort()


# Dung ae lay age trUng Hu momment
def fd_hu_momments(image):
    import cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = cv2.HuMoments(cv2.moments(image)).flatten()
    return features


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


#  feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    bins = 8
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    #	normalize the histogram
    cv2.normalize(hist, hist)
    #	return the histogram
    return hist.flatten()


global_feature = []
labels = []

for train_name in train_labels:
    dir = os.path.join(train_path, train_name)
    current_label = train_name
    for x in range(1, 100):
        file = dir + "\\" + "img (" + str(x) + ").jpg"
        # print(file)
        image = cv2.imread(file)
        image = cv2.resize(image, (100, 100))
        fv_hu_momments = fd_hu_momments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)
        features = np.hstack([fv_hu_momments, fv_haralick, fv_histogram])
        labels.append(current_label)
        global_feature.append(features)
    print("[STATUS] processed folder:{}".format(current_label))

from sklearn import preprocessing

print("Feature vector size {}".format(np.array(global_feature).shape))
print("Training Labels {}".format(np.array(labels).shape))
le = preprocessing.LabelEncoder()
target = le.fit_transform(labels)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_feature)
X_train, y_train = rescaled_features, target

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train)))

test_path = "demo\\test"
test_labels = os.listdir(test_path)
test_labels.sort()

print("___TEST___")

X_test = []
y_test = []
for test_name in test_labels:
    dir = os.path.join(test_path, test_name)
    current_label = test_name
    for x in range(1, 100):
        file = dir + "\\" + "img (" + str(x) + ").jpg"
        # print(file)
        image = cv2.imread(file)
        image = cv2.resize(image, (100, 100))
        fv_hu_momments = fd_hu_momments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)
        features = np.hstack([fv_hu_momments, fv_haralick, fv_histogram])
        y_test.append(current_label)
        X_test.append(features)
    print("[STATUS] processed folder:{}".format(current_label))

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
y_test = le.fit_transform(y_test)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_test = scaler.fit_transform(X_test)
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))
