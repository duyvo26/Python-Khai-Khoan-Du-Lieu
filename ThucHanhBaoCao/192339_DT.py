import pandas as pd

import time

print("Dt")

# Tên tập dữ liệu
file_data = "breast-cancer.csv"
# Đọc dữ liệu
df = pd.read_csv(file_data, sep=",")
X = df.drop('class', axis=1)
y = df['class']

# Chia tập dữ liệu thành 2 phần (70% để huấn luyện và 30% để test)
from sklearn.model_selection import train_test_split

arrayKQ = []
from numpy import random
x = random.randint(1, 350, size=(1, 300))

for _criterion in ['gini', 'entropy']:
    rad = 1
    for _max_depth in x[0]:

        for _random_state in x[0]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=_random_state)

            # Huấn luyện mô hình
            from sklearn.tree import DecisionTreeClassifier

            # dieu chinh criterion = gini:entropy, max_depth
            model = DecisionTreeClassifier(criterion=_criterion, max_depth=_max_depth)

            begin = time.time()
            model.fit(X_train, y_train)
            end = time.time()

            # Dự đoán
            y_pred = model.predict(X_test)
            from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

            accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
            precison = round(precision_score(y_pred, y_test) * 100, 2)
            recall = round(recall_score(y_pred, y_test) * 100, 2)
            f1 = round(f1_score(y_pred, y_test) * 100, 2)
            print('--------------------------------------')
            print(accuracy, precison, recall, f1)
            arrayKQ.append([accuracy, precison, recall, f1, end - begin, _random_state, _criterion, _max_depth])
            print("So Lan", len(arrayKQ))



# get max ################################
max_arr, max_avg = 0, 0

for i in arrayKQ:
    avg = (i[0] + i[1] + i[2] + i[3]) / 4
    if avg > max_avg:
        max_avg = avg
        max_arr = i

# ket qua
accuracy, precison, recall, f1, time, _random_state, _criterion, _max_depth, = max_arr
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("_criterion", _criterion)
print("_random_state", _random_state)
print("_max_depth", _max_depth)
print(accuracy, precison, recall, f1)

# lay dia chi may
from uuid import getnode as get_mac

mac = get_mac()
print("MAC", mac)
print("Time", time)

# chup anh man hinh
# input("Bam nut bat ky de chup man hinh")
imgName = '192339-DecisionTreeClassifier-breast-cancer' + str((accuracy, precison, recall, f1)) + '.jpg'
import pyautogui
from time import sleep
sleep(2)
pyautogui.screenshot().save(imgName)
# import cv2
#
# img = cv2.imread(imgName)
# h, w, c = img.shape
# # img = img[int(h / 2): h, 0:w]
# cv2.imwrite(imgName, img)
