import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


def read_data(file_name):
    df = pd.read_csv(file_name)
    X = df.drop("class", axis=1)
    y = df["class"]
    so_dac_trung = X.shape[1]
    so_mau = X.shape[0]
    so_lop = len(np.unique(y))
    print("Tap du lieu co", so_mau, "mau")
    print("Tap du lieu co", so_dac_trung, "dat trung")
    print("Tap du lieu co", so_lop, "lop")
    return X, y


def ImputeMissValue(X):
    X = pd.DataFrame(X)
    X = X.replace("?", np.nan)
    impute_nan = SimpleImputer(missing_values=np.nan)
    X = impute_nan.fit_transform(X)

    return X


def EncodeLable(y):
    encode = LabelEncoder()
    y = encode.fit_transform(y)
    return y


def check_miss_data(X):
    df = pd.DataFrame(X)
    # try:
    #     return ('?' in df.values) | (np.nan in df.values)
    # except:
    ListData = list(df.values)
    for i in ListData:
        for a in i:
            # print(a)
            if "?" == a or np.nan == a:
                return True
    return False


def GhiFileCSV(file_name, txt1, txt2):
    data = {
        'Name': [txt1],
        txt1: [txt2]
    }
    df_data = pd.DataFrame(data)
    from datetime import datetime
    today = str(datetime.today().strftime("%d_%m_%Y_%H"))
    df_data.to_csv('CSV\\data__' + file_name + '__duyvo-192339-' + today + '.csv', mode='a', index=False, header=False)


def BaggingClassifier_(X_train, X_test, y_train, y_test, status):
    from sklearn.ensemble import BaggingClassifier

    if status == "1":
        from sklearn.neighbors import KNeighborsClassifier
        base_model = KNeighborsClassifier(n_neighbors=10)

    if status == "2":
        from sklearn.tree import DecisionTreeClassifier
        base_model = DecisionTreeClassifier()

    if status == "3":
        from sklearn.naive_bayes import GaussianNB
        base_model = GaussianNB()

    if status == "4":
        from sklearn.svm import SVC
        base_model = SVC(C=10000, kernel="linear")

    if status == "5":
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(n_estimators=100)

    if status == "6":
        from sklearn.ensemble import BaggingClassifier
        base_model = BaggingClassifier(n_estimators=100)

    if status == "7":
        from sklearn.ensemble import AdaBoostClassifier
        base_model = AdaBoostClassifier(n_estimators=100)

    if status == "8":
        from sklearn.ensemble import GradientBoostingClassifier
        base_model = GradientBoostingClassifier(n_estimators=100)

    if status == "9":
        from sklearn.neural_network import MLPClassifier
        base_model = MLPClassifier(hidden_layer_sizes=100, max_iter=10000)

    if status == "10":
        from sklearn.linear_model import LogisticRegression
        base_model = LogisticRegression(C=200, max_iter=10000)


    model = BaggingClassifier(base_estimator=base_model, n_estimators=20)
    # huan luyen mo hinh
    model.fit(X_train, y_train)
    # danh gia
    y_pred = model.predict(X_test)

    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


def GradientBoostingClassifier_(X_train, X_test, y_train, y_test, status):
    # 1 KN, 2 DecisionTree, 3 GaussianNB, 4 SVC, 5  RandomFor,  6 BaggingClass, 7 AdaBoost, 8 GradientBoosting, 9 MLPClass, 10 LogisticReg

    from sklearn.ensemble import GradientBoostingClassifier

    if status == "1":
        from sklearn.neighbors import KNeighborsClassifier
        base_model = KNeighborsClassifier(n_neighbors=10)

    if status == "2":
        from sklearn.tree import DecisionTreeClassifier
        base_model = DecisionTreeClassifier()

    if status == "3":
        from sklearn.naive_bayes import GaussianNB
        base_model = GaussianNB()

    if status == "4":
        from sklearn.svm import SVC
        base_model = SVC(C=10000, kernel="linear")

    if status == "5":
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(n_estimators=100)

    if status == "6":
        from sklearn.ensemble import BaggingClassifier
        base_model = BaggingClassifier(n_estimators=100)

    if status == "7":
        from sklearn.ensemble import AdaBoostClassifier
        base_model = AdaBoostClassifier(n_estimators=100)

    if status == "8":
        from sklearn.ensemble import GradientBoostingClassifier
        base_model = GradientBoostingClassifier(n_estimators=100)

    if status == "9":
        from sklearn.neural_network import MLPClassifier
        base_model = MLPClassifier(hidden_layer_sizes=100, max_iter=10000)

    if status == "10":
        from sklearn.linear_model import LogisticRegression
        base_model = LogisticRegression(C=200, max_iter=10000)

    model = GradientBoostingClassifier(init=base_model, n_estimators=20)

    # huan luyen mo hinh
    model.fit(X_train, y_train)
    # danh gia
    y_pred = model.predict(X_test)

    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


def Main(file_name):
    path_file = "DATA\\" + file_name
    X, y = read_data(path_file)
    print("Có Dữ Liệu Lỗi", check_miss_data(X))
    X = ImputeMissValue(X)
    y = EncodeLable(y)

    # ACC 0, PER 1, REC 2, F1 3
    BaggingClassifier_arr, GradientBoostingClassifier_arr = [], []

    from sklearn.model_selection import KFold

    #######

    n_splits_ = 10
    cv = KFold(n_splits=n_splits_, shuffle=True, random_state=42)

    ListName = ['1 KN', '2 DecisionTree', '3 GaussianNB', '4 SVC', '5  RandomFor', '6 BaggingClass', '7 AdaBoost',
                '8 GradientBoosting', '9 MLPClass', '10 LogisticReg']

    for new_status in range(1, 3):
        coutn_listName = 0
        for a in range(1, int(n_splits_) + 1):
            array_save_data = []
            for train_index, test_index in cv.split(X, y):
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
                # Hoc 28d10m2022
                if new_status == 1:
                    try:
                        BaggingClassifier_out = list(BaggingClassifier_(X_train, X_test, y_train, y_test, str(a)))
                        array_save_data.append(BaggingClassifier_out)
                    except:
                        array_save_data.append([0, 0, 0, 0])
                        print("Co loi bo qua")
                if new_status == 2:
                    try:
                        GradientBoostingClassifier_out = list(
                            GradientBoostingClassifier_(X_train, X_test, y_train, y_test, str(a)))
                        array_save_data.append(GradientBoostingClassifier_out)
                    except:
                        array_save_data.append([0, 0, 0, 0])
                        print("Co loi bo qua")
            if new_status == 1:
                file_name = "BaggingClassifier"

            if new_status == 2:
                file_name = "GradientBoostingClassifier"

            coutnLan = 1
            print("---------" + ListName[coutn_listName] + "_START------------")
            out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
            for a in array_save_data:
                # for i in a:
                print("Lan thu", coutnLan, a)
                out_ACC += a[0]
                out_PER += a[1]
                out_REC += a[2]
                out_F1 += a[3]
                coutnLan += 1
            print("---------" + ListName[coutn_listName] + "_END------------")
            print(ListName[coutn_listName] + ":Accuracy", out_ACC / n_splits_)
            GhiFileCSV(file_name, ListName[coutn_listName] + ":Accuracy", str(out_ACC / n_splits_))
            print(ListName[coutn_listName] + ":Precision", out_PER / n_splits_)
            GhiFileCSV(file_name, ListName[coutn_listName] + ":Precision", str(out_PER / n_splits_))
            print(ListName[coutn_listName] + ":Recall", out_REC / n_splits_)
            GhiFileCSV(file_name, ListName[coutn_listName] + ":Recall", str(out_REC / n_splits_))
            print(ListName[coutn_listName] + ":F1", out_F1 / n_splits_)
            GhiFileCSV(file_name, ListName[coutn_listName] + ":F1", str(out_F1 / n_splits_))
            GhiFileCSV(file_name, "-----------------", "-----------------")
            coutn_listName += 1
            array_save_data.clear()


if __name__ == "__main__":
    # listFile = '01_lung-cancer,15_wpbc,21_breast-cancer,28_bupa,35_diabetes_sylhet,36_wdbc'
    listFile = []
    for (root, dirs, file) in os.walk("DATA"):  # lap lay danh sach
        for f in file:
            FileIMG = f
            Main(FileIMG)
