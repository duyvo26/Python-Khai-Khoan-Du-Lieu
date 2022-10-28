import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


# Hoc 21/10/2022 #############################################################################
# 1 21/10/2022
def KNeighborsClassifier_new(X_train, X_test, y_train, y_test, n_neighbors_):
    # -------------------------------------------------------------------------------------#
    # Xay dung mo hinh KNN-----------
    from sklearn.neighbors import KNeighborsClassifier

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors_)
    # Huấn luyện mô hình
    knn_model.fit(X_train, y_train)
    # Đánh giá
    y_pred = knn_model.predict(X_test)
    # print("----------KN----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


# 2 21/10/2022
def DecisionTreeClassifier_new(X_train, X_test, y_train, y_test, criterion_):
    # -------------------------------------------------------------------------------------#
    # Cay quyet dinh ---------------
    from sklearn.tree import DecisionTreeClassifier

    tree_model = DecisionTreeClassifier(criterion=criterion_)  # gini entropy
    # huan luyen mo hinh
    tree_model.fit(X_train, y_train)
    # danh gia
    y_pred = tree_model.predict(X_test)
    # print("----------DecisionTreeClassifier----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


# 3 21/10/2022
def GaussianNB_new(X_train, X_test, y_train, y_test):
    # -------------------------------------------------------------------------------------#
    # da giac ---------------
    from sklearn.naive_bayes import GaussianNB

    nb_model = GaussianNB()
    # huan luyen mo hinh
    nb_model.fit(X_train, y_train)
    # danh gia
    y_pred = nb_model.predict(X_test)
    # print("----------GaussianNB----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


# 4  21/10/2022
def SVC_new(X_train, X_test, y_train, y_test):
    # -------------------------------------------------------------------------------------#
    # Xay dung mo hinh SVC-----------
    from sklearn.svm import SVC
    model_svc = SVC(C=10000, kernel="linear", max_iter=10000)
    # Huấn luyện mô hình
    model_svc.fit(X_train, y_train)
    # Đánh giá
    y_pred = model_svc.predict(X_test)
    # print("----------SVC----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


# 5  21/10/2022
def RandomForestClassifier_new(X_train, X_test, y_train, y_test):
    # -------------------------------------------------------------------------------------#
    # rung ngau nhien ---------------
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(n_estimators=100)
    # huan luyen mo hinh
    rf_model.fit(X_train, y_train)
    # danh gia
    y_pred = rf_model.predict(X_test)
    # print("----------RandomForestClassifier----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


# 6 21/10/2022
def BaggingClassifier_new(X_train, X_test, y_train, y_test):
    # -------------------------------------------------------------------------------------#
    # bagging ---------------
    from sklearn.ensemble import BaggingClassifier

    bag_model = BaggingClassifier(n_estimators=100)
    # huan luyen mo hinh
    bag_model.fit(X_train, y_train)
    # danh gia
    y_pred = bag_model.predict(X_test)
    # print("----------BaggingClassifier----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


# 7 21/10/2022
def AdaBoostClassifier_new(X_train, X_test, y_train, y_test):
    # -------------------------------------------------------------------------------------#
    # AdaBoostClassifier ---------------
    from sklearn.ensemble import AdaBoostClassifier

    ada_model = AdaBoostClassifier(n_estimators=100)
    # huan luyen mo hinh
    ada_model.fit(X_train, y_train)
    # danh gia
    y_pred = ada_model.predict(X_test)
    # print("----------AdaBoostClassifier----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


# 8 21/10/2022
def GradientBoostingClassifier_new(X_train, X_test, y_train, y_test):
    # -------------------------------------------------------------------------------------#
    # GradientBoostingClassifier ---------------
    from sklearn.ensemble import GradientBoostingClassifier

    gb_model = GradientBoostingClassifier(n_estimators=100)
    # huan luyen mo hinh
    gb_model.fit(X_train, y_train)
    # danh gia
    y_pred = gb_model.predict(X_test)
    # print("----------GradientBoostingClassifier----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


# 9 21/10/2022
def MLPClassifier_new(X_train, X_test, y_train, y_test):
    # -------------------------------------------------------------------------------------#
    # MLP ---------------
    from sklearn.neural_network import MLPClassifier

    mlp_model = MLPClassifier(hidden_layer_sizes=100, max_iter=10000)
    # huan luyen mo hinh
    mlp_model.fit(X_train, y_train)
    # danh gia
    y_pred = mlp_model.predict(X_test)
    # print("----------MLPClassifier----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


# 10 21/10/2022
def LogisticRegression_new(X_train, X_test, y_train, y_test):
    # -------------------------------------------------------------------------------------#
    # LogisticRegression ---------------
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression(C=200, max_iter=10000)
    # huan luyen mo hinh
    lr_model.fit(X_train, y_train)
    # danh gia
    y_pred = lr_model.predict(X_test)
    # print("----------LogisticRegression----------")
    Accuracy = 100 * accuracy_score(y_test, y_pred)
    yield Accuracy
    Precision = 100 * precision_score(y_test, y_pred, average="weighted")
    yield Precision
    Recall = 100 * recall_score(y_test, y_pred, average="weighted")
    yield Recall
    F1 = 100 * f1_score(y_test, y_pred, average="weighted", zero_division=False)
    yield F1


############

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


def Main(file_name):
    path_file = "DATA\\" + file_name
    X, y = read_data(path_file)
    print("Có Dữ Liệu Lỗi", check_miss_data(X))
    X = ImputeMissValue(X)
    y = EncodeLable(y)

    # ACC 0, PER 1, REC 2, F1 3
    KN_new_out, \
    DTree_new_out, \
    GaussianNB_new_out, \
    SVC_new_out, \
    RandomForestClassifier_new_out, \
    BaggingClassifier_new_out, \
    AdaBoostClassifier_new_out, \
    GradientBoostingClassifier_new_out, \
    MLPClassifier_new_out, \
    LogisticRegression_new_out \
        = [], [], [], [], [], [], [], [], [], []

    from sklearn.model_selection import KFold

    #######

    n_splits_ = 10
    cv = KFold(n_splits=n_splits_, shuffle=True, random_state=42)
    for train_index, test_index in cv.split(X, y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        # 1 KN ####
        # print(1)
        KN_out = list(KNeighborsClassifier_new(X_train, X_test, y_train, y_test, 10))
        KN_new_out.append(KN_out)
        # 2 DTree_new_out #######
        # print(2)
        DT_out = list(DecisionTreeClassifier_new(X_train, X_test, y_train, y_test, "entropy"))
        DTree_new_out.append(DT_out)
        # 3 GaussianNB_new_out ##########
        # print(3)
        GaussianNB_out = list(GaussianNB_new(X_train, X_test, y_train, y_test))
        GaussianNB_new_out.append(GaussianNB_out)
        # 4 SVC ###
        # print(4)
        SVC_out = list(SVC_new(X_train, X_test, y_train, y_test))
        SVC_new_out.append(SVC_out)
        # 5 RandomForestClassifier ###
        # print(5)
        RandomForestClassifier_out = list(RandomForestClassifier_new(X_train, X_test, y_train, y_test))
        RandomForestClassifier_new_out.append(RandomForestClassifier_out)
        # 6 BaggingClassifier_new ###
        # print(6)
        BaggingClassifier_out = list(BaggingClassifier_new(X_train, X_test, y_train, y_test))
        BaggingClassifier_new_out.append(BaggingClassifier_out)
        # 7 AdaBoostClassifier_new ###
        # print(7)
        AdaBoostClassifier_out = list(AdaBoostClassifier_new(X_train, X_test, y_train, y_test))
        AdaBoostClassifier_new_out.append(AdaBoostClassifier_out)
        # 8 GradientBoostingClassifier_new ###
        # print(8)
        GradientBoostingClassifier_out = list(GradientBoostingClassifier_new(X_train, X_test, y_train, y_test))
        GradientBoostingClassifier_new_out.append(GradientBoostingClassifier_out)
        # 9 MLPClassifier_new ###
        # print(9)
        MLPClassifier_out = list(MLPClassifier_new(X_train, X_test, y_train, y_test))
        MLPClassifier_new_out.append(MLPClassifier_out)
        # 10 LogisticRegression ###
        # print(10)
        LogisticRegression_out = list(LogisticRegression_new(X_train, X_test, y_train, y_test))
        LogisticRegression_new_out.append(LogisticRegression_out)
        ###########

    ListName = ['KN', 'DTree', 'GaussianNB', 'SVC', 'RandomForestClassifier', 'BaggingClassifier',
                'AdaBoostClassifier', 'GradientBoostingClassifier', 'MLPClassifier', 'LogisticRegression']
    coutn_listName = 0
    print(KN_new_out, \
          DTree_new_out, \
          GaussianNB_new_out, \
          SVC_new_out, \
          RandomForestClassifier_new_out, \
          BaggingClassifier_new_out, \
          AdaBoostClassifier_new_out, \
          GradientBoostingClassifier_new_out, \
          MLPClassifier_new_out, \
          LogisticRegression_new_out)
    for a in KN_new_out, \
             DTree_new_out, \
             GaussianNB_new_out, \
             SVC_new_out, \
             RandomForestClassifier_new_out, \
             BaggingClassifier_new_out, \
             AdaBoostClassifier_new_out, \
             GradientBoostingClassifier_new_out, \
             MLPClassifier_new_out, \
             LogisticRegression_new_out:

        out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
        coutnLan = 1
        print(a)
        for i in a:
            print("Lan thu", coutnLan, i)
            out_ACC += i[0]
            out_PER += i[1]
            out_REC += i[2]
            out_F1 += i[3]
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


if __name__ == "__main__":
    # listFile = '01_lung-cancer,15_wpbc,21_breast-cancer,28_bupa,35_diabetes_sylhet,36_wdbc'
    listFile = []
    for (root, dirs, file) in os.walk("DATA"):  # lap lay danh sach
        for f in file:
            FileIMG = f
            Main(FileIMG)
