import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# # 1
# def KNeighborsClassifier(X_train, X_test, y_train, y_test, n_neighbors_):
#     # -------------------------------------------------------------------------------------#
#     # Xay dung mo hinh KNN-----------
#     from sklearn.neighbors import KNeighborsClassifier
#
#     knn_model = KNeighborsClassifier(n_neighbors=n_neighbors_)
#     # Huấn luyện mô hình
#     knn_model.fit(X_train, y_train)
#     # Đánh giá
#     y_pred = knn_model.predict(X_test)
#     print("----------KN----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))
#
#
# # 2
# def DecisionTreeClassifier(X_train, X_test, y_train, y_test, criterion_):
#     # -------------------------------------------------------------------------------------#
#     # Cay quyet dinh ---------------
#     from sklearn.tree import DecisionTreeClassifier
#
#     tree_model = DecisionTreeClassifier(criterion=criterion_)  # gini entropy
#     # huan luyen mo hinh
#     tree_model.fit(X_train, y_train)
#     # danh gia
#     y_pred = tree_model.predict(X_test)
#     print("----------DecisionTreeClassifier----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))
#
#
# # 3
# def GaussianNB(X_train, X_test, y_train, y_test):
#     # -------------------------------------------------------------------------------------#
#     # da giac ---------------
#     from sklearn.naive_bayes import GaussianNB
#
#     nb_model = GaussianNB()
#     # huan luyen mo hinh
#     nb_model.fit(X_train, y_train)
#     # danh gia
#     y_pred = nb_model.predict(X_test)
#     print("----------GaussianNB----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))
#
#
# # 4
# def SVC(X_train, X_test, y_train, y_test):
#     # -------------------------------------------------------------------------------------#
#     # SVC ---------------
#     from sklearn.svm import SVC
#
#     svm_model = SVC(C=1, kernel="linear")
#     # huan luyen mo hinh
#     svm_model.fit(X_train, y_train)
#     # danh gia
#     y_pred = svm_model.predict(X_test)
#     print("----------SVC----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))
#
#
# # 5
# def RandomForestClassifier(X_train, X_test, y_train, y_test):
#     # -------------------------------------------------------------------------------------#
#     # rung ngau nhien ---------------
#     from sklearn.ensemble import RandomForestClassifier
#
#     rf_model = RandomForestClassifier(n_estimators=100)
#     # huan luyen mo hinh
#     rf_model.fit(X_train, y_train)
#     # danh gia
#     y_pred = rf_model.predict(X_test)
#     print("----------RandomForestClassifier----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))
#
#
# # 6
# def BaggingClassifier(X_train, X_test, y_train, y_test):
#     # -------------------------------------------------------------------------------------#
#     # bagging ---------------
#     from sklearn.ensemble import BaggingClassifier
#
#     bag_model = BaggingClassifier(n_estimators=100)
#     # huan luyen mo hinh
#     bag_model.fit(X_train, y_train)
#     # danh gia
#     y_pred = bag_model.predict(X_test)
#     print("----------BaggingClassifier----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))
#
#
# # 7
#
# def AdaBoostClassifier(X_train, X_test, y_train, y_test):
#     # -------------------------------------------------------------------------------------#
#     # AdaBoostClassifier ---------------
#     from sklearn.ensemble import AdaBoostClassifier
#
#     ada_model = AdaBoostClassifier(n_estimators=100)
#     # huan luyen mo hinh
#     ada_model.fit(X_train, y_train)
#     # danh gia
#     y_pred = ada_model.predict(X_test)
#     print("----------AdaBoostClassifier----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))
#
#
# # 8
# def GradientBoostingClassifier(X_train, X_test, y_train, y_test):
#     # -------------------------------------------------------------------------------------#
#     # GradientBoostingClassifier ---------------
#     from sklearn.ensemble import GradientBoostingClassifier
#
#     gb_model = GradientBoostingClassifier(n_estimators=100)
#     # huan luyen mo hinh
#     gb_model.fit(X_train, y_train)
#     # danh gia
#     y_pred = gb_model.predict(X_test)
#     print("----------GradientBoostingClassifier----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))
#
#
# # 9
# def MLPClassifier(X_train, X_test, y_train, y_test):
#     # -------------------------------------------------------------------------------------#
#     # MLP ---------------
#     from sklearn.neural_network import MLPClassifier
#
#     mlp_model = MLPClassifier(hidden_layer_sizes=100)
#     # huan luyen mo hinh
#     mlp_model.fit(X_train, y_train)
#     # danh gia
#     y_pred = mlp_model.predict(X_test)
#     print("----------MLPClassifier----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))
#
#
# # 10
# def LogisticRegression(X_train, X_test, y_train, y_test):
#     # -------------------------------------------------------------------------------------#
#     # LogisticRegression ---------------
#     from sklearn.linear_model import LogisticRegression
#
#     lr_model = LogisticRegression(C=200)
#     # huan luyen mo hinh
#     lr_model.fit(X_train, y_train)
#     # danh gia
#     y_pred = lr_model.predict(X_test)
#     print("----------LogisticRegression----------")
#     print("Accuracy:", 100 * accuracy_score(y_test, y_pred))
#     print("Precision:", 100 * precision_score(y_test, y_pred, average="weighted"))
#     print("Recall:", 100 * recall_score(y_test, y_pred, average="weighted"))
#     print("F1:.", 100 * f1_score(y_test, y_pred, average="weighted"))


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
    df_data.to_csv('CSV\\data-' + file_name + '-duyvo-192339.csv', mode='a', index=False, header=False)


def Main(file_name):
    path_file = "DATA\\" + file_name + ".csv"
    X, y = read_data(path_file)
    print("Có Dữ Liệu Lỗi", check_miss_data(X))
    X = ImputeMissValue(X)
    y = EncodeLable(y)

    # print(X, y)
    # Chia du lieu ra 2 phan
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # them thu vien KFold # 21/10/2022

    # 4 ###
    # ACC 0, PER 1, REC 2, F1 3
    SVC_new_out, \
    KN_new_out, \
    DTree_new_out, \
    GaussianNB_new_out, \
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
        print(1)
        KN_out = list(KNeighborsClassifier_new(X_train, X_test, y_train, y_test, 10))
        KN_new_out.append(KN_out)
        # 2 DTree_new_out #######
        print(2)
        DT_out = list(DecisionTreeClassifier_new(X_train, X_test, y_train, y_test, "entropy"))
        DTree_new_out.append(DT_out)
        # 3 GaussianNB_new_out ##########
        print(3)
        GaussianNB_out = list(GaussianNB_new(X_train, X_test, y_train, y_test))
        GaussianNB_new_out.append(DT_out)
        # 4 SVC ###
        print(4)
        SVC_out = list(SVC_new(X_train, X_test, y_train, y_test))
        SVC_new_out.append(SVC_out)
        # 5 RandomForestClassifier ###
        print(5)
        RandomForestClassifier_out = list(RandomForestClassifier_new(X_train, X_test, y_train, y_test))
        RandomForestClassifier_new_out.append(RandomForestClassifier_out)
        # 6 BaggingClassifier_new ###
        print(6)
        BaggingClassifier_out = list(BaggingClassifier_new(X_train, X_test, y_train, y_test))
        BaggingClassifier_new_out.append(BaggingClassifier_out)
        # 7 AdaBoostClassifier_new ###
        print(7)
        AdaBoostClassifier_out = list(AdaBoostClassifier_new(X_train, X_test, y_train, y_test))
        AdaBoostClassifier_new_out.append(AdaBoostClassifier_out)
        # 8 GradientBoostingClassifier_new ###
        print(8)
        GradientBoostingClassifier_out = list(GradientBoostingClassifier_new(X_train, X_test, y_train, y_test))
        GradientBoostingClassifier_new_out.append(GradientBoostingClassifier_out)
        # 9 MLPClassifier_new ###
        print(9)
        MLPClassifier_out = list(MLPClassifier_new(X_train, X_test, y_train, y_test))
        MLPClassifier_new_out.append(MLPClassifier_out)
        # 10 LogisticRegression ###
        print(10)
        LogisticRegression_out = list(LogisticRegression_new(X_train, X_test, y_train, y_test))
        LogisticRegression_new_out.append(LogisticRegression_out)
        ###########

    ListName = ['KN', 'DTree', 'GaussianNB', 'SVC', 'RandomForestClassifier', 'BaggingClassifier',
                'AdaBoostClassifier', 'GradientBoostingClassifier', 'MLPClassifier', 'LogisticRegression']
    coutn_listName = 0
    for a in SVC_new_out, \
             KN_new_out, \
             DTree_new_out, \
             GaussianNB_new_out, \
             RandomForestClassifier_new_out, \
             BaggingClassifier_new_out, \
             AdaBoostClassifier_new_out, \
             GradientBoostingClassifier_new_out, \
             MLPClassifier_new_out, \
             LogisticRegression_new_out:

        out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0

        for i in a:
            out_ACC += i[0]
            out_PER += i[1]
            out_REC += i[2]
            out_F1 += i[3]
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

    #
    #
    # # KN new 1 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in KN_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------KN_END------------")
    # print("KN:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "KN:Accuracy", str(out_ACC / n_splits_))
    # print("KN:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "KN:Precision", str(out_PER / n_splits_))
    # print("KN:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "KN:Recall", str(out_REC / n_splits_))
    # print("KN:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "KN:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    # ################
    #
    # # DecisionTreeClassifier_new new 2 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in DTree_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------KN_END------------")
    # print("DecisionTreeClassifier:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "DecisionTreeClassifier:Accuracy", str(out_ACC / n_splits_))
    # print("DecisionTreeClassifier:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "DecisionTreeClassifier:Precision", str(out_PER / n_splits_))
    # print("DecisionTreeClassifier:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "DecisionTreeClassifier:Recall", str(out_REC / n_splits_))
    # print("DecisionTreeClassifier:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "DecisionTreeClassifier:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    #
    # ################
    #
    # # GaussianNB_new new 3 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in GaussianNB_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------GaussianNB_END------------")
    # print("GaussianNB:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "GaussianNB:Accuracy", str(out_ACC / n_splits_))
    # print("GaussianNB:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "GaussianNB:Precision", str(out_PER / n_splits_))
    # print("GaussianNB:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "GaussianNB:Recall", str(out_REC / n_splits_))
    # print("GaussianNB:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "GaussianNB:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    #
    # ################
    #
    # # SVC new 4 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in SVC_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------SVC_END------------")
    # print("SVC:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "SVC:Accuracy", str(out_ACC / n_splits_))
    # print("SVC:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "SVC:Precision", str(out_PER / n_splits_))
    # print("SVC:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "SVC:Recall", str(out_REC / n_splits_))
    # print("SVC:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "SVC:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    #
    # ################
    #
    # # SVC new 5 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in RandomForestClassifier_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------SVC_END------------")
    # print("RandomForestClassifier:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "RandomForestClassifier:Accuracy", str(out_ACC / n_splits_))
    # print("RandomForestClassifier:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "RandomForestClassifier:Precision", str(out_PER / n_splits_))
    # print("RandomForestClassifier:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "RandomForestClassifier:Recall", str(out_REC / n_splits_))
    # print("RandomForestClassifier:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "RandomForestClassifier:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    #
    # ################
    #
    # # BaggingClassifier new 6 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in BaggingClassifier_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------BaggingClassifier_END------------")
    # print("BaggingClassifier:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "BaggingClassifier:Accuracy", str(out_ACC / n_splits_))
    # print("BaggingClassifier:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "BaggingClassifier:Precision", str(out_PER / n_splits_))
    # print("BaggingClassifier:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "BaggingClassifier:Recall", str(out_REC / n_splits_))
    # print("BaggingClassifier:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "BaggingClassifier:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    #
    # ################
    # # AdaBoostClassifier_new_out new 7 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in AdaBoostClassifier_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------AdaBoostClassifier_END------------")
    # print("AdaBoostClassifier:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "AdaBoostClassifier:Accuracy", str(out_ACC / n_splits_))
    # print("AdaBoostClassifier:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "AdaBoostClassifier:Precision", str(out_PER / n_splits_))
    # print("AdaBoostClassifier:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "AdaBoostClassifier:Recall", str(out_REC / n_splits_))
    # print("AdaBoostClassifier:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "AdaBoostClassifier:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    #
    # ################
    # # GradientBoostingClassifier_new_out new 8 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in GradientBoostingClassifier_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------GradientBoostingClassifier_END------------")
    # print("GradientBoostingClassifier:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "GradientBoostingClassifier:Accuracy", str(out_ACC / n_splits_))
    # print("GradientBoostingClassifier:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "GradientBoostingClassifier:Precision", str(out_PER / n_splits_))
    # print("GradientBoostingClassifier:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "GradientBoostingClassifier:Recall", str(out_REC / n_splits_))
    # print("GradientBoostingClassifier:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "GradientBoostingClassifier:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    #
    # # MLPClassifier_new new 9 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in MLPClassifier_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------MLPClassifier_END------------")
    # print("MLPClassifier:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "MLPClassifier:Accuracy", str(out_ACC / n_splits_))
    # print("MLPClassifier:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "MLPClassifier:Precision", str(out_PER / n_splits_))
    # print("MLPClassifier:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "MLPClassifier:Recall", str(out_REC / n_splits_))
    # print("MLPClassifier:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "MLPClassifier:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    #
    # # LogisticRegression_new_out new 10 ####
    # out_ACC, out_PER, out_REC, out_F1 = 0, 0, 0, 0
    # for i in LogisticRegression_new_out:
    #     out_ACC += i[0]
    #     out_PER += i[1]
    #     out_REC += i[2]
    #     out_F1 += i[3]
    # print("---------LogisticRegression_END------------")
    # print("LogisticRegression:Accuracy", out_ACC / n_splits_)
    # GhiFileCSV(file_name, "LogisticRegression:Accuracy", str(out_ACC / n_splits_))
    # print("LogisticRegression:Precision", out_PER / n_splits_)
    # GhiFileCSV(file_name, "LogisticRegression:Precision", str(out_PER / n_splits_))
    # print("LogisticRegression:Recall", out_REC / n_splits_)
    # GhiFileCSV(file_name, "LogisticRegression:Recall", str(out_REC / n_splits_))
    # print("LogisticRegression:F1", out_F1 / n_splits_)
    # GhiFileCSV(file_name, "LogisticRegression:F1", str(out_F1 / n_splits_))
    # GhiFileCSV(file_name, "-----------------", "-----------------")
    #
    # ################


if __name__ == "__main__":
    listFile = '01_lung-cancer,15_wpbc,21_breast-cancer,28_bupa,35_diabetes_sylhet,36_wdbc'
    for i in listFile.split(","):
        print("______________________________", i, "___________________________")
        Main(i)

##########OLD##########
# 1
# KNeighborsClassifier(X_train, X_test, y_train, y_test, 10)
# 2
# gini, entropy
# DecisionTreeClassifier(X_train, X_test, y_train, y_test, "gini")
# 3
#     GaussianNB(X_train, X_test, y_train, y_test)
# 4
#     SVC(X_train, X_test, y_train, y_test)
# 5
#     RandomForestClassifier(X_train, X_test, y_train, y_test)
# 6
#     BaggingClassifier(X_train, X_test, y_train, y_test)
# 7
#     AdaBoostClassifier(X_train, X_test, y_train, y_test)
# 8
#     GradientBoostingClassifier(X_train, X_test, y_train, y_test)
# 9
#     MLPClassifier(X_train, X_test, y_train, y_test)
# 10
#     LogisticRegression(X_train, X_test, y_train, y_test)
