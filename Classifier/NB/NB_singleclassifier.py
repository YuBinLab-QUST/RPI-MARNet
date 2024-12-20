import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
import math

def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y

def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1

    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp + fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    sensitivity = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    mcc = float(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-06)
    f1 = float(tp*2)/(tp*2+fp+fn+1e-06)
    return acc, precision, npv, sensitivity, specificity, mcc, f1

data_start = pd.read_csv("RPI_Dataset_features.csv")
label_P = np.ones(int('positive_number'))
label_N = np.zeros(int('negative_number'))
label_start = np.hstack((label_P, label_N))
label = np.array(label_start)
data = np.array(data_start)
shu = scale(data)
y = label
[sample_num, input_dim] = np.shape(shu)
X = np.reshape(shu, (-1, 1, input_dim))

sepscores = []
ytest = np.ones((1, 2))*0.5
yscore = np.ones((1, 2))*0.5

cv_clf = GaussianNB()

skf = StratifiedKFold(n_splits=5)#对数据进行五折交叉验证

for train, test in skf.split(X, y):#遍历交叉验证的每个分组（train和test）
    y_train = to_categorical(y[train])#将训练数据的标签（y[train]）转换为独热编码（one-hot）格式，存储在y_train中
    hist = cv_clf.fit(X[train], y[train])#使用cv_clf拟合训练数据（X[train]和y[train]),训练分类器（hist）
    y_score = cv_clf.predict_proba(X[test])#计算y_score（预测概率矩阵）
    yscore = np.vstack((yscore, y_score))#将预测概率矩阵与之前的yscore矩阵合并，以备后续计算
    y_test = to_categorical(y[test])
    ytest = np.vstack((ytest, y_test))
    #根据y_test[:, 0]（真实标签）和y_score[:, 0]（预测概率）计算FPR（假阳性率）和TPR（真阳性率）
    fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)
    y_class = categorical_probas_to_classes(y_score)
    y_test_tmp = y[test]
    acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])

    print('Results: acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc))

scores = np.array(sepscores)

print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100, np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100, np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100, np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100, np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100, np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100, np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100, np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100, np.std(scores, axis=0)[7]*100))

