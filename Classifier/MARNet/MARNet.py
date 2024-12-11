import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from keras.layers import Conv1D, Dense, AveragePooling1D
import math
from keras.layers import Layer, Input,  MaxPooling1D, BatchNormalization, Add, Flatten, ReLU, Dropout, GRU
from keras.models import Model
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)


def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y


def get_shuffle(data, label):
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label


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
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp + 1e-06)
    npv = float(tn) / (tn + fn + 1e-06)
    sensitivity = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn) / (tn + fp + 1e-06)
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06)
    f1 = float(tp * 2) / (tp * 2 + fp + fn + 1e-06)
    return acc, precision, npv, sensitivity, specificity, mcc, f1


class MyMultiHeadAttention(Layer):
    def __init__(self, output_dim, num_head, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_head = num_head
        self.kernel_initializer = kernel_initializer
        super(MyMultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(self.num_head, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo', shape=(self.num_head * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.built = True

    def call_(self, x):
        for i in range(self.W.shape[0]):  # 多个头循环计算
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])
            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
            e = e / (self.output_dim ** 0.5)
            e = K.softmax(e)
            o = K.batch_dot(e, v)
            if i == 0:
                outputs = o
            else:
                outputs = K.concatenate([outputs, o])
        z = K.dot(outputs, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


'''（a) 原始残差块original'''
# def residual_block(x, kernel_size=1):
#     y = Conv1D(128, kernel_size=kernel_size, strides=1, padding='same')(x)
#     y = BatchNormalization()(y)
#     y = ReLU()(y)
#     y = Conv1D(64, kernel_size=kernel_size, strides=1, padding='same')(y)
#     y = BatchNormalization()(y)
#     # y = ReLU()(y)
#     # y = Conv1D(32, kernel_size=kernel_size, strides=1, padding='same')(y)
#     # y = BatchNormalization()(y)
#     # y = Dense(32, activation='relu')(y)
#     # projection = Conv1D(filters, kernel_size=1, strides=1, padding='same')(x)
#     projection = GRU(64)(x)
#     projection = K.expand_dims(projection, axis=2)
#     y = Add()([projection, y])
#     y = Activation('relu')(y)
#     return y


'''(b)  在pre-activation残差块加和后加BN和ReLU——————BN after addition'''
# def residual_block(x, kernel_size=1):
#     y = Conv1D(128, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
#     y = BatchNormalization()(y)
#     y = ReLU()(y)
#     y = Conv1D(64, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(y)
#     # y = BatchNormalization()(y)
#     # y = ReLU()(y)
#     # y = Conv1D(32, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(y)
#     # y = AveragePooling1D()(y)  # 删除 AveragePooling1D
#     # y = Dense(32, activation='relu')(y)
#     # projection = Conv1D(filters, kernel_size=1, strides=1, padding='same')(x)
#     projection = GRU(64)(x)
#     projection = K.expand_dims(projection, axis=2)
#     y = Add()([projection, y])
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#     return y


'''(c)  ReLU before addition'''
# def residual_block(x, kernel_size=1):
#     y = Conv1D(128, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(x)
#     y = BatchNormalization()(y)
#     y = ReLU()(y)
#     y = Conv1D(64, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(y)
#     y = BatchNormalization()(y)
#     y = ReLU()(y)
#     # y = Conv1D(32, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(y)
#     # y = BatchNormalization()(y)
#     # y = ReLU()(y)
#     # projection = Conv1D(filters, kernel_size=1, strides=1, padding='same')(x)
#     projection = GRU(64)(x)
#     projection = K.expand_dims(projection, axis=2)
#     y = Add()([projection, y])
#     return y


'''(d)  ReLU -only pre- activation'''
# def residual_block(x, kernel_size=1):
#     y = ReLU()(x)
#     y = Conv1D(128, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(y)
#     y = BatchNormalization()(y)
#     y = ReLU()(y)
#     y = Conv1D(64, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(y)
#     y = BatchNormalization()(y)
#     # y = ReLU()(y)
#     # y = Conv1D(32, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(y)
#     # y = BatchNormalization()(y)
#     # y = AveragePooling1D()(y)  # 删除 AveragePooling1D
#     # y = Dense(32, activation='relu')(y)
#     # projection = Conv1D(filters, kernel_size=1, strides=1, padding='same')(x)
#     projection = GRU(64)(x)
#     projection = K.expand_dims(projection, axis=2)
#     y = Add()([projection, y])
#     return y


'''(e） 两层pre-activation残差块'''
def residual_block(x, kernel_size=1):
    y = BatchNormalization()(x)
    y = ReLU()(y)
    y = Conv1D(128, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv1D(64, kernel_size=kernel_size, strides=1, padding='same')(y)
    # y = BatchNormalization()(y)
    # y = ReLU()(y)
    # y = Conv1D(32, kernel_size=kernel_size, strides=1, padding='same')(y)
    projection = GRU(64)(x)
    projection = K.expand_dims(projection, axis=2)
    y = Add()([projection, y])
    return y


'''原始残差块1'''
# def residual_block(x, kernel_size=1):
#     y = Conv1D(128, kernel_size=kernel_size, strides=1, padding='same')(x)
#     y = BatchNormalization()(y)
#     y = ReLU()(y)
#     y = Conv1D(64, kernel_size=kernel_size, strides=1, padding='same')(y)
#     y = BatchNormalization()(y)
#     # y = ReLU()(y)
#     # y = Conv1D(32, kernel_size=kernel_size, strides=1, padding='same')(y)
#     # y = BatchNormalization()(y)
#     # y = AveragePooling1D()(y)  # 删除 AveragePooling1D
#     # y = Dense(32, activation='relu')(y)
#     # projection = Conv1D(filters, kernel_size=1, strides=1, padding='same')(x)
#     projection = GRU(64)(x)
#     projection = K.expand_dims(projection, axis=2)
#     y = Add()([projection, y])
#     y = Activation('relu')(y)
#     return y


def build_model(input_dim, out_dim, num_head):
    input_layer = Input(shape=(1, input_dim))
    x = MyMultiHeadAttention(output_dim=out_dim, num_head=num_head)(input_layer)
    x = Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)

    x = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = AveragePooling1D(pool_size=2, strides=2, padding='same')(x)

    x = Flatten()(x)

    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(out_dim, activation='sigmoid', name="Dense_2")(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

# user_input_path = "F:\下载\啊论文\\3.1\Feature selection\降维数据集\数据集LASSO降维\RPI1807_0.00175LASSO.csv"
user_input_path = "F:\下载\啊论文\\3.1\Feature selection\不同算法下的15+217\RPI488_GINI15_217.csv"
num_head = 2
data_start = pd.read_csv(user_input_path)

label_N = np.zeros(int('245'))
label_P = np.ones(int('243'))
label_start = np.hstack((label_N, label_P))
label = np.array(label_start)


data = np.array(data_start)
shu = scale(data)
y = label
[sample_num, input_dim] = np.shape(shu)
print('sample_num, input_dim',sample_num, input_dim)
X = np.reshape(shu, (-1, 1, input_dim))
out_dim = 2
# (data, label) = get_shuffle(data, label)
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5
model = build_model(input_dim, out_dim, num_head)
sepscores = []


skf = StratifiedKFold(n_splits=5)

# for train, test in skf.split(X, y):
for fold, (train, test) in enumerate(skf.split(X, y), 1):
    y_train = to_categorical(y[train])  # generate the resonable results
    y_pred, attention_weights = model.predict(X[test])
    # 绘制注意力权重热图
    for i in range(len(attention_weights)):  # attention_weights的形状是(batch_size, num_heads, seq_len, seq_len)
        attn = attention_weights[i][0]  # 获取第一个样本的第i个头的注意力权重
        plt.figure(figsize=(10, 8))
        attn_2d = attn.reshape((attn.shape[0], attn.shape[1]))  # 转换为二维形式
        sns.heatmap(attn_2d, cmap='viridis', annot=True)  # , fmt=".2f"
        plt.title(f"Fold {fold} - Attention Head {i + 1}")
        plt.xlabel("Sequence Position")
        plt.ylabel("Features")
        plt.show()

    cv_clf = build_model(input_dim, out_dim, num_head)
    hist = cv_clf.fit(X[train],
                      y_train,
                      epochs=30)
    y_test = to_categorical(y[test])  # generate the test
    ytest = np.vstack((ytest, y_test))
    y_test_tmp = y[test]
    y_score = cv_clf.predict(X[test])  # the output of  probability
    yscore = np.vstack((yscore, y_score))
    fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)
    y_class = categorical_probas_to_classes(y_score)
    acc, precision, npv, sensitivity, specificity, mcc, f1 = calculate_performace(len(y_class), y_class, y_test_tmp)
    hist = []
    cv_clf = []
scores = np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100))


# result1=np.mean(scores, axis=0)
# H1=result1.tolist()
# sepscores.append(H1)
# result=sepscores
# data_csv_zhibiao = pd.DataFrame(data=result)
# row=yscore.shape[0]
# yscore=yscore[np.array(range(1,row)),:]
# yscore_sum = pd.DataFrame(data=yscore)
#
# ytest=ytest[np.array(range(1,row)),:]
# ytest_sum = pd.DataFrame(data=ytest)
