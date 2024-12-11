import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
# 基于L1正则化的决策树选择算法
def selectFromExtraTrees(data, label):# data为特征数据，label为数据标签
    # 创建ExtraTreesClassifier分类器clf，n_estimators表示创建的决策树个数，criterion表示划分标准，max_depth表示树的最大深度等
    clf = ExtraTreesClassifier(n_estimators=500, criterion='gini', max_depth=None,
                               min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                               max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                               min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=1,
                               random_state=None, verbose=0, warm_start=False, class_weight=None)#entropy
    clf.fit(data, label)
    # 使用分类器clf拟合数据，并得到各个特征的重要性（importance）
    importance = clf.feature_importances_
    # 创建SelectFromModel模型，并使用clf作为参数进行初始化
    model = SelectFromModel(clf, prefit=True)
    # 使用SelectFromModel模型对数据data进行特征选择，并返回选择后的新数据new_data
    new_data = model.transform(data)
    return new_data, importance

data_input = pd.read_excel(r'RPI488_2357D.xlsx')
data_ = np.array(data_input)
data = data_[:, 1:]
label = data_[:, 0]
# 使用scale函数对数据进行标准化处理，将结果存储在Zongshu变量中
Zongshu = scale(data)
RNA_shu = Zongshu[:, 0:660]
pro_shu = Zongshu[:, 660:]

# 调用selectFromExtraTrees函数对RNA_shu进行特征选择，返回选择后的新数据new_RNA_data和特征重要性index_RNA
new_RNA_data, index_RNA = selectFromExtraTrees(RNA_shu, label)
feature_numbe_RNA = -index_RNA
# 对特征重要性进行反序排序，并选取前26个特征，存储在mask_RNA变量中
H_RNA = np.argsort(feature_numbe_RNA)
mask_RNA = H_RNA[:15]
# 从RNA_shu中提取选中的特征，存储在new_data_RNA变量中
new_data_RNA = RNA_shu[:, mask_RNA]

# 调用selectFromExtraTrees函数对pro_shu进行特征选择，返回选择后的新数据new_pro_data和特征重要性index_pro
new_pro_data, index_pro = selectFromExtraTrees(pro_shu, label)
feature_numbe_pro = -index_pro
H_pro = np.argsort(feature_numbe_pro)
mask_pro = H_pro[:217]
new_data_pro = pro_shu[:, mask_pro]

# 将RNA和蛋白质的特征数据合并成一个新的特征矩阵optimal_RPI_features
optimal_RPI_features = np.hstack((new_data_RNA, new_data_pro))
