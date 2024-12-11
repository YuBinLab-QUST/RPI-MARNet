# RPI-MARNet
RPI-MARNet：
Prediction of RNA-protein interaction with interpretability based on protein language models and enhanced residual networks
# Overview
RNA–protein interactions (RPIs) are essential for understanding biological processes. This study presents a multimodal information fusion RPI prediction model based on a multi-head attention mechanism and an enhanced residual network, named RPI-MARNet.RNA and protein sequences are processed using deep feature extraction techniques, including but not limited to language models. The fused high-dimensional data undergo feature selection via LASSO to identify the optimal feature subset. The multi-head attention mechanism captures key features contributing to RPI prediction from multiple perspectives and dimensions, while the pre-activated residual network structure effectively mitigates gradient vanishing and information loss. Residual blocks incorporating GRU further enhance feature retention and propagation, improving training efficiency and model performance. 
# Requirements

 * Python 3.11
 * numpy
 * scipy
 * scikit-learn
 * pandas
 * tensorflow 
 * keras

# Guiding principles: 

**The dataset file contains seven datasets, among which RPI488, RPI369, RPI1446, RPI1807, RPI2241, NPInter v3.0.

**Feature extraction：
 * feature-RNA is the implementation of k-mer,  KGap descriptor, and Doc2vec for RNA.
 * feature-protein is the implementation of CTD information coding, Residue probing transformation (RPT) , and ProBERT for protein.
 

**Feature_selection:
 * ALL_select is the implementation of all feature selection methods used in this work, among which LLE,SE,MDS,LR,OMP,TSVD,GINI,ET,LGBM,MRMD,LASSO.

**Classifier:
 * RPI-MARNet.py is the implementation of our model in this work.
 * classical classifier is the implementation of classical classifierS compared in this work, among which RF,SVM,MLP,NB,ET,AdaBoost,KNN.
 * deep learning classifier is the implementation of deep learning classifiers compared in this work, among which  CNN, DNN, GRU,RNN,GAN.
