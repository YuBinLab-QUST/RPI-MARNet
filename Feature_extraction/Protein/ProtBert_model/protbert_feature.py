from time import time

import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import numpy as np
import gzip
import pickle


'''从给定的URL下载ProtBert模型文件（模型、配置和词汇表），并保存在指定的目录root_dir + './ProtBert_model/'中'''
def generate_protbert_features(root_dir):
    t0 = time()
    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'



    downloadFolderPath = root_dir + './ProtBert_model/'

    modelFolderPath = downloadFolderPath

    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')

    configFilePath = os.path.join(modelFolderPath, 'config.json')

    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)

    def download_file(url, filename):
        # 使用requests库从指定的URL下载文件
        response = requests.get(url, stream=True)
        # 使用tqdm.wrapattr()方法将文件对象包装成tqdm进度条对象，以便在下载过程中显示下载进度
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                           total=int(response.headers.get('content-length', 0)),
                           desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)

    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)

    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)

    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False)
    model = BertModel.from_pretrained(modelFolderPath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model = model.eval()

    def make_aseq(seq):
        protAlphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        return ' '.join([protAlphabet[x] for x in seq])

    sequences = []


    # 读取id_list.txt文件，获取蛋白质ID列表，然后循环读取每个蛋白质序列文件（RPI369_protein.fa），提取蛋白质序列并存储在sequences列表中
    with open(root_dir + './id_list.txt', 'r') as f:
        protein_list = f.readlines()
        for protein in protein_list:
            seq = open(root_dir + '../data/RPI369_protein.fa'.format(protein.strip()), 'r').readlines()
            sequences += [seq[1].strip()]

    # 将蛋白质序列转换为空格分隔的字母序列，并将一些特殊字符替换为'X'
    sequences_Example = [' '.join(list(seq)) for seq in sequences]
    sequences_Example = [re.sub(r"[-UZOB]", "X", sequence) for sequence in sequences_Example]

    all_protein_features = []

    for i, seq in enumerate(sequences_Example):
        # tokenizer对序列进行编码,tokenizer.batch_encode_plus()方法将序列转换为模型可以接受的格式，包括添加特殊标记和填充到相同长度
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():     # 使用了torch.no_grad()上下文管理器，表示不需要计算梯度
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len - 1]
            features.append(seq_emd)
        #     print(features.__len__())
        #     print(features[0].shape)
        # print(all_protein_sequences['all_protein_complex_pdb_ids'][i])
        #     print(features)
        all_protein_features += features

    # 使用pickle.dump()将所有蛋白质的特征向量保存到一个压缩的.pkl.gz文件中
    pickle.dump({'ProtBert_features': all_protein_features},
                gzip.open(root_dir + '/inputs/ProtBert_features.pkl.gz',
                          'wb')
                )

    print('Total time spent for ProtBERT:', time() - t0)

generate_protbert_features('F:\下载\啊论文\\1.19论文\code')

