import gensim    # 训练Doc2Vec模型
from Bio import SeqIO    # 读取FASTA格式的序列文件
print("gensim version is: {}".format(gensim.__version__))
import pandas as pd

def read_fa(path):
    res={}
    rescords = list(SeqIO.parse(path, format="fasta"))
    for x in rescords:
        id = str(x.id)
        seq = str(x.seq).replace("U","T")
        res[id]=seq
    return res


def train_doc2vec_model(seq_list, model_name):
    tokens = []
    for i, seq in enumerate(seq_list):
        items = []
        k = 0
        while k + 3 < len(seq):
            item = seq[k:k + 3]
            items.append(item)
            k = k + 1
        # 首先将每个RNA序列切分为长度为3的子序列(“tokens”)，并存储为TaggedDocument对象，其中包含子序列和对应的标签（在这里是序列的索引）
        doc2vec_data = gensim.models.doc2vec.TaggedDocument(items, [i])
        tokens.append(doc2vec_data)
    print("-----begin train-----")
    # 建立Doc2Vec模型，设置向量大小为128，最小词频为3，迭代次数为100，工作进程数为12。
    # vector_size决定了输出向量的大小,默认100，构建词汇表时过滤掉出现次数少于min_count的tokens，默认5，较小数据集选取1或2，大的可设置为5或10
    model = gensim.models.doc2vec.Doc2Vec(vector_size=128, min_count=3, epochs=100, workers=12)
    # # 创建Doc2Vec模型
    # model = Doc2Vec(
    #     documents=train_docs,
    #     size=300,  # 向量维度
    #     window=8,  # 窗口大小
    #     min_count=5,  # 最小词频
    #     workers=1,  # 线程数
    #     alpha=0.025,  # 学习率
    #     iter=10  # 迭代次数
    # )
    model.build_vocab(tokens)
    model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("./data/"+model_name+".model")


'''
mirna_dict = read_fa("./data/homo_mature_mirna.fa")
mirna_list = list(mirna_dict.values())
train_doc2vec_model(mirna_dict,"mirna_doc2vec")
'''
lncrna_dict = read_fa("F:\下载\啊论文\\3.1\RNA_extract\Doc2vec\RPI_H_biaohao_rna.fa")
lncrna_list = list(lncrna_dict.values())
train_doc2vec_model(lncrna_list, "RPI_H_128doc2vec")


# 读取Doc2Vec模型
model = gensim.models.doc2vec.Doc2Vec.load("./data/RPI_H_128doc2vec.model")
# model.dv[i]获取了与索引i对应的序列的向量表示
vectors = [model.dv[i] for i in range(len(model.dv))]
# 将向量表示保存为CSV文件
df = pd.DataFrame(vectors)
df.to_csv("F:\下载\啊论文\\3.1\测试集数据\RPI_H\RPI_H_128doc.csv", index=False, header=True)
