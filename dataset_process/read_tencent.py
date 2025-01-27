# 用于读取fast_text中的词向量，提取其中需要的词向量
import io
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))  # 上个目录 ...\\dataset_process


#读取fast_text中的词向量，提取需要的词向量保存在...\dataset_process\fasettext
def load_vectors(tencenttext, word2index = eval(open(os.path.dirname(path)+"\\nlpcc\\dic\\word2index.txt", 'r', encoding='UTF-8').read())):
    dictionary = {} # 存储输出的词向量
    word_number = len(word2index)  # 共有多少个词
    vec_size = 200  # 每个词向量的维度， tencent_text为300维
    word_vector = []  # 存储最终需要的词向量
    # 先随机生成一个词向量列表
    for i in range(word_number):
        word_vector.append(random_list(vec_size))

    print(len(word_vector))
    nothing = 0


    with io.open(tencenttext, 'r', encoding='utf-8', newline='\n') as fin:
        for line in fin:
            tokens = line.rstrip().split(' ')       #fasttext每一行的词 和 词向量
            if tokens[0] == "8824330":              #第一行不读
                continue
            word = tokens[0]
            vector = list(map(float, tokens[1:]))

            # 如果需要该词向量,即该词向量存在于数据集的词表中
            if word in word2index:
                print(word,"--词向量得到")
                word_vector[word2index[word]] = vector
                nothing = nothing + 1

    print("缺少", word_number - nothing, "个词向量")
    print(len(word_vector))

    f = open(path+"\\tencent\\word_vector.txt", 'w', encoding='UTF-8')
    f.write(str(word_vector))
    f.close()

# 读取挑选过词向量
def get_vector():
    f = open(path+"\\tencent\\word_vector.txt", 'r', encoding='UTF-8')
    data = eval(f.read())
    f.close()
    return data




#随机生成一个长为len，元素在-0.1到0.1的随机list
def random_list(len):
    list = []
    for i in range(len):
        num1 = np.random.random()/10    # 生成一个[0,0.1)之间的随机数
        num2 = np.random.random()
        if num2 >= 0.5:              # 另一个随机数控制正负
            num1 = num1 * (-1)
        list.append(num1)
    # print("list: ",list)
    return list




if __name__ == '__main__':
    load_vectors("D:\\TencentAI\\Tencent_AILab_ChineseEmbedding.txt")




