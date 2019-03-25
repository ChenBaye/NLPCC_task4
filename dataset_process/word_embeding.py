from gensim.models import Word2Vec
import os
import random
import numpy as np

# 从all_word_file读取词，生成并存储词向量模型
# all_word_file一般为query\\all_words.txt
path = os.path.dirname(os.path.abspath(__file__))  # 上个目录 ...\\dataset_process

def generate_model(min_count, window, size, all_word = path+"\\query\\all_words.txt"):
    data = open(all_word, "r", encoding='UTF-8').readlines()
    data = [t[:-1] for t in data]  # 去除空行
    print("line: ", len(data))
    sentences = [t.split(" ") for t in data]
    print(sentences)
    model = Word2Vec(sentences, min_count = min_count, window = window, size = size)
    #保存模型
    model.save(path+"\\word2vec\\min_count"+str(min_count)+"size"+str(size))

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


# 读取模型，并返回一个词向量list， 与word2index字典对应
def get_vector(modelname, word2index = eval(open(os.path.dirname(path)+"\\nlpcc\\dic\\word2index.txt", 'r', encoding='UTF-8').read())):
    model = Word2Vec.load(modelname)
    word_number = len(word2index)   # 共有多少个词
    vec_size = model.vector_size    # 每个词向量的维度
    word_vector = []                # 存储词向量
    # print(model["4s店"])
    # print(model["沃尔玛"])

    # 先随机生成一个词向量列表
    for i in range(word_number):
        word_vector.append(random_list(vec_size))

    print(len(word_vector))
    nothing = 0
    for key in word2index:
        try:
           vector = model[key]
           word_vector[word2index[key]] = vector
        except: # 发生异常什么都不做
            nothing = nothing + 1
        else:
            continue
    print("缺少",nothing,"个词向量")
    print(len(word_vector))
    return word_vector

if __name__ == '__main__':
    generate_model(2, 5, 380)
   # get_vector(path+"\\word2vec\\min_count5size63")