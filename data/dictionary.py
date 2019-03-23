# 提取 train_without_blankline.txt 以及 train_without_blankline 中各个分词


import jieba
from tkinter import _flatten

flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维

#本函数用于取出数据集中每行语句的分词
# 从
# 117194488	来一首周华健的花心	music.play	来一首<singer>周华健</singer>的<song>花心</song>
# 提取出
# ['来', '一首', '周华健', '的', '花心']
def get_wordfile(filename):

    data = open(filename, "r", encoding='UTF-8').readlines()
    data = [t[:-1] for t in data]      #去除空行
    data_words = [(" ".join(jieba.cut(t.split("\t")[1], HMM=True))).split(" ") for t in data]

    print("data_words:")
    print(data_words)

    return data_words

#得到无重复的分词词表
def get_word_list(data_words):
    word_list = list(set(flatten(data_words)))
    print("word_sum: ",len(word_list))
    #print(flatten(data_words))
    return word_list


#储存train_without_blankline.txt的分词
def save_train_words(train_words):
    # 储存分词
    fp1 = open("train_words.txt", 'w', encoding='UTF-8')  # 只包括train_without_blankline.txt的分词
    for t in train_words:
        for i in range(len(t)):
            fp1.write(t[i])
            if (i != (len(t) - 1)):
                fp1.write(" ")

        fp1.write("\n")


# 在all_words.txt保存所有分词
def save_all_words(train_words, test_words):
    # 储存分词
    fp2 = open("all_words.txt", 'w', encoding='UTF-8')  # 用于存储所有分词

    for t in train_words:
        for i in range(len(t)):
            fp2.write(t[i])
            if (i != (len(t) - 1)):
                fp2.write(" ")

        fp2.write("\n")

    for t in test_words:
        for i in range(len(t)):
            fp2.write(t[i])
            if (i != (len(t)-1)):
                fp2.write(" ")

        fp2.write("\n")


#得到变成数字的分词文件
def save_train_words_number(train_word_list, train_words):
    fp1 = open("train_words_number.txt", 'w', encoding='UTF-8')
    # 只包括train_without_blankline.txt的分词的数字形式文件

    word2index = {}
    #得到单词--数字 字典（word2index）

    for token in train_word_list:
        if token not in word2index.keys():
            word2index[token] = len(word2index)

    print("word2index")
    print(word2index)

    words_list_number=[]
    for t in train_words:

        number_list = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                                 t))
        words_list_number.append(number_list)

    #print("words_list_number")
    #print(words_list_number)

    #写入文件
    for t in words_list_number:
        for i in range(len(t)):
            fp1.write(str(t[i]))    #数字无法直接写入，先转成字符串
            if (i != (len(t) - 1)):
                fp1.write(" ")

        fp1.write("\n")


def save_all_words_number(dic_file, train_words, test_words):
    f = open(dic_file, 'r', encoding='UTF-8')
    word2index = eval(f.read())
    f.close()
    print("读取全部分词的字典：", word2index)

    # 得到单词--数字 字典（word2index），此字典包括测试集和训练集的所有分词

    print("all_word_sum: ",len(word2index))
    print(word2index)

    words_list_number = []
    for t in train_words:
        number_list = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                               t))
        words_list_number.append(number_list)

    for t in test_words:
        number_list = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                               t))
        words_list_number.append(number_list)

    # 写入文件
    fp2 = open("all_words_number.txt", 'w', encoding='UTF-8')
    for t in words_list_number:
        for i in range(len(t)):
            fp2.write(str(t[i]))  # 数字无法直接写入，先转成字符串
            if (i != (len(t) - 1)):
                fp2.write(" ")

        fp2.write("\n")





if __name__ == '__main__':
    #取出数据集中每行语句的分词
    train_words = get_wordfile("train_without_blankline.txt")
    test_words = get_wordfile("test_without_blankline.txt")

    # 在train_words.txt中保存train_words
    # save_train_words(train_words)

    # 在all_words.txt保存所有分词
    save_all_words(train_words, test_words)

    #得到无重复的单词列表
    # train_word_list = get_word_list(train_words)
    test_word_list = get_word_list(test_words)

    #得到变成数字的分词文件，输入参数是单词列表 和 分词语句
    # save_train_words_number(train_word_list, train_words)

    # 从保存好的字典文件读取所有分词的字典
    save_all_words_number("E:\\RNN-for-Joint-NLU\\nlpcc\\dic\\word2index.txt", train_words, test_words)

