#用于生成CRF模型训练测试的数据集
import jieba
import re
import random
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 上上个目录

# 从file中读取数据（字典）
def file_to_dictionary(filename):
    print("ready to get dictionary:",filename,"............\n")
    f = open(filename, 'r', encoding='UTF-8')
    data = eval(f.read())
    f.close()
    print(data)

    return data



def crf_train(file, outfilename):
    data = open(file, "r", encoding='UTF-8').readlines()
    data = [t[:-1] for t in data]  # 去掉'\n'
    # 数据的一行像这样：111196914    播/放/dj/歌/曲	O O B-theme O O    music.play
    # 分割成这样[原始句子的词，标注的序列]

    print("切分之前: \n", data[0])
    data = [[t.split("\t")[1].split("/"), t.split("\t")[2].split(" ")] for t in
            data]

    word2index = file_to_dictionary(path + "\\nlpcc\\dic\\word2index.txt")
    fp = open(outfilename, 'w', encoding='UTF-8')

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            print(data[i][0])
            fp.write(str(word2index[data[i][0][j]]))
            fp.write(" ")
            fp.write(data[i][1][j])
            if j != (len(data[i][0]) - 1):
                fp.write("\n")
        fp.write("\n")



if __name__ == '__main__':
    crf_train(path + "\\nlpcc\\train_test_file\\train_labeled.txt",
                 path + "\\nlpcc\\train_test_file\\crf_train.txt")

