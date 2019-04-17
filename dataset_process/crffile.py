#用于生成CRF模型训练测试的数据集
import jieba
import re
import random
import os
from nlpcc.data import *

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 上上个目录

#输出task结果文件
def crf_result(pred_intents_a, pred_slots_a, index2word, index2slot,index2intent ,index_test):

    ALL_SLOT = {'<PAD>': 0, '<UNK>': 1, "O": 2, "B-song": 3, "B-singer": 4, "B-theme": 5,
                "B-style": 6, "B-age": 7, "B-toplist": 8, "B-emotion": 9, "B-language": 10, "B-instrument": 11,
                "B-scene": 12, "B-destination": 13, "B-custom_destination": 14, "B-origin": 15,
                "B-phone_num": 16, "B-contact_name": 17, "I-song": 18, "I-singer": 19, "I-theme": 20,
                "I-style": 21, "I-age": 22, "I-toplist": 23, "I-emotion": 24, "I-language": 25,
                "I-instrument": 26, "I-scene": 27, "I-destination": 28, "I-custom_destination": 29,
                "I-origin": 30, "I-phone_num": 31, "I-contact_name": 32}

    ALL_INTENT = {'<UNK>': 0, "music.play": 1, "music.pause": 2, "music.prev": 3, "music.next": 4,
                  "navigation.navigation": 5, "navigation.open": 6, "navigation.start_navigation": 7,
                  "navigation.cancel_navigation": 8, "phone_call.make_a_phone_call": 9, "phone_call.cancel": 10,
                  "OTHERS": 11}
    # 储存结果的文件
    fp = open(path+"\\nlpcc\\result\\crf_result.txt",'w',encoding='UTF-8')
    # 读取前两列
    data = open(path+"\\nlpcc\\result\\corpus.test.nolabel.txt",'r',encoding='UTF-8').readlines()

    data = [t[:-1] for t in data]  # 去掉'\n'

    for i in range(len(data)):      # 将答案一行一行写出
        intent = index2intent[pred_intents_a[i]]
        sequence = ""
        for j in range(index_test[i][1]):   # 语句分词数目
            if 3 <= pred_slots_a[i][j] <= 17: # 如果是"B-xx"
                sequence = sequence + "<" + (index2slot[pred_slots_a[i][j]])[2:] + ">"
                # <slot_name>
                sequence = sequence + index2word[index_test[i][0][j]]
                # slot
                if not(j+1 < index_test[i][1] and       # 未到最后一个
                       pred_slots_a[i][j+1]>=18 and     # 是“I-xx"
                       (index2slot[pred_slots_a[i][j]])[2:] == (index2slot[pred_slots_a[i][j+1]])[2:]):
                    # 如果下一个不是"I-xx"，需要写上</slot_name>
                    sequence = sequence + "</" + (index2slot[pred_slots_a[i][j]])[2:] + ">"

            elif pred_slots_a[i][j] >= 18:  # 如果是"I-xx"
                sequence = sequence + index2word[index_test[i][0][j]]
                # slot
                if not(j+1 < index_test[i][1] and       # 未到最后一个
                       pred_slots_a[i][j+1] >= 18 and     # 还是是“I-xx"
                       (index2slot[pred_slots_a[i][j]])[2:] == (index2slot[pred_slots_a[i][j+1]])[2:]):
                    # 如果下一个不是"I-xx"，需要写上</slot_name>
                    sequence = sequence + "</" + (index2slot[pred_slots_a[i][j]])[2:] + ">"

            elif pred_slots_a[i][j] == 2:       # 如果是"O"，直接输出
                sequence = sequence + index2word[index_test[i][0][j]]
                # slot

            elif pred_slots_a[i][j] == 0:       # <pad>
                continue
            else:       # <unk>
                sequence = sequence + index2word[index_test[i][0][j]]
                # slot

        fp.write(data[i]+"\t"+intent+"\t"+sequence+"\n")     #写一行结果

    print("output result...")
    fp.close()





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
            fp.write(data[i][1][j]+"\n")

        fp.writelines("\n")

def crf_test(file, outfilename):
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
            fp.write("\n")

        fp.writelines("\n")

def output_crf_result(number_file):

    data = open(number_file, "r", encoding='UTF-8').readlines()
    data = [t[:-1] for t in data]  # 去掉'\n'


    word2index = file_to_dictionary(path + "\\nlpcc\\dic\\word2index.txt")
    index2word = file_to_dictionary(path + "\\nlpcc\\dic\\index2word.txt")
    slot2index = file_to_dictionary(path + "\\nlpcc\\dic\\slot2index.txt")
    index2slot = file_to_dictionary(path + "\\nlpcc\\dic\\index2slot.txt")
    intent2index = file_to_dictionary(path + "\\nlpcc\\dic\\intent2index.txt")
    index2intent = file_to_dictionary(path + "\\nlpcc\\dic\\index2intent.txt")

    pre_slot = []
    temp =[]
    for i in range(len(data)):
        if data[i] == "":
            pre_slot.append(temp)
            temp = []
        else:
            print(data[i].split("\t"))
            temp.append(slot2index[data[i].split("\t")[1]])  #记录槽标记

    pre_slot.append(temp)   #加上最后一句

    pre_intent = [2] * 5350

    test_data_ed = file_to_list(path + "\\nlpcc\\data_list\\test_list.npy")
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)

    crf_result(pre_intent, pre_slot, index2word, index2slot, index2intent, index_test)









if __name__ == '__main__':
    '''
    crf_train(path + "\\nlpcc\\train_test_file\\train_labeled.txt",
                 path + "\\nlpcc\\train_test_file\\crf_train.txt")
    crf_test(path + "\\nlpcc\\train_test_file\\test_labeled.txt",
             path + "\\nlpcc\\train_test_file\\crf_test.txt")
    '''
    output_crf_result(path + "\\nlpcc\\result\\crf_result_number.txt")


