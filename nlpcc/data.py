# coding=utf-8

import random
import numpy as np
import os

#只有如下槽



ALL_SLOT={'<PAD>': 0, '<UNK>': 1, "O": 2, "B-song": 3, "B-singer": 4, "B-theme": 5,
      "B-style": 6, "B-age": 7, "B-toplist": 8, "B-emotion": 9, "B-language": 10, "B-instrument": 11,
      "B-scene": 12, "B-destination": 13, "B-custom_destination": 14, "B-origin": 15,
      "B-phone_num": 16, "B-contact_name": 17,"I-song": 18, "I-singer": 19, "I-theme": 20,
      "I-style": 21, "I-age": 22, "I-toplist": 23, "I-emotion": 24, "I-language": 25,
      "I-instrument": 26, "I-scene": 27, "I-destination": 28, "I-custom_destination": 29,
      "I-origin": 30, "I-phone_num": 31, "I-contact_name": 32}
'''
ALL_SLOT={'<PAD>': 0, "O": 1, "B-song": 2, "B-singer": 3, "B-theme": 4,
      "B-style": 5, "B-age": 6, "B-toplist": 7, "B-emotion": 8, "B-language": 9, "B-instrument": 10,
      "B-scene": 11, "B-destination": 12, "B-custom_destination": 13, "B-origin": 14,
      "B-phone_num": 15, "B-contact_name": 16,"I-song": 17, "I-singer": 18, "I-theme": 19,
      "I-style": 20, "I-age": 21, "I-toplist": 22, "I-emotion": 23, "I-language": 24,
      "I-instrument": 25, "I-scene": 26, "I-destination": 27, "I-custom_destination": 28,
      "I-origin": 29, "I-phone_num": 30, "I-contact_name": 31}
'''
#只有如下意图



ALL_INTENT={'<UNK>': 0,"music.play": 1, "music.pause": 2, "music.prev": 3, "music.next": 4,
        "navigation.navigation": 5, "navigation.open": 6, "navigation.start_navigation": 7,
        "navigation.cancel_navigation": 8, "phone_call.make_a_phone_call": 9, "phone_call.cancel": 10,
        "OTHERS": 11}



'''
ALL_INTENT={"music.play": 0, "music.pause": 1, "music.prev": 2, "music.next": 3,
        "navigation.navigation": 4, "navigation.open": 5, "navigation.start_navigation": 6,
        "navigation.cancel_navigation": 7, "phone_call.make_a_phone_call": 8, "phone_call.cancel": 9,
        "OTHERS": 10}
'''
flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]


def data_pipeline(data, file_name, length, option):     # 规定语句长度定为 input_steps ，不足用EOS+PAD补上
    data = [t[:-1] for t in data]  # 去掉'\n'
    # 数据的一行像这样：'BOS i want to fly from baltimore to dallas round trip EOS
    # \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # 分割成这样[原始句子的词，标注的序列，intent]
    print("切分之前: \n",data[0])
    data = [[t.split("\t")[1].split(" "), t.split("\t")[2].split(" ")[:-1], t.split("\t")[2].split(" ")[-1]] for t in
            data]
    #按tab分割，导致标注序列和intent在同一块（标注序列和intent中间是空格），使用[-1:]和[-1]分开
    print("切分之后: \n",data[0])

    #data = [[t[0][1:-1], t[1][1:], t[2]] for t in data]  # 将BOS和EOS去掉，并去掉对应标注序列中相应的标注
    seq_in, seq_out, intent = list(zip(*data))
    sin = []
    sout = []


    #统计句子中分词的最大数目以便确定input_size
    max=len(seq_in[0])
    for i in seq_in:
        if len(i) > max:
            max = len(i)
    print("一个句子最多有：", max, "个分词")


    # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        print(i)
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]        #如果长度>=length，则截取length长度，并在最后补上<EOS>
            temp[-1] = '<EOS>'
        sin.append(temp)

        temp = seq_out[i]
        if len(temp) < length:
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sout.append(temp)
        data = list(zip(sin, sout, intent))

    print("结尾补上<EOS>+N*<PAD>之后: \n", data[0])
    print("\n")

    #保存数据，方便读取，测试情况下不保存
    if option != "test":
        data_to_file(file_name, data)
    return data


# 获取分词字典、槽字典、意图字典
def get_info_from_training_data(data, option):
    seq_in, seq_out, intent = list(zip(*data))
    vocab = set(flatten(seq_in))

    slot_tag = set(flatten(seq_out))
    intent_tag = set(intent)
    #上两行实际没有起作用，因为所有的槽和意图都是已知的


    # 生成word2index，为每一个word进行编号
    #word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    word2index = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token] = len(word2index)


    print("***共", len(word2index), "个词（包括了<PAD> <UNK> <EOS>）***", )
    print("word2index: ", word2index, "\n")




    # 生成index2word，将字典key与value颠倒
    index2word = {v: k for k, v in word2index.items()}
    print("index2word: ", index2word)



    # 生成slot2index
    #slot2index = {'<PAD>': 0, '<UNK>': 1, "O": 2}
    #for tag in slot_tag:
        #if tag not in tag2index.keys():
            #slot2index[tag] = len(slot2index)
    slot2index = ALL_SLOT

    print("***共", len(slot2index), "个槽（包括了<PAD>）***", )
    print("slot2index: ", slot2index, "\n")




    # 生成index2slot
    index2slot = {v: k for k, v in slot2index.items()}
    print("index2slot: ", index2slot, "\n")




    # 生成intent2index
    #intent2index = {'<UNK>': 0}
    #for ii in intent_tag:
        #if ii not in intent2index.keys():
            #intent2index[ii] = len(intent2index)
    intent2index = ALL_INTENT
    print("***共", len(intent2index), "个意图（包括了<UNK>）***", )
    print("intent2index: ", intent2index, "\n")



    # 生成index2intent
    index2intent = {v: k for k, v in intent2index.items()}
    print("index2intent: ", index2intent, "\n")


    # 保存字典，不用反复计算
    if option != "test":
        path = os.path.dirname(os.path.abspath(__file__))  # path = ...\RNN-for-Joint-NLU\nlpcc
        data_to_file(path+'\\dic\\word2index.txt', word2index)
        data_to_file(path+'\\dic\\index2word.txt', index2word)
        data_to_file(path+'\\dic\\slot2index.txt', slot2index)
        data_to_file(path+'\\dic\\index2slot.txt', index2slot)
        data_to_file(path+'\\dic\\intent2index.txt', intent2index)
        data_to_file(path+'\\dic\\index2intent.txt', index2intent)


    return word2index, index2word, slot2index, index2slot, intent2index, index2intent


# 从file中读取数据（字典）
def file_to_dictionary(filename):
    print("ready to get dictionary:",filename,"............\n")
    f = open(filename, 'r', encoding='UTF-8')
    data = eval(f.read())
    f.close()
    print(data)

    return data


# 从file中读取数据（list）
def file_to_list(filename):
    print("ready to get list:",filename,"............\n")
    data = np.load(filename).tolist()
    print(data)
    return data


# 将数据（list 或 dictionary）保存在file
def data_to_file(filename, data):
    if type(data) == type({}):  #字典可以直接存储
        f = open(filename, 'w', encoding='UTF-8')
        f.write(str(data))
        f.close()
    else:                       #列表（list)使用np.save存储
        np.save(filename, np.array(data))




# 用于产生batch
def getBatch(batch_size, train_data):
    random.shuffle(train_data)      #将训练集随机排序
    sindex = 0
    eindex = batch_size
    while eindex <= len(train_data):
        batch = train_data[sindex:eindex]   #取sindex到eindex-1作为一个batch
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp

        yield batch


def to_index(train, word2index, slot2index, intent2index):
    new_train = []
    for sin, sout, intent in train:#此时数据已经加上pad

        #print("sin")
        #print(sin)

        #print("sout")
        #print(sout)

        #print("intent")
        #print(intent)

        sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                          sin))
        #将sin（汉字分词）转换为sin_ix（用数字表示分词）
        #print("sin_ix")
        #print(sin_ix)


        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i: slot2index[i] if i in slot2index else slot2index["<UNK>"],
                           sout))
        # 将sout（slot_tag如“O”、“B-singer”）转换为sout_ix（用数字表示），没有在字典里的，统一标记成<UNK>

        #print("sout_ix")
        #print(sout_ix)
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
        #print("intent_ix")
        #print(intent_ix)

        new_train.append([sin_ix, true_length, sout_ix, intent_ix])
        #new_train的每个向量依次由 分词数组、分词数目、槽数组、意图构成 共4个分量
    return new_train