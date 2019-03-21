# coding=utf-8
# @author: cer

import random
import numpy as np

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


def data_pipeline(data, length=50):     # 规定语句长度定为 50=input_steps ，不足用EOS+PAD补上
    data = [t[:-1] for t in data]  # 去掉'\n'
    # 数据的一行像这样：'BOS i want to fly from baltimore to dallas round trip EOS
    # \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # 分割成这样[原始句子的词，标注的序列，intent]
    print(data[0])
    print(data[1])
    data = [[t.split("\t")[1].split(" "), t.split("\t")[2].split(" ")[:-1], t.split("\t")[2].split(" ")[-1]] for t in
            data]
    #按tab分割，导致标注序列和intent在同一块，使用[-1:]和[-1]分开
    print(data[0])
    print(data[1])


    #data = [[t[0][1:-1], t[1][1:], t[2]] for t in data]  # 将BOS和EOS去掉，并去掉对应标注序列中相应的标注
    seq_in, seq_out, intent = list(zip(*data))
    sin = []
    sout = []
    # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
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
    return data


def get_info_from_training_data(data):
    seq_in, seq_out, intent = list(zip(*data))
    vocab = set(flatten(seq_in))
    print("vocab\n")
    print(vocab)

    slot_tag = set(flatten(seq_out))

    print("slot_tag\n")
    print(slot_tag)

    intent_tag = set(intent)

    print("intent_tag\n")
    print(intent_tag)

    # 生成word2index，为每一个word进行编号
    word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token] = len(word2index)

    print("word2index\n")
    print(word2index)
    # 生成index2word，将字典key与value颠倒
    index2word = {v: k for k, v in word2index.items()}

    print("index2word\n")
    print(index2word)

    # 生成tag2index
    #tag2index = {'<PAD>': 0, '<UNK>': 1, "O": 2}
    #for tag in slot_tag:
        #if tag not in tag2index.keys():
            #tag2index[tag] = len(tag2index)
    tag2index = ALL_SLOT
    print("tag2index\n")
    print(tag2index)


    # 生成index2tag
    index2tag = {v: k for k, v in tag2index.items()}
    print("index2tag\n")
    print(index2tag)

    # 生成intent2index
    #intent2index = {'<UNK>': 0}
    #for ii in intent_tag:
        #if ii not in intent2index.keys():
            #intent2index[ii] = len(intent2index)
    intent2index = ALL_INTENT
    print("intent2index\n")
    print(intent2index)

    # 生成index2intent
    index2intent = {v: k for k, v in intent2index.items()}

    print("index2intent\n")
    print(index2intent)

    return word2index, index2word, tag2index, index2tag, intent2index, index2intent

#用于产生batch
def getBatch(batch_size, train_data):
    random.shuffle(train_data)      #将训练集随机排序
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]   #取sindex到eindex作为一个batch
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
        # 将sout（slot_tag如“O”、“B-singer”）转换为sout_ix（用数字表示）

        #print("sout_ix")
        #print(sout_ix)
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
        #print("intent_ix")
        #print(intent_ix)

        new_train.append([sin_ix, true_length, sout_ix, intent_ix])
        #new_train的每个向量依次由 分词数组、分词数目、槽数组、意图构成 共4个分量
    return new_train