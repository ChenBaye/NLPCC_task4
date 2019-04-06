from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import random
import numpy as np
from collections import Counter

#本文件作用
#生成句子向量
#生成句子-编号字典
#生成编号-句子字典
#对会话长度进行pad
#生成训练文件（未经过数字化）
#生成测试文件（未经过数字化）
path = os.path.dirname(os.path.abspath(__file__))  # 上个目录 ...\\dataset_process

flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维

def save_all_sentences(filename1, filename2, option = "no_pad"):
    data1 = open(filename1, "r", encoding='UTF-8').readlines()
    data1 = [t[:-1] for t in data1]  # 去除空行
    data_sentences1 = [[t.split("\t")[0], t.split("\t")[1].split("/"), t.split("\t")[3].split("/")] for t in data1]
    # data_sentences=[
    #   [116195744,	['退','出','音','乐'], 'OTHERS'],
    #   [......],
    #   ]

    data2 = open(filename2, "r", encoding='UTF-8').readlines()
    data2 = [t[:-1] for t in data2]  # 去除空行
    data_sentences2 = [[t.split("\t")[0], t.split("\t")[1].split("/"), t.split("\t")[3]] for t in data2]
    # data_sentences=[
    #   [116195744,	['退','出','音','乐'], 'OTHERS'],
    #   [......],
    #   ]

    data_sentences = data_sentences1 + data_sentences2

    path = os.path.dirname(os.path.abspath(__file__))  # 上个目录
    fp = open(path + "\\query\\"+option+"_sentences.txt", 'w', encoding='UTF-8')
    for i in range(len(data_sentences)):
        fp.write(data_sentences[i][0])  # 写语句数字编号（如：188126）
        fp.write("\t")
        for j in range(len(data_sentences[i][1])):  # 这个for循环用于写分好词的语句（如：播放/林忆莲/的/伤痕）
            fp.write(data_sentences[i][1][j])
            if j != (len(data_sentences[i][1]) - 1):  # 末尾不加"/"
                fp.write("/")
        fp.write("\n")

    print("save sentences over......")

    return data_sentences


def generate_model(min_count, window, size, all_sentences):
    documents = [TaggedDocument(t[1], [i]) for i, t in enumerate(all_sentences)]
    model = Doc2Vec(documents, vector_size=size, window=window, min_count=min_count)


    path = os.path.dirname(os.path.abspath(__file__))  # 上上个目录
    model.save(path + "\\sentence2vec\\min_count" + str(min_count) + "size" + str(size))

   # print(len(model['浅/浅'.split('/')]))
    #print(model.infer_vector(['浅','浅']))
    #print(len(model.infer_vector(['浅', '浅'])))


    print("sentences2vec over......")


#产生训练文件、测试文件
def generate_datafile(inputname, resultname, length = 30):  # 一个session最多有29轮,留一个空给<EOS>
                                                            # 因此变成30
    # data_list 的每一个元素= [
    #   [turn1, turn2, ......],
    #   [intent1, intent2, .....]
    # ]
    data_list = []

    data = open(inputname, "r", encoding='UTF-8').readlines()
    data = [t[:-1] for t in data]  # 去除空行

    data_sentences = [[t.split("\t")[0], "/".join(t.split("\t")[1].split("/")), t.split("\t")[3]] for t in data]
    # data_sentences=[
    #   [116195744,	'退/出/音/乐', 'OTHERS'],
    #   [......],
    #   ]

    session_id, sentence, intent = list(zip(*data_sentences))        #取出三部分
    print("共有", len(set(session_id)), "个session_id")

    old_id = session_id[0]
    sentences = []
    intents = []
    for m in range(len(session_id)):
        if not(old_id == session_id[m]):        # 属于同一个session，则收入同一个list
            data_list.append([sentences,intents])
            sentences = []
            intents = []
            old_id = session_id[m]

        sentences.append(sentence[m])
        intents.append(intent[m])

    data_list.append([sentences, intents])  # 加上最后一个session
    # data_list 的每一个元素= [
    #   [sentence1, sentence2, ......],
    #   [intent1, intent2, .....]
    # ]
    # sentence 用"/"分割

    print("PAD之前",data_list[0])
    # 接下来进行pad，将不满length长的session补齐<EOS>+n*<PAD>到length
    seq_in, seq_out = list(zip(*data_list))
    sin = []
    sout = []

    # 统计句子中分词的最大数目以便确定input_size
    max = len(seq_in[0])
    for i in seq_in:
        if len(i) > max:
            max = len(i)
    print("一个session最多有：", max, "个sentence")

    # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        # print(i)
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]  # 如果长度>=length，则截取length长度，并在最后补上<EOS>
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
        data = list(zip(sin, sout))

    print("结尾补上<EOS>+N*<PAD>之后: \n", data[0])
    print("\n")

    data_to_file(resultname, data)          #存储训练、测试数据
    return data

#将存储训练、测试数据变成数字形式，之后进行训练
def to_index(data, sentence2index, intent2index):
    # data 的每一个元素= [
    #   [sentence1, sentence2, ...... '<EOS>', '<PAD>'],
    #   [intent1, intent2, ..... intent30]
    # ]
    # sentence 用"/"分割

    new_train = []
    for sin, sout in data:  # 此时数据已经加上pad

        sin_ix = list(map(lambda i: sentence2index[i] if i in sentence2index else sentence2index["<UNK>"],
                          sin))

        true_length = sin.index("<EOS>")

        sout_ix = list(map(lambda i: intent2index[i] if i in intent2index else intent2index["<UNK>"],
                           sout))

        new_train.append([sin_ix, true_length, sout_ix])
        # 最终每个单元结构为[句子编号list，句子实际长度list，意图list]

    return new_train


# 用于产生batch
def getBatch(batch_size, train_data, option="train"):
    if option == "train":
        random.shuffle(train_data)  # 将训练集随机排序
    sindex = 0
    eindex = batch_size
    while eindex <= len(train_data):
        batch = train_data[sindex:eindex]  # 取sindex到eindex-1作为一个batch
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp

        yield batch







def generate_dic(all_sentences):
    # all_sentences=[
    #   [116195744,	['退','出','音','乐'], 'OTHERS'],
    #   [......],
    #   ]
    session_id, sentence, intent = list(zip(*all_sentences))
    print("共有", len(set(session_id)), "个session_id")


    most_session = Counter(session_id).most_common(1)
    print("session轮数最多的是session: ", most_session)


    # 提取list的三部分
    sentence_char = []
    for t in sentence:
        sentence_char.append("/".join(t))               #将list状态的['退','出','音','乐']
                                                        # 变回 退/出/音/乐

    sentence_vocab = set(sentence_char)                 #包括所有sentence的集合
    print("共有",len(sentence_vocab),"个sentence")

    sentence2index = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}   #为每一个sentence指定一个list
    for token in sentence_vocab:
        if token not in sentence2index.keys():
            sentence2index[token] = len(sentence2index)

    print("***共", len(sentence2index), "个句子（包括了<PAD> <UNK> <EOS>）***", )
    print("sentence2index: ", sentence2index, "\n")

    index2sentence = {v: k for k, v in sentence2index.items()}
    print("index2sentence: ", index2sentence)

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 上上个目录
    data_to_file(path+"\\nlpcc\\dic\\sentence2index.txt",sentence2index)
    data_to_file(path+"\\nlpcc\\dic\\index2sentence.txt",index2sentence)



# 从file中读取数据（dic）
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


# 读取模型，并返回一个句向量list， 与sentence2index字典对应
def get_vector(modelname, sentence2index = eval(open(os.path.dirname(path)+"\\nlpcc\\dic\\sentence2index.txt", 'r', encoding='UTF-8').read())):
    model = Doc2Vec.load(modelname)
    sentence_number = len(sentence2index)   # 共有多少个句子
    vec_size = model.vector_size    # 每个句向量的维度
    sentence_vector = []                # 存储句向量


    # 先随机生成一个句向量列表
    for i in range(sentence_number):
        sentence_vector.append(random_list(vec_size))

    print(len(sentence_vector))
    nothing = 0
    #for key in sentence2index:
    #    try:
    #       vector = model.infer_vector([key.split("/")])
    #       sentence_vector[sentence2index[key]] = vector
    #    except: # 发生异常什么都不做
    #        nothing = nothing + 1
    #    else:
    #        continue
    print("缺少",nothing,"个词向量")
    print(len(sentence_vector))



    return sentence_vector



if __name__ == '__main__':
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 上上个目录
    all_sentences = save_all_sentences(path+"\\nlpcc\\train_test_file\\no_pad_train.txt",
                       path + "\\nlpcc\\train_test_file\\test_labeled.txt",
                       option="no_pad")
    generate_model(1, 5, 300, all_sentences)

    #generate_dic(all_sentences)

    #生成训练文件
    #generate_datafile(inputname=path+"\\nlpcc\\train_test_file\\no_pad_train.txt",
     #                 resultname=path+"\\nlpcc\\data_list\\train_sentence_list.npy")
    #生成测试文件
    #generate_datafile(inputname=path + "\\nlpcc\\train_test_file\\test_labeled.txt",
     #                 resultname=path+"\\nlpcc\\data_list\\test_sentence_list.npy")

    get_vector(path+"\\dataset_process\\sentence2vec\\min_count1size300")