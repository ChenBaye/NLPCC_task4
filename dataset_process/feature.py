# 生成并存储词的特征向量（离散型）
import numpy as np
import os
import jieba.posseg as psg
import re

path = os.path.dirname(os.path.abspath(__file__))  # 上个目录 ...\\dataset_process

feature_char = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# 词性特征：52维度(包括B-pos,I-pos)
'''                                 
a 形容词 取英语形容词adjective的第1个字母。
b 区别词 取汉字“别”的声母。
c 连词 取英语连词conjunction的第1个字母。
d 副词 取adverb的第2个字母，因其第1个字母已用于形容词。
e 叹词 取英语叹词exclamation的第1个字母。
f 方位词 取汉字“方”
g 语素 绝大多数语素都能作为合成词的“词根”，取汉字“根”的声母。
h 前接成分 取英语head的第1个字母。
i 成语 取英语成语idiom的第1个字母。
j 简称略语 取汉字“简”的声母。
k 后接成分
l 习用语 习用语尚未成为成语，有点“临时性”，取“临”的声母。
m 数词 取英语numeral的第3个字母，n，u已有他用。
n 名词 取英语名词noun的第1个字母。
o 拟声词 取英语拟声词onomatopoeia的第1个字母。
p 介词 取英语介词prepositional的第1个字母。
q 量词 取英语quantit的第1个字母。
r 代词 取英语代词pronoun的第2个字母,因p已用于介词。
s 处所词 取英语space的第1个字母。
t 时间词 取英语time的第1个字母。
u 助词 取英语助词auxiliary
v 动词 取英语动词verb的第一个字母。
w 标点符号
x 非语素字 非语素字只是一个符号，字母x通常用于代表未知数、符号。
y 语气词 取汉字“语”的声母。
z 状态词 取汉字“状”的声母的前一个字母。
'''

# 标注词性特征（需要使用训练集和测试集进行分词)
def post_feature(feature_size = 26,
                 word2index = eval(open(os.path.dirname(path)+"\\nlpcc\\dic\\word2index.txt", 'r', encoding='UTF-8').read())):

    featurelist = np.zeros((len(word2index), feature_size * 2)).tolist()
    # 生成[word_size, feature_size*2]全为零矩阵

    train_data = open(path + "\\without_blankline\\train_without_blankline.txt", "r", encoding='UTF-8').readlines()
    test_data = open(path + "\\without_blankline\\test_without_blankline.txt", "r", encoding='UTF-8').readlines()
    train_data = [t[:-1] for t in train_data]
    test_data = [t[:-1] for t in test_data]
    all_data = train_data + test_data  # list合并

    all_data = [
             t.split("\t")[1]   # 第二部分 取出query

            for t in all_data]

    nothing = 0
    for t in all_data:
        for m in psg.cut(t):    # m为(word,flag)
            print(m.word,"=:=",m.flag)
            #如果是数字当做一个字，只要标为B-xxx就行
            if(isnum(m.word)):
                try:
                    index_of_word =  word2index[m.word]                 # word的角标
                except:
                    nothing = nothing + 1
                    continue
                else:
                    index_of_feature =  feature_char.index(m.flag[0])       # 词性的角标,m.flag[0]为首字母
                    index_of_vector = index_of_feature * 2      # B-词性的分量位置
                    featurelist[index_of_word][index_of_vector] = 1

            #如果是英文串当做一个字，jieba分词会标记成eng，不在词性表中，最后单独标注
            elif(ischar(m.word) or m.flag == "eng"):
                continue

            # 分词可能出现空格和空字符
            elif(m.word == "" or m.word == " "):
                continue

            #如果是汉字串,一个一个汉字正常处理
            # 如“红苹果”
            # 红=B-adj , 苹=B-n ,果=I-n
            elif(is_Chinese(m.word)):
                for n in range(len(m.word)):
                    if n == 0:    #第一个应标为B-xxx
                        index_of_word = word2index[m.word[n]]  # word的角标
                        index_of_feature = feature_char.index(m.flag[0])  # 词性的角标,m.flag[0]为首字母
                        index_of_vector = index_of_feature * 2  # B-词性的分量位置
                        featurelist[index_of_word][index_of_vector] = 1
                        print(m.word[n],":",m.flag[0],index_of_vector)
                    else:       #后续应标为I-xxx
                        index_of_word = word2index[m.word[n]]  # word的角标
                        index_of_feature = feature_char.index(m.flag[0])  # 词性的角标,m.flag[0]为首字母
                        index_of_vector = index_of_feature * 2  + 1# I-词性的分量位置
                        featurelist[index_of_word][index_of_vector] = 1
                        print(m.word[n], ":", m.flag[0], index_of_vector)

    # 针对英文字符串单独标注，英文串大多为名词n
    for t in word2index:
        if (ischar(t)):
            index_of_word = word2index[t]  # word的角标
            index_of_feature = feature_char.index("n")  # 词性的角标,m.flag[0]为首字母
            index_of_vector = index_of_feature * 2        # B-词性的分量位置
            featurelist[index_of_word][index_of_vector] = 1


    return featurelist



# 标注领域特征（使用训练集和槽字典)
def domain_feature(feature_size = 32,
                 word2index = eval(open(os.path.dirname(path)+"\\nlpcc\\dic\\word2index.txt", 'r', encoding='UTF-8').read())):

    domain = {'<PAD>': 0, "O": 1, "B-song": 2, "B-singer": 3, "B-theme": 4,
                "B-style": 5, "B-age": 6, "B-toplist": 7, "B-emotion": 8, "B-language": 9, "B-instrument": 10,
                "B-scene": 11, "B-destination": 12, "B-custom_destination": 13, "B-origin": 14,
                "B-phone_num": 15, "B-contact_name": 16, "I-song": 17, "I-singer": 18, "I-theme": 19,
                "I-style": 20, "I-age": 21, "I-toplist": 22, "I-emotion": 23, "I-language": 24,
                "I-instrument": 25, "I-scene": 26, "I-destination": 27, "I-custom_destination": 28,
                "I-origin": 29, "I-phone_num": 30, "I-contact_name": 31}

    featurelist = np.zeros((len(word2index), feature_size)).tolist()
    # 生成[word_size, feature_size]全为零矩阵

    data = open(os.path.dirname(path) + "\\nlpcc\\train_test_file\\no_pad_train.txt", "r", encoding='UTF-8').readlines()
    data = [t[:-1] for t in data]  # 去掉'\n'
    # 数据的一行像这样：111196914    播/放/dj/歌/曲	O O B-theme O O    music.play
    # 分割成这样[原始句子的词，标注的序列]
    data = [[t.split("\t")[1].split("/"),
             t.split("\t")[2].split(" "),
             ] for t in
            data]

    for t in data:
        for i in range(len(t[0])):
            index_of_word = word2index[t[0][i]] #词的角标
            index_of_vector = domain[t[1][i]]   #向量置1的位置
            featurelist[index_of_word][index_of_vector] = 1
            print(t[0][i],":",index_of_vector)


    #接着处理槽字典中出现的词
    dic_list = np.load(os.path.dirname(path) + "\\nlpcc\\slot-dictionaries\\11_slot_dics.npy").tolist()
    # 读取槽字典
    slot_category = ["age", "custom_destination", "emotion", "instrument", "language",
                     "scene", "singer", "song", "style", "theme", "toplist"]
    for j in range(len(slot_category)):     # 不同槽类型遍历
        for m in dic_list[j]:               # 遍历词
            print(m, "=:=", slot_category[j])
            char_list = seg_char(m)         # 分字
            for n in range(len(char_list)):
                if char_list[n] in word2index:      # 查看字 在不在 word2index中
                    if n == 0:  # 是槽开头的字
                        label = "B-" + slot_category[j]
                    else:
                        label = "I-" + slot_category[j]

                    index_of_word = word2index[char_list[n]]
                    index_of_vector = domain[label]
                    featurelist[index_of_word][index_of_vector] = 1
                    print(char_list[n], ":", index_of_vector)
                else:
                    continue


    return featurelist









# 领域特征

# 判断一个字符串是不是全为数字
def isnum(str):
    num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for i in range(len(str)):
        if not(str[i] in num):
            return False
    return True

# 判断一个字符串是不是全为英文字母
def ischar(str):
    for i in range(len(str)):
        if not("a"<= str[i] <="z" or "A"<= str[i] <="Z"):
            return False
    return True

#判断是否全为汉字
def is_Chinese(str):
    for ch in str:
        if not('\u4e00' <= ch <= '\u9fff'):
            return False
    return True

#把句子按字分开，不破坏英文、数字结构
def seg_char(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars


if __name__ == '__main__':

    post_vector = post_feature()        # 词性特征向量
    domain_vector = domain_feature()    # 领域特征向量
    combine_vector =np.hstack((post_vector, domain_vector)) # 合并特征向量
    ''' 
    print(len(combine_vector))
    print(len(combine_vector[0]))
    print(combine_vector[66])
    print(combine_vector[100][24])
    '''
    # 储存特征向
    np.save(path + "\\feature\\feature_vector", np.array(combine_vector))

    combine_vector = np.load(path + "\\feature\\feature_vector.npy").tolist()
