import os
import re
import numpy as np
import Levenshtein
import jieba.posseg as psg
from pypinyin import pinyin, Style

path = os.path.abspath(os.path.dirname(__file__))   #path = ...\nlpcc





# slot更正模块
def slot_correct(result_file):
    # 读取槽，针对以下11种槽进行更正（一共有15种槽），实际只更正了singer和song两个槽

    '''
    age = open(path + "\\slot-dictionaries\\age.txt", 'r', encoding='UTF-8').readlines()
    age = [m[:-1] for m in age]  # 去掉'\n'，读入每一行
    custom_destination = open(path + "\\slot-dictionaries\\custom_destination.txt", 'r', encoding='UTF-8').readlines()
    custom_destination = [m[:-1] for m in custom_destination]  # 去掉'\n'，读入每一行
    emotion = open(path + "\\slot-dictionaries\\emotion.txt", 'r', encoding='UTF-8').readlines()
    emotion = [m[:-1] for m in emotion]  # 去掉'\n'，读入每一行
    instrument = open(path + "\\slot-dictionaries\\instrument.txt", 'r', encoding='UTF-8').readlines()
    instrument = [m[:-1] for m in instrument]  # 去掉'\n'，读入每一行
    language = open(path + "\\slot-dictionaries\\language.txt", 'r', encoding='UTF-8').readlines()
    language = [m[:-1] for m in language]  # 去掉'\n'，读入每一行
    scene = open(path + "\\slot-dictionaries\\scene.txt", 'r', encoding='UTF-8').readlines()
    scene = [m[:-1] for m in scene]  # 去掉'\n'，读入每一行
    singer = open(path + "\\slot-dictionaries\\singer.txt", 'r', encoding='UTF-8').readlines()
    singer = [m[:-1] for m in singer]  # 去掉'\n'，读入每一行
    song = open(path + "\\slot-dictionaries\\song.txt", 'r', encoding='UTF-8').readlines()
    song = [m[:-1] for m in song]  # 去掉'\n'，读入每一行
    style = open(path + "\\slot-dictionaries\\style.txt", 'r', encoding='UTF-8').readlines()
    style = [m[:-1] for m in style]  # 去掉'\n'，读入每一行
    theme = open(path + "\\slot-dictionaries\\theme.txt", 'r', encoding='UTF-8').readlines()
    theme = [m[:-1] for m in theme]  # 去掉'\n'，读入每一行
    toplist = open(path + "\\slot-dictionaries\\toplist.txt", 'r', encoding='UTF-8').readlines()
    toplist = [m[:-1] for m in toplist]  # 去掉'\n'，读入每一行
    # 所有字典变成数组
    dic_list = [age, custom_destination, emotion, instrument, language,
                scene, singer, song, style, theme, toplist]
    np.save(path + "\\slot-dictionaries\\11_slot_dics", np.array(dic_list))
    '''

    dic_list = np.load(path + "\\slot-dictionaries\\11_slot_dics.npy").tolist()
    slot_category = ["age", "custom_destination", "emotion", "instrument", "language",
                     "scene", "singer", "song", "style", "theme", "toplist"]


    # 读取训练结果
    data = open(result_file, 'r', encoding='UTF-8').readlines()
    data = [m[:-1] for m in data]  # 去掉'\n'，读入每一行
    data = [[t.split("\t")[0],  # 第一部分 数字不变
             t.split("\t")[1],  # 第二部分 分出中文字、英文、数字
             t.split("\t")[2],  # 第三部分 意图
             t.split("\t")[3]]  # 第四部分 序列（未标注）
            for t in data]


    for j in range(len(data)):
        for i in range(len(slot_category)):
            if not(slot_category[i]=="singer" or slot_category[i]=="song"):
                continue
            # 匹配 <slot>...</slot>形式的字符串
            slot_list = re.findall("[<]"+slot_category[i]+"[>].*?[<]/"+slot_category[i]+"[>]", data[j][3])  # 提取尖括号中内容
            # print(slot_list)
            real_slot_list =[]          # 存储了一个一个的槽["xx","xxx"....]
            correct_slot_list = []      # 存储了一个一个的更正后的槽["xx","xxx"....]
            for str in slot_list:       # 取出真正的槽值
                index_start = str.index(">")
                index_end = str.index("</")
                real_slot_list.append(str [index_start+1 : index_end])
                                        # 槽值是否在dic中
            for k in range(len(real_slot_list)):
                correct_slot = correct(real_slot_list[k], dic_list[i]) #进行槽值更正
                if correct_slot != real_slot_list[k]:
                    print(data[j][3])
                    data[j][3] = data[j][3].replace(real_slot_list[k], real_slot_list[k]+"||"+correct_slot)
                    print(data[j][3])


    fp = open(path + "\\result\\correct_result.txt", 'w', encoding='UTF-8')

    for i in range(len(data)):
        fp.write(data[i][0])  # 写语句数字编号（如：188126）
        fp.write("\t")

        fp.write(data[i][1])

        fp.write("\t")

        fp.write(data[i][2])

        fp.write("\t")  # 此处用空tab隔开
        fp.write(data[i][3])  # 最后写 意图（如：OTHERS）

        fp.write("\n")

    fp.close()


# 部分语句采用基于规则的方法
def rule_based(result_file):
    # 读取训练结果
    data = open(result_file, 'r', encoding='UTF-8').readlines()
    data = [m[:-1] for m in data]  # 去掉'\n'，读入每一行

    data = [[t.split("\t")[0],  # 第一部分 数字不变
             t.split("\t")[1],  # 第二部分 分出中文字、英文、数字
             t.split("\t")[2],  # 第三部分 意图
             t.split("\t")[3]]  # 第四部分 序列（未标注）
            for t in data]

    num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    call_token =["呼叫", "拨打", "电话", "打给"]  # 含有打电话含义的字符串
    cancel_token = ["取消", "退出", "放弃", "结束", "不用", "没需要", "停止", "关闭"]
    start_token = ["开始", "继续", "恢复", "切换"]

    slot2index = file_to_dictionary(path + "\\dic\\slot2index.txt")
    dic_list = np.load(path + "\\slot-dictionaries\\11_slot_dics.npy").tolist()
    slot_category = ["age", "custom_destination", "emotion", "instrument", "language",
                     "scene", "singer", "song", "style", "theme", "toplist"]
    navigation_open = ["打开导航", "显示导航", "我要导航", "开启导航", "给我导航", "帮我导航", "导航过去"]

    # 读取姓氏表
    family_name = open(path + "\\slot-dictionaries\\family_name.txt", 'r', encoding='UTF-8').readlines()
    family_name = [m[:-1] for m in family_name]  # 去掉'\n'，读入每一行
    for i in range(len(data)):  # 采取基于规则的方法
        if (data[i][1] in  navigation_open):
            data[i][2] = "navigation.open"
            data[i][3] = data[i][1]  # 该意图无槽

        # 开始导航、继续导航、恢复导航、切换导航    ==>navigation.start_navigation
        elif token_in_str(start_token, data[i][1]) and ("导航" in data[i][1]):
            data[i][2] = "navigation.start_navigation"
            data[i][3] = data[i][1]     # 该意图无槽

        # token 通话、token 电话     ==>phone_call.cancel
        elif token_in_str(cancel_token, data[i][1]) and (("通话" in data[i][1]) or ("电话" in data[i][1])):
            data[i][2] = "phone_call.cancel"
            data[i][3] = data[i][1]  # 该意图无槽

        elif data[i][1] in cancel_token:        # 只有取消、退出等动词
            data[i][3] = data[i][1]  # 该意图无槽
            session = data[i][0]
            if session != data[i-1][0]:         # 是对话开头
                data[i][2] = "OTHERS"
            else:
                data[i][2] = "OTHERS"
                j = i
                while True:
                    j = j - 1
                    if data[j][0] != session:
                        break
                    else:
                        if "navigation" in data[j][2]:
                            data[i][2] = "navigation.cancel_navigation"

                        elif "music" in data[j][2]:
                            data[i][2] = "music.pause"

                        elif "phone_call" in data[j][2]:
                            data[i][2] = "phone_call.cancel"

        elif ("上一首" in data[i][1]) or ("上一曲" in data[i][1]):
            # 针对music.prev采取基于规则的方法
            data[i][2] = "music.prev"
            data[i][3] = data[i][1]     # 该意图无槽

        elif(token_in_str(num, data[i][1])): # 针对电话号码采用基于规则的方法
            char_list = seg_char(data[i][1])    # 先分字
            # 11位数字单独出现标记成电话号码
            if (len(char_list) == 1) and isphonenum(char_list[0]) and len(char_list[0]) == 11:
                data[i][3] = "<phone_num>" + data[i][1] + "</phone_num>"
                data[i][2] = "phone_call.make_a_phone_call"
            # 不是11位数字单独出现必须在3位以上且有上文
            elif (len(char_list) == 1) and isphonenum(char_list[0]) and len(char_list[0]) >= 3:
                # 只出现了电话号码，查看上一轮意图 ,及电话号码长度
                if("phone_call" in data[i-1][2]):
                    data[i][3] = "<phone_num>" + data[i][1] + "</phone_num>"
                    data[i][2] = "phone_call.make_a_phone_call"

            elif token_in_str(call_token, data[i][1]):      # 出现了打电话相关字符串
                data[i][3] = ""
                for t in char_list:
                    if isphonenum(t):    # 如果是电话号码，加上槽标记
                        data[i][3] = data[i][3] + "<phone_num>" + t + "</phone_num>"
                    else:                   # 如果不是，直接复制
                        data[i][3] = data[i][3] + t
                data[i][2] = "phone_call.make_a_phone_call"

        # 长度在2、3，可能是单独出现的人名(或singer)
        elif(2 <= len(data[i][1]) <= 3):
            if data[i][1][0] in family_name:    #如果开头是姓氏，确定为人名
                if data[i-1][0] == data[i][0]:    #上文为同一个session
                    if "phone_call" in data[i-1][2]:
                    #如果上文为同一个session且为phone_call领域
                        data[i][2] = "phone_call.make_a_phone_call"
                        data[i][3] = "<contact_name>" + data[i][1] + "</contact_name>"
                    #elif ("music" in data[i-1][2]):
                     #   data[i][2] = "music.play"
                     #   data[i][3] = "<singer>" + data[i][1] + "</singer>"
                else:
                    if not(data[i][1] in dic_list[6]):
                    #如果没有上一轮，标为OTHERS
                        data[i][2] = "OTHERS"
                        data[i][3] =  data[i][1]

        if data[i][2] == "OTHERS":      #整理OTHERS类的输出
            data[i][3] = data[i][1]

        # 还可添加若intent = OTHERS 则无槽
        # 还可添加 使用 hanlp进行词性分析如果只有名词或动词...

        if data[i][2] == "navigation.navigation":        #可能有起始地和目的地2种槽
            char_list = list(psg.cut(data[i][1]))      # 分词并得到词性

            if all_n(char_list):                    # 如果整个query都是名词、量词
                if data[i][1] in dic_list[1]:
                    data[i][3] = "<custom_destination>" + data[i][1] + "</custom_destination>"
                else:
                    data[i][3] = "<destination>" + data[i][1] + "</destination>"

            elif (("从" in data[i][1]) and ("到" in data[i][1])):     # ...从xxx到xxx
                index1 = data[i][1].index("从")
                index2 = data[i][1].index("到")
                if index2<index1:
                    continue

                str1 = data[i][1][ : index1]
                str2 = data[i][1][index1+1 : index2]            # str2和str3必须都为名词量词
                str3 = data[i][1][index2+1 : ]
                if all_n(list(psg.cut(str2))) and all_n(list(psg.cut(str3))):

                    if str3 in dic_list[1]:
                        data[i][3] = str1 + "从" + "<origin>" + str2 + "</origin>" + "到" + "<custom_destination>" + str3 + "</custom_destination>"
                    else:
                        data[i][3] = str1 + "从" + "<origin>" + str2 + "</origin>" + "到" + "<destination>" + str3 + "</destination>"



            elif (("从" in data[i][1]) and ("去" in data[i][1])):  # ...从xxx去xxx
                index1 = data[i][1].index("从")
                index2 = data[i][1].index("去")
                if index2 < index1:
                    continue
                str1 = data[i][1][: index1]
                str2 = data[i][1][index1 + 1: index2]  # str2和str3必须都为名词量词
                str3 = data[i][1][index2 + 1:]
                if all_n(list(psg.cut(str2))) and all_n(list(psg.cut(str3))):
                    if str3 in dic_list[1]:
                        data[i][3] = str1 + "从" + "<origin>" + str2 + "</origin>" + "去" + "<custom_destination>" + str3 + "</custom_destination>"
                    else:
                        data[i][3] = str1 + "从" + "<origin>" + str2 + "</origin>" + "去" + "<destination>" + str3 + "</destination>"


            elif ("从" in data[i][1]) and (index != len(data[i][1]) - 1):# 如果有“...从xxx”的句式
                index = data[i][1].index("从")
                left_str = data[i][1][:index]           # “从”左侧字符串
                right_str = data[i][1][index + 1:]      # “从”右侧字符串
                if all_n(list(psg.cut(right_str))): # 右侧全为名词、量词
                    data[i][3] = left_str + "从" + "<origin>" + right_str + "</origin>"



            elif "到" in data[i][1]:  # 有"...到xxx"这样的句式
                index = data[i][1].index("到")
                left_str = data[i][1][:index]  # “到”左侧字符串
                right_str = data[i][1][index + 1:]  # “到”右侧字符串
                if index != len(data[i][1]) - 1:  # “到”不是最后一个字
                    if all_n(list(psg.cut(right_str))):  # 右侧全为名词、量词
                        if right_str in dic_list[1]:
                            right_str = "<custom_destination>" + right_str + "</custom_destination>"
                        else:
                            right_str = "<destination>" + right_str + "</destination>"
                        data[i][3] = left_str + "到" + right_str


            elif "去" in data[i][1]:             # 有"...去xxx"这样的句式
                index = data[i][1].index("去")
                left_str = data[i][1][:index]       # “去”左侧字符串
                right_str = data[i][1][index+1:]    # “去”右侧字符串

                if index!=len(data[i][1])-1:    # “去”不是最后一个字
                    if all_n(list(psg.cut(right_str))):   # 右侧全为名词、量词
                        if right_str in dic_list[1]:
                            right_str = "<custom_destination>" + right_str + "</custom_destination>"
                        else:
                            right_str = "<destination>" + right_str + "</destination>"
                        data[i][3] = left_str + "去" + right_str





    fp = open(path + "\\result\\rule_result.txt", 'w', encoding='UTF-8')

    for i in range(len(data)):
        fp.write(data[i][0])  # 写语句数字编号（如：188126）
        fp.write("\t")

        fp.write(data[i][1])

        fp.write("\t")

        fp.write(data[i][2])

        fp.write("\t")  # 此处用空tab隔开
        fp.write(data[i][3])  # 最后写 意图（如：OTHERS）

        fp.write("\n")

    fp.close()

# 判断一个字符串是不是全是数字
def isphonenum(str):
    num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for i in range(len(str)):
        if not(str[i] in num):
            return False

    return True


# token列表中的元素是否在str中
def token_in_str(token, str):
    for t in token:
        if t in str:
            return True

    return False



#把句子按字分开，不破坏英文、数字结构
def seg_char(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars


# 输入slot 和 slot字典，纠正slot
def correct(slot, slot_list):   # 在dic中
    # 计算编辑距离和拼音距离
    if slot in slot_list or len(slot) == 1 or slot == "gin" or slot == "你什么时候带我回家":
        # 槽可以在字典找到
        return slot
    edit_distance = [Levenshtein.distance(slot, t) for t in slot_list]
    pinyin_distance = [Levenshtein.distance(pinyin_str(slot), pinyin_str(t)) for t in slot_list]
    has_num, hanzi = num_in_str(slot)

    if has_num:               # 槽中有数字，先数字转汉字
        if hanzi in slot_list:
            print(slot, "改为", hanzi)
            return hanzi
        else:
            choose1 = slot_list[edit_distance.index(min(edit_distance))]
            if Levenshtein.distance(slot, choose1) <= 1:  # 拼音只相差一位
                print(slot, "改为", choose1)
                return choose1
            else:
                return slot
    else:

        choose1 = slot_list[edit_distance.index(min(edit_distance))]  # 两种距离得出的最相似槽
        choose2 = slot_list[pinyin_distance.index(min(pinyin_distance))]
        if Levenshtein.distance(slot, choose2) <= 1:    #拼音只相差一位
            print(slot, "改为", choose2)
            return choose2

        if choose1 == choose2:
            if len(choose1)<=4 or len(slot)<=4:
                #  如果长度小于4，必须拥有相同字符长度，且为一字之差
                if len(choose1) == len(slot) and Levenshtein.distance(slot, choose1) == 1:
                    print(slot, "改为", choose1)
                    return choose1
                else:
                    return slot
            else:
                # 如果长度大于都等于5
                if len(choose1) - len(slot) >= 2 or len(slot) - len(choose1) >= 2:
                    return slot
                else:
                    print(slot, "改为", choose1)
                    return choose1
        else:
            return slot



    #else:

     #   #if (min(edit_distance)>1 or )
     #   print(slot, "改为", slot_list[edit_distance.index(min(edit_distance))])
     #   return slot_list[edit_distance.index(min(edit_distance))]

# 从file中读取数据（字典）
def file_to_dictionary(filename):
    print("ready to get dictionary:",filename,"............\n")
    f = open(filename, 'r', encoding='UTF-8')
    data = eval(f.read())
    f.close()

    return data

# list中全为名词
def all_n(list):
    for t in list:
        if t.flag[0] != "n" and t.flag[0] != "m":
            return False
    return True



# 汉字转拼音
# "天空" ==>"tian1kong1"
def pinyin_str(str):
    #print(str)
    pinyin_str = ""
    for t in pinyin(str, style=Style.TONE3):
        pinyin_str = pinyin_str + t[0]
    return pinyin_str

# has_num 返回str中是否有数字(bool型)
# hanzi 返回str中数字成汉字
def num_in_str(str):
    hanzi = ""
    has_num = False
    dic = {"0":"零", "1":"一", "2":"二", "3":"三", "4":"四", "5":"五",
           "6":"六", "7":"七", "8":"八", "9":"九"}
    for t in str:
        if t in dic:
            hanzi = hanzi + dic[t]
            has_num = True
        else:
            hanzi = hanzi + t


    return has_num,hanzi





if __name__ == '__main__':


    #print(Levenshtein.distance("体面", "几面"))
    #print(pinyin_str("体面"))
    #print(pinyin_str("几面"))
    #print(Levenshtein.distance(pinyin_str("体面"), pinyin_str("几面")))

    slot_correct(path + "\\result\\rule_result.txt")
    #rule_based(path + "\\result\\answer_1025.txt")

