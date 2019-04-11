import os
import re
import numpy as np
import Levenshtein

path = os.path.abspath(os.path.dirname(__file__))   #path = ...\nlpcc

# 结合字典进行槽识别
def combine_dic(result_file):
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

    dic_slot = []  # 经过DIC预处理得到的一批槽
    intent = []  # 经过DIC预处理的到的意图
    for i in range(len(data)):
        dic_slot.append([0] * 40)       #新建Time_steps大的数组
        #temp_str = data[i][1]
        for j in range(len(slot_category)):
            value = ""
            for m in range(len(dic_list[j])):
                if m == 1:      # 不匹配custom_destination类型的槽
                    continue
                if(dic_list[j][m] in data[i][1]) and len(dic_list[j][m]) > len(value):
                    value =  dic_list[j][m]

            if (len(value)>=1 and j==6 and value!="高飞"):                        #singer
                data[i][2] = "music.play"
                print(i," ",value," ",j)
            elif (len(value)>=3 and j==7 and value!="打电话" and value!="不需要"):     #song
                data[i][2] = "music.play"
                print(i, " ", value, " ", j)
            elif (j!=6 and j!=7 and len(value)>=3):
                data[i][2] = "music.play"
                print(i, " ", value, " ", j)
            else:
                continue


            #类似于给  播放一首我们不一样 打上<song>标记
            # temp_str = "播放一首我们不一样"
            # value = "我们不一样"

            #temp_str = insert_slot(value, temp_str, slot_category[j])

        #data[i][3] = temp_str



    fp = open(path + "\\result\\dic_result.txt", 'w', encoding='UTF-8')

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

# 类似于给  播放一首我们不一样 打上<song>标记
# temp_str = "播放一首我们不一样"
# value = "我们不一样"
def insert_slot(value, temp_str, slot_str):
    # 找到value的起始、结束位置
    index_start = temp_str.index(value)
    index_end = index_start + len(value)

    return temp_str[:index_start]+"<"+slot_str+">"+temp_str[index_start:index_end]+"</"+slot_str+">"+temp_str[index_end:]



# slot更正模块
def slot_correct(result_file):
    # 读取槽，针对以下11种槽进行更正（一共有15种槽）

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

    for t in data:
        for i in range(len(slot_category)):
            if not(i==6 or i==7):
                continue
            # 匹配 <slot>...</slot>形式的字符串
            slot_list = re.findall("[<]"+slot_category[i]+"[>].*?[<]/"+slot_category[i]+"[>]", ''.join(t.split("\t")[3]))  # 提取尖括号中内容
            # print(slot_list)
            real_slot_list =[]
            for str in slot_list:       # 取出真正的槽值
                index_start = str.index(">")
                index_end = str.index("</")
                real_slot_list.append(str [index_start+1 : index_end])
                                        # 槽值是否在dic中
            for j in range(len(real_slot_list)):
                real_slot_list[j] = correct(real_slot_list[j], dic_list[i]) #进行槽值更正


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

    cancel_token = ["取消", "退出", "放弃", "结束", "不用", "没需要", "停止", "关闭"]
    start_token = ["开始", "继续", "恢复", "切换"]
    for i in range(len(data)):  # 针对"取消"采取基于规则的方法

        # 开始导航、继续导航、恢复导航、切换导航、导航    ==>navigation.start_navigation
        if (data[i][1] == "导航") or(token_in_str(start_token, data[i][1]) and ("导航" in data[i][1])):
            data[i][2] = "navigation.start_navigation"
            data[i][3] = data[i][1]     # 该意图无槽

        # token 通话、token 电话     ==>phone_call.cancel
        elif token_in_str(cancel_token, data[i][1]) and (("通话" in data[i][1]) or ("电话" in data[i][1])):
            data[i][2] = "phone_call.cancel"
            data[i][3] = data[i][1]  # 该意图无槽

        elif data[i][1] in cancel_token:
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


# token列表中的元素是否在str中
def token_in_str(token, str):
    for t in token:
        if t in str:
            return True

    return False




# 输入slot 和 slot字典，纠正slot
def correct(slot, slot_list):   # 在dic中

    if (slot in slot_list) or (len(slot) <= 2):
        return slot
    else:

        edit_distance = [Levenshtein.distance(slot, t) for t in slot_list]
        print(slot, "改为", slot_list[edit_distance.index(min(edit_distance))])
        return slot_list[edit_distance.index(min(edit_distance))]

'''
def pinyin_str(str):
    print(str)
    return ''.join(pinyin(str, style=Style.TONE3))
'''


if __name__ == '__main__':
    combine_dic(path + "\\result\\answer_0.txt")
    #slot_correct(path + "\\result\\blstm_crf_slot.txt")
    rule_based(path + "\\result\\dic_result.txt")