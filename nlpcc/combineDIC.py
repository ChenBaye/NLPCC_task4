import os
import re
import numpy as np

path = os.path.abspath(os.path.dirname(__file__))   #path = ...\nlpcc


# 结合槽字典预先进行槽识别
def combine_dic():
    slot2index = file_to_dictionary(path + "\\dic\\slot2index.txt")
    dic_list = np.load(path + "\\slot-dictionaries\\11_slot_dics.npy").tolist()
    slot_category = ["age", "custom_destination", "emotion", "instrument", "language",
                     "scene", "singer", "song", "style", "theme", "toplist"]

    # 读取测试文件
    data = open(path + "\\result\\corpus.test.nolabel.txt", 'r', encoding='UTF-8').readlines()
    data = [m[:-1] for m in data]  # 去掉'\n'，读入每一行

    data = [[t.split("\t")[0],  # 第一部分 数字
             t.split("\t")[1],  # 第二部分 序列
             seg_char(t.split("\t")[1]),  # 第三部分 分出中文字、英文、数字
             ]
            for t in data]


    dic_slot = []           # 存储经过DIC预处理得到的一批槽
    dic_intent = [0] *5350  # 存储经过DIC预处理的到的意图

    for i in range(len(data)):
        dic_slot.append([0] * 45)       #新建Time_steps大的数组
        #temp_str = data[i][1]
        for j in range(len(slot_category)):
            if j == 1:  # 不匹配custom_destination类型的槽
                continue
            value = ""
            for m in range(len(dic_list[j])):
                if(dic_list[j][m] in data[i][1]) and (len(dic_list[j][m]) > len(value)):
                    value = dic_list[j][m]

            if (len(value)>=1 and j==6 and value!="高飞"):                        # singer6
                print(i," ",value," ",j)
            elif (len(value)>=3 and j==7 and value!="打电话" and value!="不需要" and value!="80000"):     # song7
                print(i, " ", value, " ", j)
            elif (j==8 and len(value) >= 1):                        # style8
                print(i, " ", value, " ", j)
            elif (j==4 and len(value) >= 1 and value!="外国"):                        # language4
                print(i, " ", value, " ", j)
            elif (j==3 and len(value) >= 1):                        # instrument3
                print(i, " ", value, " ", j)
            elif (j==2 and len(value) >= 1):                      # emotion2
                print(i, " ", value, " ", j)
            elif (j == 9 and (len(value) >= 3)):                      # theme9
                print(i, " ", value, " ", j)
            elif (j == 10 and (len(value) >= 2)):                      # toplist10
                print(i, " ", value, " ", j)
            elif (j!=6 and j!=7 and len(value)>=3):                             #other slot
                print(i, " ", value, " ", j)
            else:
                continue

            dic_intent[i] = 1
            dic_slot[i] = change_slot(value, data[i][2], dic_slot[i], slot_category[j], slot2index)
            print(data[i][1])
            print(value)
            #print(dic_slot[i])

    np.save(path + "\\slot-dictionaries\\dic_slot", np.array(dic_slot))
    np.save(path + "\\slot-dictionaries\\dic_intent", np.array(dic_intent))





# 类似于给  播放一首我们不一样 打上<song>标记
# char_list = [播,放,一,首,我,们,不,一,样]
# value = "我们不一样"
# slot = [O,O,O........O]
# 预先识别出一部分槽
def change_slot(value, char_list, slot, slot_label, slot2index):
    value_list = seg_char(value)
    str = "".join(char_list)
    index_start = str.index(value)
    now_index = 0
    for i in range(len(char_list)):
        if now_index == index_start:
            for j in range(i, i+len(value_list)):
                if j == i:
                    slot[j] = slot2index["B-"+slot_label]
                else:
                    slot[j] = slot2index["I-" + slot_label]

            return slot
        else:
            now_index = now_index + len(char_list[i])

    return slot

# 从file中读取数据（字典）
def file_to_dictionary(filename):
    print("ready to get dictionary:",filename,"............\n")
    f = open(filename, 'r', encoding='UTF-8')
    data = eval(f.read())
    f.close()

    return data


'''
# 类似于给  播放一首我们不一样 打上<song>标记
# temp_str = "播放一首我们不一样"
# value = "我们不一样"
def insert_slot(value, temp_str, slot_str):
    # 找到value的起始、结束位置
    index_start = temp_str.index(value)
    index_end = index_start + len(value)

    return temp_str[:index_start]+"<"+slot_str+">"+temp_str[index_start:index_end]+"</"+slot_str+">"+temp_str[index_end:]
'''

#把句子按字分开，不破坏英文、数字结构
def seg_char(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars


if __name__ == '__main__':
    combine_dic()
