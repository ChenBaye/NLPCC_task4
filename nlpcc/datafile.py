# _*_coding:utf-8_*_
#用于对初始数据进行处理，便于模型使用
import jieba
import re
import random

#只有如下槽
slot=["song","singer","theme","style","age","toplist","emotion","language","instrument","scene",
      "destination","custom_destination","phone_num","contact_name"]
#只有如下意图
intent=["music.play", "music.pause", "music.prev", "music.next", "navigation.navigation"
    , "navigation.open", "navigation.start_navigation", "navigation.cancel_navigation",
        "phone_call.make_a_phone_call", "phone_call.cancel"]

train_data = open("corpus.train.txt", "r",encoding='UTF-8').readlines()
test_data = open("corpus.test.nolabel.txt", "r",encoding='UTF-8').readlines()



def getslot(list):#得到未处理序列中的槽 list
   slot_list=re.findall("[<].*?[>]",list)   #提取尖括号中内容
   temp_list=[]
   #print(slot_list)

   if len(slot_list) != 0:  #可能无槽
        for i in range(len(slot_list)):          #删除重复及尖括号
            slot_list[i] = slot_list[i].replace("<", "")
            slot_list[i] = slot_list[i].replace(">", "")
            if slot_list[i][0]!='/':
                temp_list.append(slot_list[i])



   #print(temp_list)
   return temp_list                         #返回无重复的slot







def data_handle(data):#
    data = [t[:-1] for t in data]  # 去掉'\n'，读入每一行


    #原始数据：117194488	来一首周华健的花心	music.play	来一首<singer>周华健</singer>的<song>花心</song>
    #处理后：117194488	来 一首 周华健 的 花心	music.play	O O B-SINGER O B-SONG I-SONG

    data = [[t.split("\t")[0],      #第一部分 数字不变
             (" ".join(jieba.cut(t.split("\t")[1], HMM=True))).split(" "),#第二部分 jieba分词
             t.split("\t")[2],      #第三部分 意图
             t.split("\t")[3]]      #第四部分 序列（未标注）
    for t in data]

    for t in data:
        print(t)

    #下面完成序列标注

    for i in range(len(data)):
        word_num=len(data[i][1])    #每句话多少个词
        list = ["O"] * word_num     #初始化标记序列
        data[i][3] = data[i][3].replace("<","")
        data[i][3] = data[i][3].replace(">", "")
        data[i][3] = data[i][3].replace("/", "")
        #print(data[i][3])
        for j in range(len(slot)):    #找出slot位置
            if data[i][3].find(slot[j])!=-1:    #找出了对应slot
                position=[i.start() for i in re.finditer(slot[j], data[i][3])]
                #print(position)
                for k in range(len(data[i][1])):    #遍历之前的分词是否在slot位置
                    word_position=data[i][3].find(data[i][1][k]) #分词的位置
                    for m in range(0,len(position)-1,2):  #遍历槽的左右边界，槽是一对边界，所以加2
                        if word_position > position[m] and word_position < position[m+1]:  #夹在边界之间
                            if k==0:                #第一个词 必为B-slot
                                list[k]="B-"+slot[j]
                            elif list[k-1] != ("B-"+slot[j]) and list[k-1] != ("I-"+slot[j]):
                                list[k] = "B-" + slot[j]
                            else:
                                list[k] = "I-" + slot[j]


        data[i][3]=list
        print(data[i])

    #写入文件
    fp = open("train2.txt", 'w',encoding='UTF-8')
    for i in range(len(data)):
        fp.write(data[i][0])
        fp.write("\t")
        for j in range(len(data[i][1])):
            fp.write(data[i][1][j])
            if j!=(len(data[i][1])-1):
                fp.write(" ")

        fp.write("\t")


        for k in range(len(data[i][3])):
            fp.write(data[i][3][k])
            if k!=(len(data[i][3])-1):
                fp.write(" ")


        fp.write(" ")       #此处用空格隔开而而非tab
        fp.write(data[i][2])

        fp.write("\n")

    fp.close()

    data = open("train2.txt", "r", encoding='UTF-8').readlines()
    fp1 = open("testfile.txt", 'w', encoding='UTF-8')
    fp2 = open("trainfile.txt", 'w', encoding='UTF-8')
    #抽取10%作为测试集,剩余90%作为训练集
    for t in data:
        if random.uniform(0, 100) <= 10:
            fp1.write(t)
        else:
            fp2.write(t)

    fp1.close()
    fp2.close()










if __name__ == '__main__':
    train_data = open("train1.txt", "r", encoding='UTF-8').readlines()
    test_data = open("corpus.test.nolabel.txt", "r", encoding='UTF-8').readlines()

    data_handle(train_data)

