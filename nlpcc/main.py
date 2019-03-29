# coding=utf-8
# @author: cer
import tensorflow as tf
from nlpcc.data import *
from nlpcc import *
# from model import Model
from nlpcc.model import Model
from nlpcc.my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
import operator
import matplotlib.pyplot as plt
import os
import copy

input_steps = 40    # 每一条数据设置为input_steps长度（input_steps个槽、词），一句最长实际上为40
embedding_size = 300 # 词向量维度
hidden_size = 100   # 隐藏层的节点数
n_layers = 2        # lstm层数
batch_size = 25     # 批大小，每次训练给神经网络喂入的数据量大小
vocab_size = 14405  # 共14405个不同词，，在编程中又加入了<PAD> <UNK> <EOS>，变成14405
slot_size = 33      # 有多少种slot_tag
intent_size = 12    # 有多少种意图
epoch_num = 50      # 将所有样本全部训练一次为一个epoch
path = os.path.abspath(os.path.dirname(__file__))   #path = ...\nlpcc



def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size, n_layers)
    model.build()
    return model



def train(is_debug=False):
    model = get_model()
    sess = tf.Session()
    if is_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())
    # print(tf.trainable_variables())

    '''
    train_data = open(path+"\\train_test_file\\train_labeled.txt", "r", encoding='UTF-8').readlines()
    test_data = open(path+"\\train_test_file\\test_labeled.txt", "r", encoding='UTF-8').readlines()

    train_data_ed = data_pipeline(train_data, path+"\\data_list\\train_list.npy", input_steps, "no_test")
    test_data_ed = data_pipeline(test_data, path+"\\data_list\\test_list.npy", input_steps, "no_test")

    all_data = train_data_ed + test_data_ed     # list合并
    # 要得到（训练集+测试集）的词集合、槽集合
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
    get_info_from_training_data(all_data, "no_test")
    print("get list.....")
    '''

    # 上一步保存后可以直接读取字典，节省时间
    train_data_ed = file_to_list(path+"\\data_list\\train_list.npy")
    test_data_ed = file_to_list(path+"\\data_list\\test_list.npy")


    word2index = file_to_dictionary(path+"\\dic\\word2index.txt")
    index2word = file_to_dictionary(path+"\\dic\\index2word.txt")
    slot2index = file_to_dictionary(path+"\\dic\\slot2index.txt")
    index2slot = file_to_dictionary(path+"\\dic\\index2slot.txt")
    intent2index = file_to_dictionary(path+"\\dic\\intent2index.txt")
    index2intent = file_to_dictionary(path+"\\dic\\index2intent.txt")



    # print("slot2index: ", slot2index)
    # print("index2slot: ", index2slot)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    P = []
    F1_MACRO = []
    P_intent = []
    P_slot = []
    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(getBatch(batch_size, index_train)):#此处已经生成了所有batch
            # 执行一个batch的训练
            _, loss, decoder_prediction, intent, mask, slot_W = model.step(sess, "train", batch)
            # if i == 0:
            #     index = 0
            #     print("training debug:")
            #     print("input:", list(zip(*batch))[0][index])
            #     print("length:", list(zip(*batch))[1][index])
            #     print("mask:", mask[index])
            #     print("target:", list(zip(*batch))[2][index])
            #     # print("decoder_targets_one_hot:")
            #     # for one in decoder_targets_one_hot[index]:
            #     #     print(" ".join(map(str, one)))
            #     print("decoder_logits: ")
            #     for one in decoder_logits[index]:
            #         print(" ".join(map(str, one)))
            #     print("slot_W:", slot_W)
            #     print("decoder_prediction:", decoder_prediction[index])
            #     print("intent:", list(zip(*batch))[3][index])
            # mean_loss += loss
            # train_loss += loss
            # if i % 10 == 0:
            #     if i > 0:
            #         mean_loss = mean_loss / 10.0
            #     print('Average train loss at epoch %d, step %d: %f' % (epoch, i, mean_loss))
            #     mean_loss = 0
        train_loss /= (i + 1)
        print("[Epoch {}] Average train loss: {}\n".format(epoch, train_loss))

        #####################################################################
        ##  运行至此，已经完成一轮（一个epoch=全体数据）的训练
        #####################################################################


        # 每训一个epoch，测试一次（针对整个测试集测试，得到F1、Accurate、P）
        Right_intent = 0        #正确识别意图
        Right_slot = 0          #正确识别槽
        sum = 0                 #测试数据总条数
        Both_right = 0          #同时正确识别意图和槽
        
        pred_intents = []
        pred_slots = []
        slot_accs = []
        intent_accs = []
        for j, batch in enumerate(getBatch(batch_size, index_test)):
            # 每次循环从标记好的测试集取出一个batch
            # decoder_prediction在此处为 slot_number X batch_size大小的矩阵，
            # 其中slot_number不同的数据结果不同
            # intent为   1 X batch_size 大小的矩阵
            decoder_prediction, intent = model.step(sess, "test", batch)

            #print(decoder_prediction)

            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            # 此处将decoder_prediction转置
            #print(decoder_prediction)

            #############################################################
            # 以下部分计算P，子任务3
            #############################################################


            for index in range(len(batch)):
                # 每个batch一般有batch_size条对话记录
                sum = sum + 1
                sen_len = batch[index][1]
                temp = 0
                # print("Input Sentence        : ", index_seq2word(batch[index][0], index2word)[:sen_len])
                # print("Slot Truth            : ", index_seq2slot(batch[index][2], index2slot)[:sen_len])
                # print("Slot Prediction       : ", index_seq2slot(decoder_prediction[index], index2slot)[:sen_len])
                # print("Intent Truth          : ", index2intent[batch[index][3]])
                # print("Intent Prediction     : ", index2intent[intent[index]])
                if operator.eq(index_seq2slot(batch[index][2], index2slot)[:sen_len],
                               index_seq2slot(decoder_prediction[index], index2slot)[:sen_len]):
                    Right_slot = Right_slot + 1
                    temp = temp + 1

                # print(batch[index][3])
                # print(intent[index])

                if operator.eq(index2intent[batch[index][3]],
                               index2intent[intent[index]]):
                    Right_intent = Right_intent + 1
                    temp = temp + 1

                if temp == 2:
                    Both_right = Both_right + 1

            #############################################################
            # 以下部分计算F1、Accurate，子任务1
            #############################################################

            pred_intents.append(intent)
            # 将各个 batch 预测出的 intent 列入 pred_intents

            # 此处因为转置decoder_prediction为 batch_size X slot_number大小的矩阵，
            slot_pred_length = list(np.shape(decoder_prediction))[1]    # 得到slot_number
            pred_padded = np.lib.pad(decoder_prediction, ((0, 0), (0, input_steps-slot_pred_length)),
                                     mode="constant", constant_values=0)
            # 将decoder_prediction的每个行向量补齐到input_steps长度得到 pred_padded
            # print("pred_padded:\n",pred_padded)

            pred_slots.append(pred_padded)

            true_slot = np.array((list(zip(*batch))[2]))
            # 提取出当前batch的slot（已经扩充到了input_steps大小）

            #print(true_slot)

            true_length = np.array((list(zip(*batch))[1]))
            # 提取出当前batch的真正slot数目，形成一个数组（如果第一条数据实际只有3个槽，则数组第一项为3）
            # print("true_length:",true_length)

            true_slot = true_slot[:, :slot_pred_length]
            # 此处将batch的slot截取成和decoder_prediction一样的长度（slot_pred_length）

            slot_acc = accuracy_score(true_slot, decoder_prediction, true_length)
            # 计算槽准确率，此时true_slot与decoder_prediction单个向量长度均为slot_pred_length
            # 并不是 input_steps
            # slot_acc与P_right_slot不同，前者为   正确的槽数/全部槽数，后者为 槽正确的句子数/全部句子数

            intent_acc = accuracy_score(list(zip(*batch))[3], intent)

            slot_accs.append(slot_acc)
            # 将各个batch的slot_acc排成数组，最终计算均值，即为整个测试集的slot_acc
            intent_accs.append(intent_acc)


        pred_intents_a = np.vstack(pred_intents)
        # print("pred_intents:",pred_intents_a)
        pred_intents_a = flatten(pred_intents_a)
        # 提前把pred_intents_a一维化
        # print("pred_intents:", pred_intents_a)
        # print(len(pred_intents_a))

        true_intents_a = np.array(list(zip(*index_test))[3])[:len(pred_intents_a)]
        #print("true_intents_a:",true_intents_a)

        pred_slots_a = np.vstack(pred_slots)
        # 之前的pred_slots将各个batch的slot隔开，现在使用vstack竖向堆叠，生成一个
        # 行：测试集条数，列：input_steps的slot向量组
        # print("pred_slots_a: ",pred_slots_a)

        true_slots_a = np.array(list(zip(*index_test))[2])[:pred_slots_a.shape[0]]
        # 取出测试集中的slot部分，pred_slots_a.shape[0]实际上就是测试集语句条数
        # print("true_slots_a: ", true_slots_a)

        # print("Intent accuracy for epoch {}: {}".format(epoch, np.average(intent_accs)))
        # 所有batch的intent_acc的平均值与下面的P_right_intent相等

        # print("Slot accuracy for epoch {}: {}".format(epoch, np.average(slot_accs)))
        # slot_acc与P_right_slot不同，前者为   正确的槽数/全部槽数，后者为 槽正确的句子数/全部句子数

        # print("Slot F1 score for epoch {}: {}".format(epoch, f1_for_sequence_batch(true_slots_a, pred_slots_a)))

        # print("Intent F1 score（含OTHERS类）for epoch {}: {}".
        #      format(epoch, f1_for_sequence_batch_new(true_intents_a, pred_intents_a,"others")))

        print("Intent F1 score（不含OTHERS类）for epoch {}: {}".
              format(epoch, f1_for_sequence_batch_new(true_intents_a, pred_intents_a, "no_others")))


        # 查准率（precision），指的是预测值为1且真实值也为1的样本在预测值为1的所有样本中所占的比例
        # 以西瓜问题为例，算法挑出来的西瓜中有多少比例是好西瓜。
        # 召回率（recall），也叫查全率，指的是预测值为1且真实值也为1的样本在真实值为1的所有样本中所占的比例。
        # 所有的好西瓜中有多少比例被算法挑了出来。
        # f1为以上二者调和平均数
        # f1_macro为计算出每一个类的Precison和Recall后计算F1，最后将F1平均。


        # 每一轮（一个epoch）的最后结果
        print("sum = ", sum)
        print("Right_slot = ", Right_slot)
        print("Right_intent = ", Right_intent)
        print("Both_right = ", Both_right)

        print("P_Right_slot = ", Right_slot / sum)
        print("P_Right_intent = ", Right_intent / sum)
        print("P_Both_right = ", Both_right / sum)

        P.append(Both_right / sum)
        F1_MACRO.append(f1_for_sequence_batch_new(true_intents_a, pred_intents_a, "no_others"))
        P_intent.append(Right_intent / sum)
        P_slot.append(Right_slot / sum)

    # 输出折线图
    output_picture(P, F1_MACRO, P_intent, P_slot)




#将结果输出为折线图
def output_picture(P, F1_MACRO, P_intent, P_slot):
    x = range(1,len(P)+1)    # x = [1, 2, ... , len(list)]
    plt.xticks(np.arange(0, len(P)+1, 10))
    plt.plot(x, P, label='P')
    plt.plot(x, F1_MACRO, label='F1_MACRO')
    plt.plot(x, P_intent, label='P_intent')
    plt.plot(x, P_slot, label='P_slot')

    # plt.legend()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
               ncol=3, mode="expand", borderaxespad=0.)

    # plt.show()
    plt.xlabel("epoch")  # 横坐标为轮数
    plt.savefig(path+"\\result\\result.jpg")

#加入基于规则的模型,针对music.prev、music.next（在训练集中，凡是这两个意图，均无槽)
# 数据格式
# index_test 每个单元 = [分词list、分词数目、槽list、意图]
# def rule_based_model(index_test, pred_intents_a, pred_slots):








# 单独用于测试分词等是否正确，没有被调用
def test_data():
    train_data = open(path+"\\train_test_file\\train_labeled.txt", "r", encoding='UTF-8').readlines()
    test_data = open(path+"\\train_test_file\\test_labeled.txt", "r", encoding='UTF-8').readlines()
    train_data_ed = data_pipeline(train_data, path+"\\data_list\\train_list.npy", input_steps, "no_test")   # 此处数据已经进行PAD
    test_data_ed = data_pipeline(test_data, path+"\\data_list\\test_list.npy", input_steps, "no_test")

    all_data = train_data_ed+test_data_ed
    # 要得到（训练集+测试集）的词集合、槽集合

    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(all_data, "no_test")
    # print("slot2index: ", slot2index)
    # print("index2slot: ", index2slot)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)
    batch = next(getBatch(batch_size, index_test))  #取出一个batch
    unziped = list(zip(*batch))

    print("word num: ", len(word2index.keys()), "slot num: ", len(slot2index.keys()), "intent num: ",
          len(intent2index.keys()))
    print(np.shape(unziped[0]), np.shape(unziped[1]), np.shape(unziped[2]), np.shape(unziped[3]))
    print(np.transpose(unziped[0], [1, 0]))
    print(unziped[1])
    print(np.shape(list(zip(*index_test))[2]))


if __name__ == '__main__':
    #train(is_debug=True)
    #test_data()
    train()

