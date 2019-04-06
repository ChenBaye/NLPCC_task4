# coding=utf-8
# @author: cer
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys
import os
from dataset_process import word_embeding
from dataset_process import read_fasttext
import random
from dataset_process.sentence_embeding import *
from nlpcc.my_metrics import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维


class sentence_Model:
    def __init__(self, input_steps, embedding_size, hidden_size, vocab_size,
                 intent_size, epoch_num, batch_size=16, n_layers=1):
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.intent_size = intent_size
        self.epoch_num = epoch_num
        self.encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size],
                                             name='encoder_inputs')
        # 每session输入的实际长度，除了padding
        self.inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='inputs_actual_length')
        self.intent_targets = tf.placeholder(tf.int32, [batch_size, input_steps],
                                              name='slot_targets')

    def build(self):
        # 加载句子向量

        self.embeddings = tf.Variable(self.load_sentence_embeding())
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        # 使用单个LSTM cell
        encoder_f_cell_0 = LSTMCell(self.hidden_size)
        encoder_b_cell_0 = LSTMCell(self.hidden_size)
        encoder_f_cell = DropoutWrapper(encoder_f_cell_0,output_keep_prob=0.5)
        encoder_b_cell = DropoutWrapper(encoder_b_cell_0,output_keep_prob=0.5)
        #防止过拟合

        # encoder_inputs_time_major = tf.transpose(self.encoder_inputs_embedded, perm=[1, 0, 2])
        # T代表时间序列的长度，B代表batch size，D代表隐藏层的维度
        # 下面四个变量的尺寸：T*B*D，T*B*D，B*D，B*D
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,     #前向的lstm cell数目
                                            cell_bw=encoder_b_cell,     #后向的lstm cell数目
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.inputs_actual_length,
                                            dtype=tf.float32, time_major=True)
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        self.encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )#取出最后的状态
        print("encoder_outputs: ", encoder_outputs)
        print("encoder_outputs[0]: ", encoder_outputs[0])
        print("encoder_final_state_c: ", encoder_final_state_c)
        print("encoder_final_state_h: ", encoder_final_state_h)

        self.intent_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.intent_size], -1, 1),
                             dtype=tf.float32, name="slot_W")
        self.intent_b = tf.Variable(tf.zeros([self.intent_size]), dtype=tf.float32, name="slot_b")

        # 求intent_W
        # encoder_outputs:
        # time * batch_size * (forward_hidden+backward_hidden)
        # intent.w:
        # (forward_hidden+backward_hidden) * intent_size

        print("encoder_outputs shape", encoder_outputs.shape)
        encoder_outputs = tf.transpose(encoder_outputs, perm=[1, 0, 2])
        # encoder_outputs先转置=>>
        # batch_size * time * (forward_hidden+backward_hidden)
        # 把A的前两个维度展为一个维度
        temp = tf.reshape(encoder_outputs, [-1, self.hidden_size * 2])

        # 此时就可以matmul了
        intent_logits = tf.add(tf.matmul(temp, self.intent_W), self.intent_b)
        print("intent_logits shape: ", intent_logits.shape)


        # 定义intent标注的损失
        scores = tf.reshape(intent_logits, [-1, self.input_steps, self.intent_size])
        # #算好分数后，再重新reshape成[batchsize, timesteps, num_class]
        print("scores shape: ", scores.shape)


        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            scores, self.intent_targets, self.inputs_actual_length)

        print("intent_targets shape: ", self.intent_targets.shape)

        loss = tf.reduce_mean(-log_likelihood)
        self.loss = loss

        optimizer = tf.train.AdamOptimizer(0.0001)
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        print("vars for loss function: ", self.vars)
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))
        print("******************************")


        decode_tags, best_score  = tf.contrib.crf.crf_decode(
            scores, transition_params, self.inputs_actual_length)
        # decode_tags大小[batch_size * input_steps]
        print("decode_tags shape: ", decode_tags.shape)


        correct_prediction = tf.equal(tf.reshape(decode_tags, [-1]), tf.reshape(self.intent_targets, [-1]))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        mask = tf.sequence_mask(self.inputs_actual_length)


        self.intent = tf.transpose(decode_tags, perm=[1, 0])
        print("self.intent shape: ", self.intent.shape)

        self.mask = mask

        print("loss: ", loss)
        print("acc: ", accuracy)


    def step(self, sess, mode, trarin_batch):
        """ perform each batch"""
        if mode not in ['train', 'test']:
            print >> sys.stderr, 'mode is not supported'
            sys.exit(1)
        unziped = list(zip(*trarin_batch))
        # print(np.shape(unziped[0]), np.shape(unziped[1]),
        #       np.shape(unziped[2]), np.shape(unziped[3]))
        if mode == 'train':
            output_feeds = [self.train_op, self.loss,
                            self.intent, self.mask]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.inputs_actual_length: unziped[1],
                         self.intent_targets: unziped[2]
                         }
        if mode in ['test']:
            output_feeds = [self.intent, self.mask]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.inputs_actual_length: unziped[1]}

        results = sess.run(output_feeds, feed_dict=feed_dict)   #为tensor赋值
        return results

    # 装载词向量
    def load_sentence_embeding(self, option = "sentence2vec"):
        if option == "sentence2vec":
            path1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 上上个目录
            list = get_vector(path1 + "\\dataset_process\\sentence2vec\\min_count1size300")  # 生成向量

        else:
            print("get fasttext word_vector...")
            list = read_fasttext.get_vector()       #读取fastext的词向量
        return list


if __name__ == '__main__':
    input_steps = 30  # 每一条数据设置为input_steps长度（input_steps个句）
    embedding_size = 300  # 词向量维度
    hidden_size = 100  # 隐藏层的节点数
    n_layers = 2  # lstm层数
    batch_size = 25  # 批大小，每次训练给神经网络喂入的数据量大小
    vocab_size = 17902  # 17902个句子
    slot_size = 33  # 有多少种slot_tag
    intent_size = 12  # 有多少种意图
    epoch_num = 50  # 将所有样本全部训练一次为一个epoch


    # model = sentence_Model(input_steps, embedding_size, hidden_size, vocab_size,
    #              intent_size, epoch_num, batch_size, n_layers)
    model = sentence_Model(input_steps, embedding_size, hidden_size, vocab_size,intent_size, epoch_num, batch_size, n_layers)
    model.build()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    path = os.path.abspath(os.path.dirname(__file__))  # path = ...\nlpcc
    intent2index = file_to_dictionary(path + "\\dic\\intent2index.txt")
    index2intent = file_to_dictionary(path + "\\dic\\index2intent.txt")

    sentence2index = file_to_dictionary(path + "\\dic\\sentence2index.txt")
    index2sentence = file_to_dictionary(path + "\\dic\\index2sentence.txt")

    train_data_ed = file_to_list(path + "\\data_list\\train_sentence_list.npy")
    test_data_ed = file_to_list(path + "\\data_list\\test_sentence_list.npy")

    index_train = to_index(train_data_ed, sentence2index, intent2index)
    index_test = to_index(test_data_ed, sentence2index, intent2index)

    for epoch in range(epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        for i, batch in enumerate(getBatch(batch_size, index_train)):  # 此处已经生成了所有batch
            # 执行一个batch的训练
            _, loss, decoder_prediction, mask = model.step(sess, "train", batch)

        train_loss /= (i + 1)
        print("[Epoch {}] Average train loss: {}\n".format(epoch, loss))

        #####################################################################
        ##  运行至此，已经完成一轮（一个epoch=全体数据）的训练
        #####################################################################

        # 每训一个epoch，测试一次（针对整个测试集测试，得到F1、Accurate、P）
        Right_intent = 0  # 正确识别意图


        pred_intents = []
        true_intents = []
        for j, batch in enumerate(getBatch(batch_size, index_test, "test")):

            decoder_prediction, mask = model.step(sess, "test", batch)
            #print(decoder_prediction)



            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
            # batch_size * input_steps

            true_length = np.array((list(zip(*batch))[1]))
            # 提取出当前batch的真正intent数目，形成一个数组（如果第一条数据实际只有3个槽，则数组第一项为3）
            # print("true_length:",true_length)

            batch_true_intents = np.array(list(zip(*batch))[2])

            #删除pad
            for i in range(len(true_length)):
                temp = (decoder_prediction[i])[:true_length[i]]
                temp1 = (batch_true_intents[i])[:true_length[i]]
                pred_intents.append(temp)
                true_intents.append(temp1)



        pred_intents_a = flatten(pred_intents)
        print("共测试句子：",len(pred_intents_a))

        true_intents_a = flatten(true_intents)

        print("intent accu: ",accuracy_score(true_intents_a, pred_intents_a))

