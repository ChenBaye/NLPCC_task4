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
from dataset_process import read_tencent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:
    def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size=16, n_layers=1, feature=False):
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.feature = feature      #是否使用特征向量
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.epoch_num = epoch_num
        self.encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size],
                                             name='encoder_inputs')
        # 每句输入的实际长度，除了padding
        self.inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='inputs_actual_length')
        self.slot_targets = tf.placeholder(tf.int32, [batch_size, input_steps],
                                              name='slot_targets')
        self.intent_targets = tf.placeholder(tf.int32, [batch_size],
                                             name='intent_targets')

    def build(self):


        self.embeddings = tf.Variable(self.load_word_embeding(option = "tencent"))
        '''
        # 拼接特征向量和词向量
        if self.feature == True:
            feature_vector = self.load_feature()
            feature_vector = tf.constant(feature_vector)
            #W = tf.Variable(tf.random_uniform([84, 100], -1, 1),
            #                dtype=tf.float32, name="slot_W")

            #b = tf.Variable(tf.zeros([100]), dtype=tf.float32)
            #self.feature = tf.add(tf.matmul(feature_vector, W), b)

            self.feature = tf.layers.dense(feature_vector, 30)
            self.embeddings = tf.concat((self.embeddings, self.feature), 1)
            self.embedding_size = 230
        '''

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        print("embeddings :",self.embeddings.shape)
        # Encoder

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

        self.slot_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.slot_size], -1, 1),
                             dtype=tf.float32, name="slot_W")
        self.slot_b = tf.Variable(tf.zeros([self.slot_size]), dtype=tf.float32, name="slot_b")


        # 求slot
        # encoder_outputs:
        # time * batch_size * (forward_hidden+backward_hidden)
        # slot.w:
        # (forward_hidden+backward_hidden) * slot_size

        encoder_outputs = tf.transpose(encoder_outputs, perm=[1, 0, 2])
        # encoder_outputs先转置=>>
        # batch_size * time * (forward_hidden+backward_hidden)
        # 把A的前两个维度展为一个维度
        temp = tf.reshape(encoder_outputs, [-1, self.hidden_size * 2])

        # 此时就可以matmul了
        slot_logits = tf.add(tf.matmul(temp, self.slot_W), self.slot_b)
        print(slot_logits.shape)


        # 定义slot标注的损失
        scores = tf.reshape(slot_logits, [-1, self.input_steps, self.slot_size])
        # #算好分数后，再重新reshape成[batchsize, timesteps, num_class]

        self.crf_params = tf.get_variable("crf", [self.slot_size, self.slot_size], dtype=tf.float32)


        log_likelihood, self.crf_params = tf.contrib.crf.crf_log_likelihood(
            scores, self.slot_targets, self.inputs_actual_length, self.crf_params)
        print("slot_targets shape: ", self.slot_targets.shape)

        decode_tags, best_score = tf.contrib.crf.crf_decode(
            scores, self.crf_params, self.inputs_actual_length)
        print("decode_tags shape: ", decode_tags.shape)
        # decode_tags大小[batch_size * slot_size]


        loss = tf.reduce_mean(-log_likelihood)
        self.loss = loss

        optimizer = tf.train.AdamOptimizer(0.001)
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        print("vars for loss function: ", self.vars)
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

        correct_prediction = tf.equal(tf.reshape(decode_tags, [-1]), tf.reshape(self.slot_targets, [-1]))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        mask = tf.sequence_mask(self.inputs_actual_length)


        self.slot = tf.transpose(decode_tags, perm=[1, 0])
        print("self.slot shape: ", self.slot.shape)

        self.mask = mask
        self.intent = tf.Variable(tf.constant(2, shape=[self.batch_size]))

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
            output_feeds = [self.train_op, self.loss, self.slot,
                            self.intent, self.mask,self.slot_W]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.inputs_actual_length: unziped[1],
                         self.slot_targets: unziped[2],
                         self.intent_targets: unziped[3]}
        if mode in ['test']:
            output_feeds = [self.slot, self.intent]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.inputs_actual_length: unziped[1]}

        results = sess.run(output_feeds, feed_dict=feed_dict)   #为tensor赋值
        return results

    # 装载词向量
    def load_word_embeding(self, option = "word2vec"):
        if option == "word2vec":
            path1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 上上个目录
            list = word_embeding.get_vector(path1 + "\\dataset_process\\word2vec\\min_count1size300")  # 生成向量
        elif option == "fasttext":
            print("get fasttext word_vector...")
            list = read_fasttext.get_vector()       #读取fastext的词向量
        elif option == "tencent":
            print("get tencent word_vector")        #读取tencent词向量
            list = read_tencent.get_vector()

        #list = (np.array(list)[:,0:100]).tolist()                        #取前100维
        return list

    # 读取特征向量
    def load_feature(self):
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 上上个目录
        combine_vector = np.load(path + "\\dataset_process\\feature\\feature_vector.npy").tolist()
        return combine_vector