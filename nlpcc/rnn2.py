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
                 intent_size, epoch_num, batch_size=16, n_layers=1):
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
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

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def build(self):

        self.embeddings = tf.Variable(self.load_word_embeding(option = "tencent"))

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)


        with tf.name_scope('cell'):

            cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=n),
                                                   output_keep_prob=0.5) for n in [self.hidden_size, self.hidden_size]]
            # self.hidden_size = 隐藏层层数
            # num_units 输出向量维度
            Cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


        with tf.name_scope('rnn'):
            # hidden一层 输入是[seq_length ,batch_size, embendding_dim]
            # hidden二层 输入是[seq_length ,batch_size, 2*hidden_dim]
            # 2*hidden_dim = embendding_dim + hidden_dim
            _, final_states = tf.nn.dynamic_rnn(cell=Cell,
                                          inputs=self.encoder_inputs_embedded,
                                          sequence_length=self.inputs_actual_length,
                                          dtype=tf.float32, time_major=True)
            # final_states = [(c,h)...(c,h)]  一共num_layer个

            output = tf.concat((final_states[0].h, final_states[1].h), 1)
            # [batch_size ,2* hidden_size]
            print(output.shape)


        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(output, keep_prob=0.5)

        with tf.name_scope('output'):
            self.slot_W = []
            self.slot = tf.Variable(tf.constant(2, shape=[self.input_steps, self.batch_size]))

            w = tf.Variable(tf.truncated_normal([self.hidden_size * 2, self.intent_size], stddev=0.1), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[self.intent_size]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            # batch_size * intent_size
            print(self.logits.shape)


            self.intent = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')
            print(self.intent.shape)
            self.mask = []

        with tf.name_scope('loss'):

            losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.intent_targets, depth=self.intent_size, dtype=tf.float32),
            logits=self.logits)

            self.loss = tf.reduce_mean(losses)
            print("loss: ", self.loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(0.001)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, 5)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.intent, tf.argmax(self.intent_targets, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

            print("loss: ", self.loss)
            print("acc: ", self.accuracy)


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
        return list
