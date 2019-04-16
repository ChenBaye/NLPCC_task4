import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys
import os
from dataset_process import word_embeding
from dataset_process import read_fasttext

class Model(object):
    def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size=16, n_layers=1):
        self.input_steps = input_steps
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_hidden = hidden_size
        self.num_layers = n_layers
        self.intent_size = intent_size
        self.vocab_size = vocab_size
        self.encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size])
        self.intent_targets = tf.placeholder(tf.int32, [batch_size],
                                             name='intent_targets')
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)

        self.inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='inputs_actual_length')
        self.slot_targets = tf.placeholder(tf.int32, [batch_size, input_steps],
                                           name='slot_targets')

    def build(self):
        with tf.name_scope("embedding"):
            self.embeddings = tf.Variable(self.load_word_embeding())
            self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        with tf.name_scope("birnn"):
            fw_cells = [tf.contrib.rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [tf.contrib.rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            fw_cells = [DropoutWrapper(cell, output_keep_prob=0.5) for cell in fw_cells]
            bw_cells = [DropoutWrapper(cell, output_keep_prob=0.5) for cell in bw_cells]
            self.rnn_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells,
                bw_cells,
                self.x_emb,
                sequence_length=self.inputs_actual_length,
                dtype=tf.float32,
                time_major = True)

        print(self.rnn_outputs)
        # self.rnn_outputs:[seq_length, batch_size, 2 * hidden_dim]
        self.rnn_outputs = tf.transpose(self.rnn_outputs, [1, 0, 2])
        print(self.rnn_outputs)
        # self.rnn_outputs:[batch_size, seq_length, 2 * hidden_dim]


        with tf.name_scope("attention"):
            # tf.layers.dense 经过一个全连接层，激活函数为tanh ，改变最后一个维度的大小为1
            self.attention_score = tf.nn.softmax(tf.layers.dense(self.rnn_outputs, 1, activation=tf.nn.tanh), axis=1)
            print("self.attention_score shape: ", self.attention_score.shape)
            self.attention_out = tf.squeeze(
                tf.matmul(tf.transpose(self.rnn_outputs, perm=[0, 2, 1]), self.attention_score),
                axis=-1)
            # self.attention_out shape = [batch_size ,2 * hidden_size]
            print("self.attention_out shape: ", self.attention_out.shape)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(self.attention_out, self.intent_size, activation=None)
            # self.logits = [batch_size, intent_size]
            print("self.logits shape: ", self.logits.shape)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)
            # self.predictions shape = [batch_size]
            print("self.predictions shape: ",self.predictions.shape)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.intent_targets
                ))
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.intent_targets)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.intent = self.predictions
            self.mask = []
            self.slot_W = []
            self.slot = tf.Variable(tf.constant(2, shape=[self.input_steps, self.batch_size]))

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
        else:
            print("get fasttext word_vector...")
            list = read_fasttext.get_vector()       #读取fastext的词向量
        return list