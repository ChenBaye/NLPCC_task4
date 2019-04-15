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
        self.encoder_inputs = tf.placeholder(tf.int32, [batch_size, input_steps],
                                             name='encoder_inputs')
        self.filter_sizes = [3, 4, 5]       # 过滤器的大小、
        self.num_filters =  128                # 每层网络上卷积核的个数
        self.l2_loss = tf.constant(0.0)

        # 每句输入的实际长度，除了padding
        self.inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='inputs_actual_length')
        self.slot_targets = tf.placeholder(tf.int32, [batch_size, input_steps],
                                              name='slot_targets')
        self.intent_targets = tf.placeholder(tf.int32, [batch_size],
                                             name='intent_targets')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')



    def build(self):

        #self.embeddings = tf.Variable(self.load_word_embeding())

        #self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)


        self.W = tf.Variable(
            # 它将词汇表的词索引映射到低维向量表示。它基本上是我们从数据中学习到的lookup table
            tf.Variable(self.load_word_embeding()),
            name="W")
        # 创建实际的embedding操作。embedding操作的结果是形状为 [None, sequence_length, embedding_size] 的3维张量积
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.encoder_inputs)
        # print "^^^^^^^embedded_chars^^^^^^",self.embedded_chars.get_shape()
        # (?, 56, 128)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        # print "^^^^^^^embedded_chars_expanded^^^^^^",self.embedded_chars_expanded.get_shape()
        # (?, 56, 128, 1)




        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                # 定义参数，也就是模型的参数变量
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                """
                TensorFlow卷积conv2d操作接收的参数说明：
                self.embedded_chars_expanded：一个4维的张量，维度分别代表batch, width, height 和 channel
                W:权重
                strides：步长，是一个四维的张量[1, 1, 1, 1]，第一位与第四位固定为1，第二第三为长宽上的步长
                这里都是设为1
                padding：选择是否填充0，TensorFlow提供俩个选项，SAME、VAILD
                """
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # 非线性激活
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 最大值池化，h是卷积的结果，是一个四维矩阵，ksize是过滤器的尺寸，是一个四维数组，第一位第四位必须是1，第二位是长度，这里为卷积后的长度，第三位是宽度，这里为1
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.input_steps - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # 在 tf.reshape中使用-1是告诉TensorFlow把维度展平，作为全连接层的输入
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout，以概率1-dropout_keep_prob，随机丢弃一些节点
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, 0.5)
        '''
            使用经过max-pooling (with dropout applied)的特征向量，我们可以通过做矩阵乘积，然后选择得分最高的类别进行预测。
            我们也可以应用softmax函数把原始的分数转化为规范化的概率，但是这不会改变我们最终的预测结果。
        '''
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.intent_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.intent_size]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            # tf.nn.xw_plus_b 是进行 Wx+b 矩阵乘积的方便形式
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            print("self.scores shape: ", self.scores.shape)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.intent = self.predictions
            print("shape: ", self.intent.shape)

        # 定义损失函数
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=tf.one_hot(self.intent_targets,
                depth=self.intent_size, dtype=tf.float32))
            self.loss = tf.reduce_mean(losses) + 0 * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.intent_targets, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        self.slot = tf.Variable(tf.constant(2, shape=[self.input_steps, self.batch_size]))
        self.mask = []
        self.slot_W = []

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
            feed_dict = {self.encoder_inputs: unziped[0],
                         self.inputs_actual_length: unziped[1],
                         self.slot_targets: unziped[2],
                         self.intent_targets: unziped[3]}
        if mode in ['test']:
            output_feeds = [self.slot, self.intent]
            feed_dict = {self.encoder_inputs: unziped[0],
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
