import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = os.path.abspath(os.path.dirname(__file__))   #path = ...\nlpcc

# 将输入list扩展为length长
def append_list(list, batch_size, input_step):
    for i in range(len(list)):
        while len(list[i])<input_step:
            list[i].append(0)

    while len(list)<batch_size:
        list.append([0]*input_step)

    return list

def append_len(length_list, batch_size):
    while len(length_list) < batch_size:
        length_list.append(0)

    return length_list


if __name__ == '__main__':
    # 一次只能测试一个模型

    '''
    with tf.Session() as sess:
        meta_file = path + "\\model_jointmodel\\jointmodel29.meta"
        model_path = path + "\\model_jointmodel\\jointmodel29"
        # model_path并不是真正的path，实际上为各个文件的前缀

        saver = tf.train.import_meta_graph(meta_file)       # 读取网络结构
        print("get meta...")
        saver.restore(sess, model_path)  # 读取网络中的参数
        print("get variable...")
        # 测试数据如下
        # 112197732	到/石/仁	O B-destination I-destination	navigation.navigation
        # 112196269	我/想/听/陈/奕/迅/的/不/如/不/见	O O O B-singer I-singer I-singer O B-song I-song I-song I-song	music.play
        graph = tf.get_default_graph()
        encoder_inputs = graph.get_tensor_by_name("encoder_inputs:0")
        encoder_inputs_actual_length = graph.get_tensor_by_name("encoder_inputs_actual_length:0")
        #intent = graph.get_tensor_by_name("self.intent:0")

        input_list = [
                            [4295, 3349, 3679],
                            [1789, 501, 1246, 3644, 292, 1589, 3719, 1074, 3703, 1074, 3704]
                          ]
        length_list = [3, 11]

        input_list = append_list(input_list, 25, 45)
        length_list = append_len(length_list, 25)

        feed_dict = {encoder_inputs: np.transpose(input_list, [1,0]),
                     encoder_inputs_actual_length: length_list}

        intent = graph.get_tensor_by_name("ArgMax:0")
        result = sess.run(intent,feed_dict= feed_dict)
        print(result)
        # 得出意图[ 5  1 11 11 .....11]
        #   5 - navigation.navigation
        #   1 - music.play
        # 均正确


        # 训练完才发现忘记给intent和slot张量命名.........
        # 所以两个张量为自动生成的名字
        # jointmodel中intent张量为"ArgMax:0"
        # blstmcrf模型中slot张量为“transpose_1:0”
    '''

    '''
    with tf.Session() as sess:
        meta_file = path + "\\model_blstmcrf\\blstmcrf_model2.meta"
        model_path = path + "\\model_blstmcrf\\blstmcrf_model2"
        # model_path并不是真正的path，实际上为各个文件的前缀

        saver = tf.train.import_meta_graph(meta_file)       # 读取网络结构
        print("get meta...")
        saver.restore(sess, model_path)  # 读取网络中的参数
        print("get variable...")
        # 测试数据如下
        # 112197732	到/石/仁	O B-destination I-destination	navigation.navigation
        # 112196269	我/想/听/陈/奕/迅/的/不/如/不/见	O O O B-singer I-singer I-singer O B-song I-song I-song I-song	music.play
        graph = tf.get_default_graph()
        encoder_inputs = graph.get_tensor_by_name("encoder_inputs:0")
        encoder_inputs_actual_length = graph.get_tensor_by_name("inputs_actual_length:0")
        #intent = graph.get_tensor_by_name("self.intent:0")

        input_list = [
                            [4295, 3349, 3679],
                            [1789, 501, 1246, 3644, 292, 1589, 3719, 1074, 3703, 1074, 3704]
                          ]
        length_list = [3, 11]

        input_list = append_list(input_list, 25, 45)
        length_list = append_len(length_list, 25)

        feed_dict = {encoder_inputs: np.transpose(input_list, [1,0]),
                     encoder_inputs_actual_length: length_list}

        slot = graph.get_tensor_by_name("transpose_1:0")
        result = sess.run(slot,feed_dict= feed_dict)
        result = np.transpose(result, [1, 0])
        print("slot:")
        print(result[0])
        print(result[1])
        # 得出槽
        # [2 2 2 0 0 0.......0]
        # O O O.....
        # [ 2  2  2  4 19 19  2  3 18 18 18 0 0 ....0]
        # O O O B-singer I-singer I-singer O B-song I-song I-song I-song ....
        # 均正确


        # 训练完才发现忘记给intent和slot张量命名.........
        # 所以两个张量为自动生成的名字
        # jointmodel中intent张量为"ArgMax:0"
        # blstmcrf模型中slot张量为“transpose_1:0”
    '''