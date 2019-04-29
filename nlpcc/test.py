# _*_coding:utf-8_*_
import jieba
import re
import os
import jieba.posseg as psg



def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars




if __name__ == '__main__':
    seg_list = jieba.cut("去青岛香港东路青岛大学", cut_all=True)  # 全模式
    print("Full Mode:" + "/".join(seg_list))

    seg_list = jieba.cut("去青岛香港东路青岛大学", cut_all=False)  # 精确模式
    print("Default Mode:" + "/".join(seg_list))

    seg_list = jieba.cut("去青岛香港东路青岛大学", HMM=False)  # 不使用HMM模型
    t=" ".join(seg_list)
    print(t)

    seg_list = jieba.cut("播放风情万的歌曲", HMM=True)  # 使用HMM模型
    print("~".join(seg_list).split('~'))
    print("请帮我放一首the well in the best girls run and run".split(" "))
    print(list(jieba.cut("请帮我放一首the well in the best girls run and run", HMM=True)))

    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM=False)  # 搜索引擎模式
    print("/".join(seg_list))

    seg_list = jieba.lcut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM=True)
    print(os.path.abspath(os.path.dirname(__file__)))
    print(os.path.dirname(os.path.abspath(__file__)))

    print(seg_char("*请帮我100放,t一首1*2the well, in the best girls run and run"))
    print(seg_char("*11 text"))

    '''
    fw_cell = tf.nn.rnn_cell.BasicRNNCell(128)
    bw_cell = tf.nn.rnn_cell.BasicRNNCell(256)
    inputs = tf.Variable(tf.random_normal([100, 40, 300]))  # [batch_size,timestep,embedding_dim]
    inputs = tf.unstack(inputs, 40, axis=1)
    print(inputs[0])
    outputs, fw, bw = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32)
    print(len(outputs))     # 40，40个时间步
    print(outputs[0].shape)     # (100, 384),每个时间步的输出神经元为384=128+256
    print(outputs[1].shape)     # (100, 384)
    print(fw.shape)     # (100, 128),前向RNN隐藏层
    print(bw.shape)  # (100, 128)，后向RNN传播隐藏层

    a = [
        [1,2,3],
        [4,5,6]
    ]
    b = tf.reshape(a,[-1])
    c = tf.reshape(b,[3,2])
    sess = tf.Session()
    print(sess.run(b))
    print(sess.run(c))
    '''
    slot_category = ["age", "custom_destination", "emotion", "instrument", "language",
                     "scene", "singer", "song", "style", "theme", "toplist"]
    str = "放一首<singer>罗大佑</singer>的<song>皇后</song><singer>大</singer><song>道东</song>"
    slot_list = re.findall("[<]" + slot_category[6] + "[>].*?[<]/" + slot_category[6] + "[>]", str)  # 提取尖括号中内容
    print(slot_list)
    i = str.index(">")
    j = str.index("</")
    print(str.index(">"))
    print(str.index("</"))
    print(str[i+1:j])

    temp_str = "播放一首我们不一样asdf"
    value = "我"
    index_start = temp_str.index(value)
    index_end = index_start + len(value)
    print(index_start)
    print(index_end)
    #print(temp_str[:index_start])
    print(temp_str[index_start+1:])
    #print(temp_str[index_end:])

    for x in psg.cut("堤口路水果市场"):
        print(x.word, x.flag)
    print("ArgMax:0".split(","))









