# coding=utf-8
# @author: cer
import numpy as np
import numpy.ma as ma
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def accuracy_score(true_data, pred_data, true_length=None):
    true_data = np.array(true_data)
    pred_data = np.array(pred_data)
    assert true_data.shape == pred_data.shape
    if true_length is not None:             # 如果输入了true_length，实际上指计算slot_acc时走if分支
        val_num = np.sum(true_length)
        assert val_num != 0
        res = 0
        for i in range(true_data.shape[0]):
            res += np.sum(true_data[i, :true_length[i]] == pred_data[i, :true_length[i]])
    else:                                   # 如果没输入true_length，实际上指计算intent_acc时走else分支
        val_num = np.prod(true_data.shape)
        assert val_num != 0
        res = np.sum(true_data == pred_data)
    res /= float(val_num)
    return res


def get_data_from_sequence_batch(true_batch, pred_batch, padding_token):
    """从序列的batch中提取数据：
    [[3,1,2,0,0,0],[5,2,1,4,0,0]] -> [3,1,2,5,2,1,4]"""
    true_ma = ma.masked_equal(true_batch, padding_token)
    # 将true_batch中每个向量的等于0的分量全部标记

    pred_ma = ma.masked_array(pred_batch, true_ma.mask)
    # 使用true_ma的标记，将pred_ma相同位置标记
    true_ma = true_ma.flatten()
    pred_ma = pred_ma.flatten()
    true_ma = true_ma[~true_ma.mask]
    pred_ma = pred_ma[~pred_ma.mask]

    return true_ma, pred_ma


def f1_for_sequence_batch(true_batch, pred_batch, average="macro", padding_token=0):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, padding_token)
    # true和pred均为一维数组
    labels = list(set(true))
    #print("true: ", true)
    #print("pred: ", pred)
    return f1_score(true, pred, labels=labels, average=average)


# 下面不包括OTHERS类、<UNK>类
def f1_for_sequence_batch_new(true_batch, pred_batch, option):
    import warnings
    warnings.filterwarnings("ignore")       #忽略警告
    ALL_INTENT = {'<UNK>': 0, "music.play": 1, "music.pause": 2, "music.prev": 3, "music.next": 4,
                  "navigation.navigation": 5, "navigation.open": 6, "navigation.start_navigation": 7,
                  "navigation.cancel_navigation": 8, "phone_call.make_a_phone_call": 9, "phone_call.cancel": 10,
                  "OTHERS": 11}
    new_dict = {v: k for k, v in ALL_INTENT.items()}

    for i in range(1,len(new_dict)):  #打印所有意图的F1
        label = []
        label.append(i)
        print(new_dict[i],":",f1_score(true_batch, pred_batch, labels =label,average="macro" ))

    labels = list(set(true_batch))
    # print("len_intent_all: ",len(true_batch))
    if option == "no_others":       # 测算没有others的F1
        labels.remove(11)

    return f1_score(true_batch, pred_batch, labels=labels, average="macro")


def accuracy_for_sequence_batch(true_batch, pred_batch, padding_token=0):
    true, pred = get_data_from_sequence_batch(true_batch, pred_batch, padding_token)
    return accuracy_score(true, pred)


#函数供测试使用
if __name__ == '__main__':
    y_true = [1, 2, 3]
    y_pred = [1, 1, 3]
    print(f1_score(y_true,y_pred,labels=[1],average='macro'))
    print(f1_for_sequence_batch_new(y_true,y_pred,"n"))