# _*_coding:utf-8_*_
import jieba

if __name__ == '__main__':
    seg_list = jieba.cut("去青岛香港东路青岛大学", cut_all=True)  # 全模式
    print("Full Mode:" + "/".join(seg_list))

    seg_list = jieba.cut("去青岛香港东路青岛大学", cut_all=False)  # 精确模式
    print("Default Mode:" + "/".join(seg_list))

    seg_list = jieba.cut("去青岛香港东路青岛大学", HMM=False)  # 不使用HMM模型
    t=" ".join(seg_list)
    print(t)

    seg_list = jieba.cut("singer阿杜singer的歌", HMM=True)  # 使用HMM模型
    print("/".join(seg_list))

    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM=False)  # 搜索引擎模式
    print("/".join(seg_list))

    seg_list = jieba.lcut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM=True)
    print(seg_list)



