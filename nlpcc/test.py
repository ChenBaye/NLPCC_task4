# _*_coding:utf-8_*_
import jieba
import re
import os



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

    seg_list = jieba.cut("请帮我放一首the well in the best girls run and run", HMM=True)  # 使用HMM模型
    print("~".join(seg_list).split('~'))
    print("请帮我放一首the well in the best girls run and run".split(" "))
    print(list(jieba.cut("请帮我放一首the well in the best girls run and run", HMM=True)))

    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM=False)  # 搜索引擎模式
    print("/".join(seg_list))

    seg_list = jieba.lcut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造", HMM=True)
    print(os.path.abspath(os.path.dirname(__file__)))
    print(os.path.dirname(os.path.abspath(__file__)))

    print(seg_char("小明硕士毕业于中国科学院计算所，后在 haha123 日本456京都 78 9 大学深"))





