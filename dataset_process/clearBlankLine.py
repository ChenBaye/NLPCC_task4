# coding = utf-8
import os

def clearBlankLine(inputfile,outputfile):#清除数据中的空行
    file1 = open(inputfile, 'r', encoding='utf-8') # 要去掉空行的文件
    file2 = open(outputfile, 'w', encoding='utf-8') # 生成没有空行的文件
    try:
        for line in file1.readlines():
            if line == '\n':
                line = line.strip("\n")
            file2.write(line)
    finally:
        file1.close()
        file2.close()


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))  #上个目录...\dataset_process
    print(path)
    clearBlankLine(inputfile=path+"\\raw_dataset\\corpus.train.txt", outputfile=path+"\\without_blankline\\train_without_blankline.txt")
    # 输入原始训练集，输出无空行的训练集
    clearBlankLine(inputfile=path+"\\raw_dataset\\corpus.test.txt", outputfile=path+"\\without_blankline\\test_without_blankline.txt")


