# coding = utf-8
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
   # clearBlankLine(inputfile="corpus.train.txt", outputfile="train_without_blankline.txt")
    # 输入原始训练集，输出无空行的训练集
   # clearBlankLine(inputfile="corpus.test.txt", outputfile="test_without_blankline.txt")
    # 输入原始测试集，输出无空行的测试集
    clearBlankLine(inputfile="C:\\Users\\pc\\Desktop\\nlpcc2018-task4-master\\nlpcc2018-task4-master\data\\Golden\\task4-subtask2-result1.txt",
                   outputfile="C:\\Users\\pc\\Desktop\\nlpcc2018-task4-master\\nlpcc2018-task4-master\data\\Golden\\task4-subtask2-result2.txt")
    clearBlankLine(
        inputfile="C:\\Users\\pc\\Desktop\\nlpcc2018-task4-master\\nlpcc2018-task4-master\data\\Golden\\task4-subtask4-result1.txt",
        outputfile="C:\\Users\\pc\\Desktop\\nlpcc2018-task4-master\\nlpcc2018-task4-master\data\\Golden\\task4-subtask4-result2.txt")
