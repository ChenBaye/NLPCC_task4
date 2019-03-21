# coding = utf-8
def clearBlankLine():#清除数据中的空行
    file1 = open('corpus.train.txt', 'r', encoding='utf-8') # 要去掉空行的文件
    file2 = open('train1.txt', 'w', encoding='utf-8') # 生成没有空行的文件
    try:
        for line in file1.readlines():
            if line == '\n':
                line = line.strip("\n")
            file2.write(line)
    finally:
        file1.close()
        file2.close()


if __name__ == '__main__':
    clearBlankLine()
