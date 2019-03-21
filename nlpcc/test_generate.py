import random
data = open("train2.txt", "r" ,encoding='UTF-8').readlines()
fp = open("test.txt", 'w', encoding='UTF-8')


if __name__ == '__main__':
    for t in data:
        if random.uniform(0, 100) <= 10:
            fp.write(t)


