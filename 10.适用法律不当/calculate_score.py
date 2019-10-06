law_rate_threshold =[0.0001,0.001,1.0]
acc_len = len(law_rate_threshold)


law_rate_path = "./utils/law_rate.txt"
law_rate=[]
with open(law_rate_path,'r') as f:
    lines = f.readlines()
    for line in lines:
        rate = float(line.strip("\n").split(' ')[1])
        law_rate.append(rate)




def calculate_score(path):
    print(path)
    inf = open(path, "r")
    size = [0] * acc_len
    corrects = [0] * acc_len
    acc = [0.0] * acc_len
    line = inf.readline()
    while line:
        l = line.strip('\n').split(',')
        res = int(l[0])
        truth = int(l[1])
        rate = law_rate[truth]
        for i in range(acc_len):
            if rate < law_rate_threshold[i]:
                size[i] += 1
                if res == truth:
                    corrects[i] += 1
        line = inf.readline()

    inf.close()
    for i in range(acc_len):
        acc[i] = (corrects[i] * 1.0) / size[i]
        print("Rate <= {:f}:  Accuracy--{:.8f}:{}/{}".format(law_rate_threshold[i], acc[i], corrects[i], size[i]))


if __name__ == '__main__':
    paths = ["/home/chenrunjin/data/github_code/output/rf/test.txt",
             "/home/chenrunjin/data/github_code/output/svc/test.txt",
             "/home/chenrunjin/data/github_code/output/OLTR/pretrained_OLTR/test.txt",
             "/home/chenrunjin/data/github_code/output/textcnn/pretrained_textcnn_dim128_filter_2,3,4_num100/test.txt",
             "/home/chenrunjin/data/github_code/output/BERT/pretrained_bert/test.txt",
             "/home/chenrunjin/data/github_code/output/knn/pretrained_textcnn_dim128_filter_2,3,4_num100/test.txt",
             ]

    for path in paths:
        calculate_score(path)