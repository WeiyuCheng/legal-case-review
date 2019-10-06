import torch
import torchtext.data as data
from torchtext.vocab import Vectors
import jieba
import re
import argparse

def load_stopwords(stopwords_fname):
    """ load stopwords into set from local file """
    stopwords = set()
    with open(stopwords_fname, "r", encoding="utf-8") as f:
        for line in f.readlines():
            stopwords.add(line.strip())

    return stopwords

stopwords = None

jieba.load_userdict("/home/chenrunjin/data/CAIL_codes/dictionary/userdict.txt")


def y_tokenize(y):
    return int(y)

def multi_y_tokenize(y):
    y = [int(x) for x in y.split('&')]
    label = [0]*183
    for x in y:
        label[x]=1
    return label

def cut_line(line):
    """ cut the single line using `jieba` """

    global stopwords

    # remove the date and time
    line = re.sub(r"\d*年\d*月\d*日", "", line)
    line = re.sub(r"\d*[时|时许]", "", line)
    line = re.sub(r"\d*分", "", line)

    word_list = jieba.cut(line)

    if stopwords is None:
        print("DEBUG: stopwords loaded.")
        stopwords = load_stopwords("/home/chenrunjin/data/CAIL_codes/utils/stopwords.txt")

    # remove the stopwords
    words = []
    for word in word_list:
        if word not in stopwords:
            words.append(word)

    text = " ".join(words)

    # correct some results
    # merge「王」and「某某」into「王某某」
    text = re.sub(" 某某", "某某", text)

    # merge「2000」and「元」into「2000元」
    text = re.sub(" 元", "元", text)
    text = re.sub(" 余元", "元", text)

    text = re.sub("价 格", "价格", text)

    text = text.split(' ')

    if len(text) < 5:
        for i in range(5-len(text)):
            text.append('<pad>')

    return text




def process_data(text_field, label_field, data_dir, batch_size, **kwargs):
    """
    :param text_field: text_field
    :param label_field: label_field
    :param data_dir: input_dir and it contains train.tsv test.tsv val.tsv
    :param batch_size:
    :param mode: all/mini/sample
    :param kwargs:
    :return: only return the first label if multi labels
    """
    text_field.tokenize = cut_line
    label_field.tokenize = y_tokenize
    train_dataset, dev_dataset, test_dataset = data.TabularDataset.splits(
        path=data_dir, format='tsv', skip_header=True,
        train='train.tsv', validation='val.tsv', test='test.tsv',
        fields=[
            ('index', None),
            ('text', text_field),
            ('label', label_field)
        ]
    )
    text_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_dataset, dev_dataset,test_dataset),
        batch_sizes=(batch_size, batch_size,batch_size),
        sort_key=lambda x: len(x.text), **kwargs)

    return train_iter, dev_iter, test_iter

