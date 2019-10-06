#!usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import jieba

MODEL = "svc"

# the directory where the data lies
DATA_DIR  = "./data/svc_rf_train/"

# training file name
TRAIN_FNAME = "train.json"

# valid data file name
VALID_FNAME = "valid.json"

# testing file name
TEST_FNAME = "test.json"

# location of `law.txt` file
LAW_FILE_LOC = "./utils/law.txt"

# location of `accu.txt` file
ACCU_FILE_LOC = "./utils/accu.txt"

# the location of stopwords
STOPWORDS_LOC = "./utils/stopwords.txt"

# TF-IDF model dumped file location
TFIDF_LOC = "./predictor/model/"+MODEL+"/tfidf.model"

# accusation model dumped file location
ACCU_LOC = "./predictor/model/"+MODEL+"/accusation.model"

# article model dumped file location
ART_LOC = "./predictor/model/"+MODEL+"/article.model"

# imprisonment model dumped file location
IMPRISON_LOC = "./predictor/model/"+MODEL+"/imprisonment.model"

# mid data location
MID_DATA_PKL_FILE_LOC = "./utils/mid-data.pkl"

# dump the mid-data to local `.pkl` file or not
DUMP = False

# print something log info or not
DEBUG = True

# digitize the death penalty and life imprisonments
DEATH_IMPRISONMENT = -2
LIFE_IMPRISONMENT = -1

def load_stopwords(stopwords_fname):
    """ load stopwords into set from local file """
    stopwords = set()
    with open(stopwords_fname, "r", encoding="utf-8") as f:
        for line in f.readlines():
            stopwords.add(line.strip())

    return stopwords

stopwords = None

jieba.load_userdict("./predictor/userdict.txt")

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
        stopwords = load_stopwords(STOPWORDS_LOC)

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

    return text


def load_law_and_accu_index():
    """ load laws and accusation name and make index """
    law = {}
    lawname = {}
    with open(LAW_FILE_LOC, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            lawname[len(law)] = line.strip()
            law[line.strip()] = len(law)
            line = f.readline()


    accu = {}
    accuname = {}
    with open(ACCU_FILE_LOC, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            accuname[len(accu)] = line.strip()
            accu[line.strip()] = len(accu)
            line = f.readline()


    return law, accu, lawname, accuname

def load_law_index():
    """ only load laws name and make index """
    law = {}
    lawname = {}
    with open(LAW_FILE_LOC, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            lawname[len(law)] = line.strip()
            law[line.strip()] = len(law)
            line = f.readline()

    return law,lawname


law,lawname = load_law_index()
accu={}
accuname={}


def get_class_num(kind):
    global law
    global accu

    if kind == "law":
        return len(law)
    elif kind == "accu":
        return len(accu)
    else:
        raise KeyError


def get_name(index, kind):
    global lawname
    global accuname

    if kind == "law":
        return lawname[index]
    elif kind == "accu":
        return accuname[index]
    else:
        raise KeyError


def get_time(imprison_dict):
    if imprison_dict['death_penalty']:
        return DEATH_IMPRISONMENT

    if imprison_dict['life_imprisonment']:
        return LIFE_IMPRISONMENT

    return int(imprison_dict["imprisonment"])


def get_label(d, kind):
    """ get the index of the law or accusation
    NOTICE: only return the fist label of multi-label data (0-182)
    """
    global law
    global accu

    if kind == "law":
        return law[str(d["meta"]["relevant_articles"][0])]
    elif kind == "accu":
        return accu[d["meta"]["accusation"][0]]
    elif kind == "time":
        return get_time(d["meta"]["term_of_imprisonment"])
    else:
        raise KeyError

def get_all_label(d, kind):
    """ get the index of the law or accusation
    NOTICE: return all labels of multi-label data
    """
    global law
    global accu

    if kind == "law":
        # return law[str(d["meta"]["relevant_articles"][0])]
        labels = []
        for label in d["meta"]["relevant_articles"]:
            labels.append(str(law[str(label)]))
        return "&".join(labels)
    elif kind == "accu":
        # return accu[d["meta"]["accusation"][0]]
        labels = []
        for label in d["meta"]["accusation"]:
            labels.append(str(accu[str(label)]))
        return "&".join(labels)
    elif kind == "time":
        return get_time(d["meta"]["term_of_imprisonment"])
    else:
        raise KeyError


if __name__ == '__main__':
    print(law)
    print(lawname)

