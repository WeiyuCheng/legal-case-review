from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import json
from sklearn.metrics import accuracy_score
import jieba
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
from keras.utils import plot_model

FactTag = ['当事人','本院认为','本院查明','上诉人诉称','原告诉称', '申诉人诉称','被上诉人辩称',
           '被申诉人辩称','被告辩称','证据','一审原告诉称', '一审被告辩称']
def read_trainData(path):
    fin = open(path, 'r', encoding='utf8')
    allen = []
    alltext = []
    alllabel = []
    reason = []
    label = -1
    
    print("path.find('simple')",path.find('simple'))
    if path.find('simple') == -1:
        label = 0
    else:
        label = 1

    lines = fin.readlines()
    for l in lines:
        d = json.loads(l)
        if d.get('procedureId') == '一审':
            fact = ''
            for tag in FactTag:
                if d.get(tag)!=None:
                    fact += d.get(tag)
            if fact !=  '':
                #allen.append(len(fact))
                alltext.append(fact)
                reason.append(d.get('reason'))
                alllabel.append(label) 
        
        
    fin.close()

    return reason,alltext, alllabel

re,text,y = read_trainData("./normalcriminal2016.json")
rea,te,yy = read_trainData("./simplecriminal2016.json")
ro,tt,yty = read_trainData("./simplecriminal2017.json")
ron,tte,yt = read_trainData("./normalcriminal2017.json")
reas = np.concatenate((re,rea,ro,ron))
text = np.concatenate((text,te,tt,tte))
y = np.concatenate((y,yy,yty,yt)) 

data = list(zip(reas,text,y) )
np.random.shuffle(data)
reas = [i[0] for i in data]
text = [i[1] for i in data]
l = [i[2] for i in data]

l = np.array(l)
print(text[0])
print(reas[0])
def cut_texts(texts=None, need_cut=True, word_len=1, texts_cut_savepath=None):
        '''
        文本分词剔除停用词
        :param texts:文本列表
        :param need_cut:是否需要分词
        :param word_len:保留词语长度
        :param texts_cut_savepath:保存路径
        :return:
        '''
        if need_cut:
            if word_len > 1:
                texts_cut = [[word for word in jieba.lcut(one_text) if len(word) >= word_len] for one_text in texts]
            else:
                texts_cut = [jieba.lcut(one_text) for one_text in texts]
        else:
            if word_len > 1:
                texts_cut = [[word for word in one_text if len(word) >= word_len] for one_text in texts]
            else:
                texts_cut = texts

        if texts_cut_savepath is not None:
            with open(texts_cut_savepath, 'w') as f:
                json.dump(texts_cut, f)
        return texts_cut

def text2seq(texts_cut=None, tokenizer_fact=None, num_words=2000, maxlen=30):
        '''
        文本转序列，训练集过大全部转换会内存溢出，每次放5000个样本
        :param texts_cut: 分词后的文本列表
        :param tokenizer:转换字典
        :param num_words:字典词数量
        :param maxlen:保留长度
        :return:向量列表
        '''
        texts_cut_len = len(texts_cut)

        if tokenizer_fact is None:
            tokenizer_fact = Tokenizer(num_words=num_words)
            if texts_cut_len > 10000:
                print('文本过多，分批转换')
            n = 0
            # 分批训练
            while n < texts_cut_len:
                tokenizer_fact.fit_on_texts(texts=texts_cut[n:n + 10000])
                n += 10000
                if n < texts_cut_len:
                    print('tokenizer finish fit %d samples' % n)
                else:
                    print('tokenizer finish fit %d samples' % texts_cut_len)
            self.tokenizer_fact = tokenizer_fact

        # 全部转为数字序列
        fact_seq = tokenizer_fact.texts_to_sequences(texts=texts_cut)
        print('finish texts to sequences')

        # 内存不够，删除
        del texts_cut

        n = 0
        fact_pad_seq = []
        # 分批执行pad_sequences
        while n < texts_cut_len:
            fact_pad_seq += list(pad_sequences(fact_seq[n:n + 10000], maxlen=maxlen,
                                               padding='post', value=0, dtype='int'))
            n += 10000
            if n < texts_cut_len:
                print('finish pad_sequences %d samples' % n)
            else:
                print('finish pad_sequences %d samples' % texts_cut_len)
        self.fact_pad_seq = fact_pad_seq
def cut_text(alltext):
    count = 0
    cut = jieba
    train_text = []
    for text in alltext: 
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append(' '.join(cut.cut(text)))
    return train_text

fact = cut_text(text)
import pickle
import jieba
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

jieba.setLogLevel('WARN')

def text2seq(texts_cut=None, toke_fact=None, num_words=2000, maxlen=400):
        '''
        文本转序列，训练集过大全部转换会内存溢出，每次放5000个样本
        :param texts_cut: 分词后的文本列表
        :param tokenizer:转换字典
        :param num_words:字典词数量
        :param maxlen:保留长度
        :return:向量列表
        '''
        texts_cut_len = len(texts_cut)
        tokenizer_fact = Tokenizer(num_words=num_words)
        if toke_fact is None:
            
            if texts_cut_len > 10000:
                print('文本过多，分批转换')
            n = 0
            # 分批训练
            while n < texts_cut_len:
                tokenizer_fact.fit_on_texts(texts=texts_cut[n:n + 10000])
                n += 10000
                if n < texts_cut_len:
                    print('tokenizer finish fit %d samples' % n)
                else:
                    print('tokenizer finish fit %d samples' % texts_cut_len)
            

        # 全部转为数字序列
        fact_seq = tokenizer_fact.texts_to_sequences(texts=texts_cut)
        print('finish texts to sequences')

        # 内存不够，删除
        del texts_cut

        n = 0
        fact_pad_seq = []
        # 分批执行pad_sequences
        while n < texts_cut_len:
            fact_pad_seq += list(pad_sequences(fact_seq[n:n + 10000], maxlen=maxlen,
                                               padding='post', value=0, dtype='int'))
            n += 10000
            if n < texts_cut_len:
                print('finish pad_sequences %d samples' % n)
            else:
                print('finish pad_sequences %d samples' % texts_cut_len)
        return  fact_pad_seq
    
fact = text2seq(fact)    
fact = np.array(fact)


