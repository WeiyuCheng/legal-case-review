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

re,text,y = read_trainData("../data/normalcriminal2016.json")
rea,te,yy = read_trainData("../data/simplecriminal2016.json")
ro,tt,yty = read_trainData("../data/simplecriminal2017.json")
ron,tte,yt = read_trainData("../data/normalcriminal2017.json")
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