from DataRead import reas ,text ,l
#人名，时间，地名，组织名使用哈工大的pyltp模板工具进行提取然后计算
# timecount = []
# personcount = []
# placecount = []
# orgscount = []
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer

class LTP(object):
    def __init__(self):
        cws_model_path = os.path.join('../data/ltp_data_v3.4.0', 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        pos_model_path = os.path.join('../data/ltp_data_v3.4.0', 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        ner_model_path = os.path.join('../data/ltp_data_v3.4.0', 'ner.model')   # 命名实体识别模型路径，模型名称为`pos.model`        

        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load(cws_model_path) # 加载模型
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(pos_model_path)  # 加载模型
        self.recognizer = NamedEntityRecognizer() # 初始化实例
        self.recognizer.load(ner_model_path)  # 加载模型
    # 分词
    def segment(self, text):
        words = list(self.segmentor.segment(text))
        return words

    # 词性标注
    def postag(self, words):
        postags = list(self.postagger.postag(words))
        return postags

    # 获取文本中的时间
    def get_time(self, text):

        # 开始分词及词性标注
        words = self.segment(text)
        #print(words)
        postags = self.postag(words)
        #print(postags)

        time_lst = []

        i = 0
        for tag, word in zip(postags, words):
            if tag == 'nt':
                j = i
                while postags[j] == 'nt' or words[j] in ['至', '到']:
                    j += 1
                time_lst.append(''.join(words[i:j]))
            i += 1
 
        # 去重子字符串的情形
        remove_lst = []
        for i in time_lst:
            for j in time_lst:
                if i != j and i in j:
                    remove_lst.append(i)

        text_time_lst = []
        for item in time_lst:
            if item not in remove_lst:
                text_time_lst.append(item)

        # print(text_time_lst)
        return text_time_lst
    
    
    #提取人名地名组织名
    def get_name(self,text):
        persons, places, orgs = set(), set(), set()

        words = self.segment(text)
        #print("words333333333333")
        postags = self.postag(words)
        #print(postags)
        netags = list(self.recognizer.recognize(words, postags))  # 命名实体识别
        #print(netags)
        # print(netags)
        i = 0
        for tag, word in zip(netags, words):
            j = i
            # 人名
            if 'Nh' in tag:
                if str(tag).startswith('S'):
                    persons.add(word)
                elif str(tag).startswith('B'):
                    union_person = word
                    while netags[j] != 'E-Nh':
                        j += 1
                        if j < len(words):
                            union_person += words[j]
                    persons.add(union_person)
            # 地名
            if 'Ns' in tag:
                if str(tag).startswith('S'):
                    places.add(word)
                elif str(tag).startswith('B'):
                    union_place = word
                    while netags[j] != 'E-Ns':
                        j += 1
                        if j < len(words):
                            union_place += words[j]
                    places.add(union_place)
            # 机构名
            if 'Ni' in tag:
                if str(tag).startswith('S'):
                    orgs.add(word)
                elif str(tag).startswith('B'):
                    union_org = word
                    while netags[j] != 'E-Ni':
                        j += 1
                        if j < len(words):
                            union_org += words[j]
                    orgs.add(union_org)

            i += 1

        # print('人名：', '，'.join(persons))
        # print('地名：', '，'.join(places))
        # print('组织机构：', '，'.join(orgs))
        return persons,places,orgs


    # 释放模型
    def free_ltp(self):
        self.segmentor.release()
        self.postagger.release()
timecount = []
personcount = []
placecount = []
orgscount = []
if __name__ == '__main__':
    ltp = LTP()
    p_arr = []
    # 输入文本
    for sent in range(len(text)): #range(1): #
        time_list = ltp.get_time(text[sent])
        person, place, orgs = ltp.get_name(text[sent])
        timecount.append(len(time_list))
        personcount.append(len(person))
        placecount.append(len(place))
        orgscount.append(len(orgs))
        if (len(time_list) == 0):
                p_arr = np.array(time_list)
        else:
                p_arr = np.append(p_arr,time_list,axis=0)
          #输出文本中提取的时间
        print('提取时间： %s' % str(time_list))
    # print(len(text))
    # print("timetime::::::::::",timecount)
    # print(len(timecount))
    # print(len(personcount))
    # print(placecount)
    # print(orgscount)
    ltp.free_ltp()

#提取金额数量和最大金额值特征
# moneymax = []
# moneycount = []
import re
Regx = re.compile("(([1-9]\\d*[\\d,，]*\\.?\\d*)|(0\\.[0-9]+))(元|万元|亿元)|[\d一二三四五六七八九十壹贰叁肆伍陆柒捌玖拾]+元")   # ]+元") #
 
#此处txt为上文的文本
txt = text[0]

#对整个文本进行分句，根据个人统计和测试，只需要用逗号对文本分句足矣
def cutFun(initial_txt):
    result = re.split('。',initial_txt)
    return result
 
#符合要求的文本加入集合中
#如果一个句子中有“元”那么将该句子存放在一个临时变量中以供试用

def YUAN_contain(content):
    temp_result_set = set()
    str = '元'
    for term in cutFun(content):
        if str in term:
            #print(term)
            temp_result_set.add(term)
    return temp_result_set

def money_contain(txt):
    temp_result_set = YUAN_contain(txt)
 
    #for term in temp_result_set:
       # print(term)
    print('___________________________')
 
    
    result_set = set()
    for term in temp_result_set:
        for ch in term:
            if ch.isdigit():
                result_set.add(term)
 

    #for term in result_set:
       # print(term)
    print('___________________________')

    #用正则式提取金额
    mon = []
    for term in result_set:
        i = Regx.search(term)
        if i != None:
            mon.append(i.group())
            print(i.group())
    print(mon)
    #quchong
    remove_lst = []
    for i in mon:
        for j in mon:
                if i != j and i in j:
                    remove_lst.append(i)

    mon_lst = []
    for item in mon:
        if item not in remove_lst:
                mon_lst.append(item)


    return mon_lst

def max_mon(mon):
    money = []
    for i in range(len(mon)):
        mon[i] = mon[i].replace(',','')
        mon[i] = mon[i].replace('，','')
        b = mon[i].find('百')
        q = mon[i].find('千')
        w = mon[i].find('万')
        sw = mon[i].find('十万')
        bw = mon[i].find('百万')
        qw = mon[i].find('千万')
        y = mon[i].find('亿')
        e = mon[i].find("元")
        if b != -1:
            money.append(float(str(mon[i])[:b])*100)
        elif q!= -1:
            money.append(float(str(mon[i])[:q])*1000)
        elif w!= -1:
            money.append(float(str(mon[i])[:w])*10000) 
        elif sw!= -1:
            money.append(float(str(mon[i])[:sw])*100000)
        elif bw!= -1:
            money.append(float(str(mon[i])[:bw])*1000000)
        elif qw!= -1:
            money.append(float(str(mon[i])[:qw])*10000000)
        elif y!= -1:
            money.append(float(str(mon[i])[:y])*100000000)
        elif e != -1:
            money.append(float(str(mon[i])[:e]))
            
    return money


 
def nantonum (mon):
    if(mon!= None):
        for i in range(len(mon)):
            if mon[i] == None:
                mon[i] = 0
        #return mon

common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2,'壹': 1, '贰': 2,'叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9,
                            '拾': 10,  '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}


common_used_numerals = {}
for key in common_used_numerals_tmp:
    common_used_numerals[key] = common_used_numerals_tmp[key]


print("common_used_numerals",common_used_numerals)
def chinese2digits(uchars_chinese):
    total = 0
    r = 1  # 表示单位：个十百千...
    for i in range(len(uchars_chinese) - 1, -1, -1):
        val = common_used_numerals.get(uchars_chinese[i])
        if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
            if val > r:
                r = val
                total = total + val
            else:
                r = r * val
                # total =total + r * x
        elif val >= 10:
            if val > r:
                r = val
            else:
                r = r * val
        else:
            total = total + r * val
    return total


num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九','十','壹', '贰','叁', '肆', '伍', '陆', '柒', '捌', '玖',
                            '拾']

more_num_str_symbol = ['〇','零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']

def changeChineseNumToArab(oriStr):
    lenStr = len(oriStr);
    aProStr = ''
    if lenStr == 0:
        return aProStr;

    hasNumStart = False;
    numberStr = ''
    for idx in range(lenStr):
        if oriStr[idx] in num_str_start_symbol:
            if not hasNumStart:
                hasNumStart = True;

            numberStr += oriStr[idx] 
        else:
            if hasNumStart:
                if oriStr[idx] in more_num_str_symbol:
                    numberStr += oriStr[idx]
                    continue
                else:
                    numResult = str(chinese2digits(numberStr))
                    numberStr = ''
                    hasNumStart = False;
                    aProStr += numResult

            aProStr += oriStr[idx]
            pass

    if len(numberStr) > 0:
        resultNum = chinese2digits(numberStr)
        aProStr += str(resultNum)

    return aProStr

moneymax = []
moneycount = []
    

for sent in range(len(text)): #range(1): #
    money_list = money_contain(text[sent])
    moneylist = []
    for s in range(len(money_list)):
        moneylist.append(changeChineseNumToArab(money_list[s]))
    print("hanyou zhongwen ", moneylist)
    moneylist = max_mon(moneylist)
    print(moneylist)
    #nantonum(money_list) 
    if money_list != []:
          monmax = max(moneylist)
    else:
          monmax = 0
    
    moneymax.append(monmax)  
    moneycount.append(len(moneylist))
    print(sent)
    print('最大金额： %s' % monmax)
    

# print(moneymax)
# print(moneycount)


#统计案情描述文本长度和句子个数
# textlen = []
# sentencecount = []

textlen = []
sentencecount = []
for sent in range(len(text)): #range(1): #
    textlen.append(len(text[sent]))  
# #print(textlen)
# print(len(textlen))

for sent in range(len(text)): #range(1): #
    sentencecount.append(len(text[sent].split('。')))  
# print(sentencecount)
# print(len(sentencecount))

#统计文本中是否出现'当事人无异议/无异议" "事实清楚/清楚" "理由充分/充分"
#yi = []
#qingchu = []
#chongfen = []
import string 
def yiyi(x):

        if (str.find(x,"当事人无异议") != -1):
            return 1
        elif (str.find(x,"无异议") != -1):
            return 1
        else:
            return 0
    
yi = []
for i in range(len(text)):
       yi.append(yiyi(text[i])) 
# print(len(yi))
# print(yi)

def shishiqingchu(x):

        if (str.find(x,"事情清楚") != -1):
            return 1
        elif (str.find(x,"清楚") != -1):
            return 1
        else:
            return 0
    
qingchu = []
for i in range(len(text)):
       qingchu.append(shishiqingchu(text[i])) 
# print(len(qingchu))
# print(qingchu)

def liyouchongfen(x):

        if (str.find(x,"理由充分") != -1):
            return 1
        elif (str.find(x,"充分") != -1):
            return 1
        else:
            return 0
    
chongfen = []
for i in range(len(text)):
       chongfen.append(liyouchongfen(text[i])) 
# print(len(chongfen))
# print(chongfen)

#案由特征
#案由选取样本数量大于50的案由进行 one_hot编码
tmp = list(set(reas))
tmp.sort(key = reas.index)
print(tmp)
print(len(tmp))
import pandas as pd
rea = []
reacount = np.zeros(163)

for i in range(len(reas)):
    rea.append(tmp.index(reas[i]))
    reacount[tmp.index(reas[i])] += 1 
j =0 
left = []
for i in range(163):
    if(reacount[i] > 10):
        left.append(i)
        j += 1
#         print(reacount[i])
#         print("zhegebiaoqianshi000", i)
# print("j",j)
# print(left)
 

reasons = []
for i in range(len(reas)):
        if rea[i] in left:
            reasons.append(rea[i])
        else:
            reasons.append(-1)
            
# print("reasons",reasons)

xreasons = pd.get_dummies(reasons) 
# print ("xre",xreasons)
     
#提取的所有特征作为样本
x_data_train = np.vstack([timecount,
                          moneycount,
                          moneymax,
                          personcount,
                          placecount,
                          orgscount,
                          textlen,
                          sentencecount,
                          yi,
                          qingchu,
                          chongfen])
x = np.transpose(x_data_train)

x = np.hstack((x,xreasons))

# print(x)
# print(x.shape)