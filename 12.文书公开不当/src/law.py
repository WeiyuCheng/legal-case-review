import json
import jieba
import jieba.posseg as pseg


'''
word_file = open("LegalwordDict.txt", 'r', encoding="utf-8")
words = word_file.read()
word_list = words.split('\n')
print(word_list)
'''
# 添加词典
jieba.load_userdict("LegalwordDict.txt")
jieba.add_word("审理未成年人刑事案件")
jieba.add_word("作案时未满")
jieba.add_word("作案时未成年")
jieba.add_word("犯罪时未满")
jieba.add_word("犯罪时未成年")
jieba.add_word("未满十八岁")
jieba.add_word("未满十八周岁")


# 判断被告人年龄
def age_judge(people):
    # 确定审判时间，由于数据的审判年份一致，所以这里简化，直接输入常量
    judge_date = 2014
    seg_list = jieba.cut(people, cut_all=True)
    words_list = []
    for word in seg_list:
        words_list.append(word)
    start = -1
    end = -1
    birth_date = 0
    flag = 0
    # 找到被告人的信息
    for i in range(len(words_list)):
        if words_list[i] == "被告人":
            start = i
            break
    # 找到出生信息
    for j in range(start, len(words_list), 1):
        if words_list[j] == "出生":
            flag = 1
            end = j
            break
        if words_list[j] == "生于":
            if flag == 0:
                flag = 2
            end = j
            break
    # 若flag=0，说明未匹配成功，当作无出生信息记录处理
    # 其他情况则开始锁定出生年份
    if flag == 1:
        for k in range(end, start, -1):
            if words_list[k] == "年":
                if words_list[k-1].isdigit():
                    birth_date = int(words_list[k-1])
                    break
    if flag == 2:
        if words_list[end+2] == "年":
            if words_list[end+1].isdigit():
                birth_date = int(words_list[end+1])
        else:
            for k in range(start, end, 1):
                if words_list[k] == "年":
                    if words_list[k-1].isdigit():
                        birth_date = int(words_list[k - 1])
                        break
    # 计算年龄
    if birth_date > 0:
        age = judge_date - birth_date
        return age
    else:
        return 100


f = open("criminal.json", 'r', encoding="utf-8")
ln = 0
criminal_dic = []
for line in f.readlines():
    ln += 1
    dic = json.loads(line)
    criminal_dic.append(dic)
    '''
    # 判断被告人年龄，筛选被告人未成年的样本
    if "当事人" in dic.keys():
        age_result = age_judge(dic["当事人"])
        if age_result < 18:
            print(dic["id"] + '\t' + dic["docName"]+'\t'+str(age_result))
    '''
    # 由于所有条文的keys不一致，所以进行全部浏览
    for key in dic.keys():
        criminal_text = dic[key]
        word_list = jieba.cut(criminal_text, cut_all=True)  # 全模式分词
        result = pseg.cut(criminal_text)  # 带词性标注的分词
        # 关键词匹配
        if "审理未成年人刑事案件" in criminal_text \
                or "作案时未满" in criminal_text \
                or "作案时未成年" in criminal_text \
                or "犯罪时未成年" in criminal_text \
                or "犯罪时未满" in criminal_text:
            print(dic["id"] + '\t' + dic["docName"])
            break
        else:
            continue
f.close()
print(ln)  # 输出所检测的样本总数

f2 = open("civil.json", 'r', encoding="utf-8")
ln2 = 0
civil_dic = []
for l in f2.readlines():
    ln2 += 1
    dic2 = json.loads(l)
    civil_dic.append(dic2)
    reason = dic2["reason"]
    reason_list = jieba.cut(reason, cut_all=False)  # 精准模式分词
    # 首先筛选出离婚纠纷的条文
    if "离婚" in reason_list:
        for key in dic2.keys():
            civil_text = dic2[key]
            word_list = jieba.cut(civil_text, cut_all=False)  # 精准模式分词
            result = pseg.cut(civil_text)  # 带词性标注分词
            # 关键词匹配
            if "抚养" in civil_text:
                print(dic2["id"] + '\t' + dic2["docName"])
                break
f2.close()
print(ln2)  # 输出检测的样本总数
