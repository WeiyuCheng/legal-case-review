#Inital format
import csv
import json
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dict_crime_first_instances = {} #{'id':{'当事人':'', '审理经过':'',  '本院认为':'', '裁判结果':''}...}
dict_crime_first_instance_tmp = {'当事人':'', '审理经过':'',  '本院认为':'', '裁判结果':'','公诉机关称':''} #提取模板
filename = input("请输入刑事案例文件的文件地址：")
#filename = u'C://Users//BenWong//Desktop//yanyan//project//Justice-project//test_1//criminal.json'
file = open(filename,'r', encoding = 'utf-8')

while 1:
    line = file.readline()
    if not line:
        break
    try: # right = 3842
        if(json.loads(line)['procedureId'] == "一审"):
            dict_crime_first_instances[json.loads(line)['id']] = {} #{...{'id':{}}}
            for key in dict_crime_first_instance_tmp.keys():
                #防止不匹配导致提取信息不足
                try:
                    dict_crime_first_instances[json.loads(line)['id']][key] = json.loads(line)[key]
                except:
                    continue
    except: # wrong = 0
        pass
    
#刑事一审诉请提取
dict_crime_first_instance_appeal = {}
#从"审理经过"中初步提取
error_list1 = []    #错误列表1

for key, value in dict_crime_first_instances.items():
    try: # right = 3795
        #{...'id':'诉请句'}；用的是审理经过的value作为string list
        dict_crime_first_instance_appeal[key] = re.findall(r"指控被告人[^。]*犯[^。]+罪(?=[^。]*?(?=。))|被告人[^。]*罪(?=[^。]*?(?=。))|被告人[^。]*某.*一案(?=[^。]*?(?=。))|被告人[^。]*一案(?=.*?(?=。))|指控[^。]*罪(?=.*?(?=。))", value['审理经过']) 
    except: # wrong = 25
        error_list1.append((key, value))
#这里之所以 wrong+right ！= 上面的3840，是因为数据集中存在一些重复id号，我们把它当成同一文书处理

#将已经提取过的诉求进行简化使其只包含关键词
error_list2 = []       #错误列表2

appeal_temp = {}       #用来临时保存列表的一个字典
for key in dict_crime_first_instances.keys():
    appeal_temp[key] = []

for key, value in dict_crime_first_instance_appeal.items():
    for calendar in value:
        sub_calendars = re.findall(r"(?<=犯).*罪|(?<=某).*(?=一案)|(?<=某).*罪|(?<=涉嫌).*罪|(?<=因).*(?=一案)|(?<=被告人).*(?=一案)", calendar)
        for sub_calendar in sub_calendars:
            if sub_calendar not in appeal_temp[key]:
                appeal_temp[key].append(sub_calendar)                

for key, value in appeal_temp.items():
    if value == []: # wrong = 28
        error_list1.append((key, dict_crime_first_instances[key]))
        #if '审理经过' in dict_crime_first_instances[key].keys():
        #    print(dict_crime_first_instances[key]['审理经过'])
        # 这里是检查那些有“审理经过”项的，发现无法通过新的正则表达式识别，所以无需改动

# key:id  value: 罪名列表
dict_crime_first_instance_appeal = appeal_temp

#对于错误列表1中的尝试从“当事人”项中提取并放入到诉求中
for i in error_list1[:]:
    try:
        #i[0]为id
        dict_crime_first_instance_appeal[i[0]] = re.findall(r"因涉嫌.*(?=现羁押.*)|因涉嫌.*(?=现关押.*)|证据指控被告人.*构成.*罪|证据指控被告人.*犯.*罪|认为被告人.*构成.*罪|起诉书指控.*犯.*罪", i[1]['当事人'])
        if dict_crime_first_instance_appeal[i[0]] != []:   # right = 26
            error_list1.remove(i)
        else:  # wrong = 13
            #print(i[1]['当事人'])
            #发现这里出来的当事人，凭人也是无法判断的，所以认为这部分已经无法用更好的正则表达式表示
            pass
    except:
        pass
    
#对于错误列表1中的尝试从“公诉机关称”项中提取并放入到诉求中
for i in error_list1[:]:
    try:  # right = 6
        dict_crime_first_instance_appeal[i[0]] = re.findall(r"(?<=指控)[^。]*犯[^。]*罪|因涉嫌.*(?=现关押)|证据指控被告人[^。]*构成[^。]*罪|证据指控被告人[^。]*犯[^。]*罪|认为被告人[^。]*构成[^。]*罪|起诉书指控[^。]*犯[^。]*罪|认为被告人[^。]*罪", i[1]['公诉机关称'])
        #print(dict_crime_first_instance_appeal[i[0]])
        #print(i[1]['公诉机关称'])
        #用于检查对于“公诉机关称的提取”，发现可以提取完剩下所有含有“公诉机关称”的项
        error_list1.remove(i)
    except:  # wrong = 21（这里错误的原因单纯是因为没有“公诉机关称”）
        pass
    
#这一部分代码的作用是为了进行可视化，可以看到有多少长句，所有罪名的均值情况，罪名总数
sns.set(color_codes = True)
f, axes = plt.subplots(1, 2, figsize = (10,5), sharex = False)

string_length = 0
num = 0
num_long_string = 0
threshold = 100
string_len = []
for key, value in dict_crime_first_instance_appeal.items():
    for i in value:
        string_length += len(i)
        num += 1
        string_len.append(len(i))
        if len(i) > threshold:
            num_long_string += 1
avg = string_length/num
print('average:', avg,'total number:', num,'Number of long string:',num_long_string,'(threshold:', threshold,')')

x = np.array(string_len)
sns.kdeplot(x, shade = True, ax = axes[0])

#刑事一审回应提取
dict_crime_first_instance_reply = {}

error_list3 = []

for key, value in dict_crime_first_instances.items():
    try: # right = 3812
    #{...'id':'诉请句'}；用的是审理经过的value作为string
    #由于裁判结果一般都较为规整，所以匹配的时候以；和。作为分隔可以很好地提取出罪名，在这一步中，只要是有“裁判结果”的项，都可以成功提取，而且也没有长句
        dict_crime_first_instance_reply[key] = re.findall(r"；[^；。]*犯[^；。]+罪|某[^；。]*犯[^；。]+罪(?=判处)|犯[^；。]+罪", value['裁判结果'])
    except: # wrong = 8
        error_list3.append((key,value))
        
#剩下的部分从“本院认为”中去处理，可以完全提取 
for i in error_list3[:]:
    try:  # right = 8
        dict_crime_first_instance_reply[i[0]] = re.findall(r"构成[^；。]*罪", i[1]['本院认为'])
        error_list3.remove(i)
    except: # wrong = 0
        pass
    
string_length = 0
num = 0
num_long_string = 0
threshold = 50
string_len = []
for key, value in dict_crime_first_instance_reply.items():
    for i in value:
        string_length += len(i)
        num += 1
        string_len.append(len(i))
        if len(i) > threshold:
            num_long_string += 1
avg = string_length/num
print('average:', avg,'total number:', num,'Number of long string:',num_long_string,'(threshold:', threshold,')')

sns.set(color_codes = True)
x = np.array(string_len)
sns.kdeplot(x, shade = True, ax = axes[1])

plt.show()

filename1 = input('请输入罪名文件的文件地址：')
#filename1 = u'C://Users//BenWong//Desktop//Justice-project//罪名.csv'
csv_reader = csv.reader(open(filename1,'r', encoding='UTF-8-sig'))
accusations = []
for row in csv_reader:
    accusations.append(row[0])

right = 0
total = 0
for key, value in dict_crime_first_instance_appeal.items():
    total += 1
    match_list_appeal = [0 for i in range(len(accusations))]
    match_list_reply = [0 for i in range(len(accusations))]
    for extracted_appeal in value:
        for i in range(len(accusations)):
            if (re.search(accusations[i], extracted_appeal) != None):
                match_list_appeal[i] = 1
    for extracted_reply in dict_crime_first_instance_reply[key]:
        for i in range(len(accusations)):
            if (re.search(accusations[i], extracted_reply) != None):
                match_list_reply[i] = 1
    flag = True
    for index in range(len(match_list_appeal)):
        if match_list_appeal[index] == 1:
            if match_list_reply[index] != 1:
                flag = False
                break
    if flag:
        right += 1

print('match:', right, 'total:', total, 'match_rate:', right/total*100, '%' )