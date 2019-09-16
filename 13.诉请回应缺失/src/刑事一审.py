#Inital format
import json
import re

dict_crime_first_instances = {} #{'id':{'当事人':'', '审理经过':'',  '本院认为':'', '裁判结果':''}...}
dict_crime_first_instance_tmp = {'当事人':'', '审理经过':'',  '本院认为':'', '裁判结果':'', '公诉机关称':''} #提取模板
filename = input("请输入刑事案例文件的文件地址：")
#filename = u'C://Users//Administrator//Desktop//深度学习入门//把手网数据集//6.1课题四案件评查与减假暂//把手案例网数据_small//dataset_small//criminal.json'
file = open(filename,'r', encoding = 'utf-8')

while 1:
    line = file.readline()
    if not line:
        break
    try:
        if(json.loads(line)['procedureId'] == "一审"):
            dict_crime_first_instances[json.loads(line)['id']] = {} #{...{'id':{}}}
            for key in dict_crime_first_instance_tmp.keys():
                #防止不匹配导致提取信息不足
                try:
                    dict_crime_first_instances[json.loads(line)['id']][key] = json.loads(line)[key]
                except:
                    continue
    except:
        pass

#刑事一审诉请提取
dict_crime_first_instance_appeal = {}
#从"审理经过"中初步提取
error_list1 = []    #错误列表1
for key, value in dict_crime_first_instances.items():
    try:
        #{...'id':'诉请句'}；用的是审理经过的value作为string
        dict_crime_first_instance_appeal[key] = re.findall(r"指控被告人.*犯.*罪|被告人.*罪|被告人.*某.*一案|被告人.*一案|指控.*罪", value['审理经过'])[0]
    except:
        error_list1.append((key, value))
        #print((key,value))


#将已经提取过的诉求进行简化使其只包含关键词]
num = 0
error_list2 = []       #错误列表2
for key, value in dict_crime_first_instance_appeal.items():
    try:
        dict_crime_first_instance_appeal[key] = re.findall(r"(?<=犯).*罪|(?<=某).*一案|(?<=某).*罪|(?<=涉嫌).*罪|(?<=因).*一案|(?<=被告人).*一案", value)[0]
    except:
        error_list2.append((key, value))

#对于错误列表1中的尝试从“当事人”项中提取并放入到诉求中
for i in error_list1[:]:
    try:
        #i[0]为id
        dict_crime_first_instance_appeal[i[0]] = re.findall(r"因涉嫌.*(?=现羁押.*)|因涉嫌.*(?=现关押.*)|证据指控被告人.*构成.*罪|证据指控被告人.*犯.*罪|认为被告人.*构成.*罪|起诉书指控.*犯.*罪", i[1]['当事人'])[0]
        #print(dict_crime_first_instance_appeal[i[0]])#debug
        error_list1.remove(i)     #把error_list1中成功通过“当事人”处理之后的项目移除
    except:
        try:
            error_list2.append((i[0], i[1]['当事人']))
        except:
            pass
        pass

#对于错误列表1中的尝试从“公诉机关称”项中提取并放入到诉求中
for i in error_list1[:]:
    try:
        dict_crime_first_instance_appeal[i[0]] = re.findall(r"(?<=指控).*犯.*罪|因涉嫌.*(?=现关押.*)|证据指控被告人.*构成.*罪|证据指控被告人.*犯.*罪|认为被告人.*构成.*罪|起诉书指控.*犯.*罪", i[1]['公诉机关称'])[0]
        error_list1.remove(i)
    except:
        try:
            error_list2.append((i[0], i[1]['公诉机关称']))
        except:
            pass
        pass
# for i in error_list2:
#     print(i)
#debug

dict_crime_first_instance_reply = {}

error_list3 = []
for key, value in dict_crime_first_instances.items():
    try:
        try:
        #{...'id':'诉请句'}；用的是审理经过的value作为string
            dict_crime_first_instance_reply[key] = re.findall(r"(?<=被告人).*犯.*罪|某.*判处", value['裁判结果'])[0]
        except:
            error_list3.append((key,value))
    except:
        error_list3.append((key, value))
        
#接下来是匹配部分 
#简单地写了一个最暴力的版本，运行会花到8s的时间，之后有时间可能需要改算法
#匹配率97%说明前面的提取部分可能还需要改进，毕竟法律文书匹配率应该不至于只有97%
import csv
filename1 = input('请输入罪名文件的文件地址：')
#filename1 = u'C://Users//Administrator//Desktop//深度学习入门//把手网数据集//6.1课题四案件评查与减假暂//把手案例网数据_small//dataset_small//罪名.csv'
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
