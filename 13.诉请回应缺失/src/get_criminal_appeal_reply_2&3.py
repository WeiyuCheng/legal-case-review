#Inital format
import json
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dict_crime_second_instances = {}
dict_crime_second_instance_tmp = { '上诉人诉称':'',  '公诉机关称':'', '审理经过':'', '裁判结果':''}
#filename = u'C://Users//BenWong//Desktop//yanyan//project//Justice-project//test_1//criminal.json'
filename = input("请输入刑事案例文件的文件地址：")
file = open(filename,'r', encoding = 'utf-8')
while 1:
    line = file.readline()
    if not line:
        break
    try:
        if((json.loads(line)['procedureId'] == "二审")or(json.loads(line)['procedureId'] == "再审")):
            dict_crime_second_instances[json.loads(line)['id']] = {} #{...{'id':{}}}
            for key in dict_crime_second_instance_tmp.keys():
                try:
                    dict_crime_second_instances[json.loads(line)['id']][key] = json.loads(line)[key]
                except:
                    continue
    except:
        pass

print("总体二审与再审个数：" + str(len( dict_crime_second_instances)))

dict_crime_second_instance_appeal_easy = {}
remove_list = []
error_list1 = []
for key, value in dict_crime_second_instances.items():
    try:
        if(re.findall(r"驳回抗诉|驳回上诉|撤回上诉", value['裁判结果']) != []):
            dict_crime_second_instance_appeal_easy[key] = re.findall(r"驳回上诉|驳回抗诉|撤回上诉", value['裁判结果'])
            remove_list.append(key)
        else:
            error_list1.append((key, value))
    except:
        error_list1.append((key, value))
for i in remove_list:
    dict_crime_second_instances.pop(i)

print("可直接判断匹配的案件个数：" + str(len( dict_crime_second_instance_appeal_easy))+ "，" + "需进一步观察的案件个数：" + str(len(dict_crime_second_instances)))

dict_crime_instance_appeal = {}
dict_crime_instance_reply = {}
for i in error_list1[:]:
    try:
        # i[0]为id
        dict_crime_instance_appeal[i[0]] = re.findall(r"减轻处罚|从轻处罚|量刑过重|量刑不当|量刑畸重|量刑畸轻|量刑偏重|立功|违反法定的诉讼程序|不服|请求缓刑|宣告缓刑|适用法律错误|请求二审法院改判|.*为由提出上诉",
                                                      i[1]['上诉人诉称'])
        dict_crime_instance_reply[i[0]] = re.findall(r"撤销|维持|变更",i[1]['裁判结果'])
        if dict_crime_instance_appeal[i[0]] != [] and dict_crime_instance_reply[i[0]] != []:
            error_list1.remove(i)
        else:
            pass
    except:
        pass

for i in error_list1[:]:
    try:
        # i[0]为id
        dict_crime_instance_appeal[i[0]] = re.findall(r"减轻处罚|从轻处罚|量刑过重|量刑不当|量刑畸重|量刑畸轻|量刑偏重|立功|违反法定的诉讼程序|不服|请求缓刑|宣告缓刑|适用法律错误|请求二审法院改判|.*为由提出上诉",
                                                      i[1]['公诉机关称'])
        dict_crime_instance_reply[i[0]] = re.findall(r"撤销|维持|变更",i[1]['裁判结果'])
        if dict_crime_instance_appeal[i[0]] != [] and dict_crime_instance_reply[i[0]] != []:
            error_list1.remove(i)
        else:
            pass
    except:
        pass

for i in error_list1[:]:
    try:
        # i[0]为id
        dict_crime_instance_appeal[i[0]] = re.findall(r"减轻处罚|从轻处罚|量刑过重|量刑不当|量刑畸重|量刑畸轻|量刑偏重|立功|违反法定的诉讼程序|不服|请求缓刑|宣告缓刑|适用法律错误|请求二审法院改判|.*为由提出上诉",
                                                      i[1]['审理经过'])
        dict_crime_instance_reply[i[0]] = re.findall(r"撤销|维持|变更",i[1]['裁判结果'])
        if dict_crime_instance_appeal[i[0]] != [] and dict_crime_instance_reply[i[0]] != []:
            error_list1.remove(i)
        else:
            pass
    except:
        pass

print("其中仍无法匹配数目：" + str(len(error_list1)))