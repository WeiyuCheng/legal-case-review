from util import *

#下面三个dict存储民事中三个类型的案件
dict_civil_instances_1 = {}
dict_civil_instances_2 = {}
dict_civil_instances_3 = {}
dict_civil_instances_tmp = {'原告诉称':'', '审理经过':'',  '本院认为':'', '裁判结果':'', '申诉人诉称':'','上诉人诉称':'', 'caseType':''} #提取模板
filename = input("请输入民事案例文件的文件地址：")
#filename = u'C://Users//BenWong//Desktop//yanyan//project//Justice-project//test_1//civil.json'
file = open(filename,'r', encoding = 'utf-8')

###分出一审，二审，再审
not_civil = []
while 1:
    line = file.readline()
    if not line:
        break
    if json.loads(line)['caseType'] != "民事":
        not_civil.append(json.loads(line)['id'])
        continue

    if(json.loads(line)['procedureId'] == "一审"):
        dict_civil_instances_1[json.loads(line)['id']] = {} #{...{'id':{}}}
        for key in dict_civil_instances_tmp.keys():
            #防止不匹配导致提取信息不足
            try:
                dict_civil_instances_1[json.loads(line)['id']][key] = json.loads(line)[key]
            except:
                continue
    elif (json.loads(line)['procedureId'] == "二审"):
        dict_civil_instances_2[json.loads(line)['id']] = {}  # {...{'id':{}}}
        for key in dict_civil_instances_tmp.keys():
            # 防止不匹配导致提取信息不足
            try:
                dict_civil_instances_2[json.loads(line)['id']][key] = json.loads(line)[key]
            except:
                continue
    else:
        dict_civil_instances_3[json.loads(line)['id']] = {}  # {...{'id':{}}}
        for key in dict_civil_instances_tmp.keys():
            # 防止不匹配导致提取信息不足
            try:
                dict_civil_instances_3[json.loads(line)['id']][key] = json.loads(line)[key]
            except:
                continue

#确定案件数目
all_1 = len(dict_civil_instances_1)
all_2 = len(dict_civil_instances_2)
all_3 = len(dict_civil_instances_3)
print("各类数目：")
print("num of not_civil：" + str(len(not_civil)))
print("num of first：" + str(all_1))
print("num of second：" + str(all_2))
print("num of review：" + str(all_3))



###一审
easy_1 = {}     #驳回类
easy_1_plus ={} #撤回类
error_1 = {}    #没有裁判结果
error_11 ={}   #complex中不匹配
error_111 = {} # complex中没有原告诉称或裁判结果;本来有一堆，后来发现全都不是民事
complex_appeal_1 = {}   #complex的appeal内容
complex_reply_1 = {}    #complex的reply内容
class_appeal_1 = {}     #appeal的类别
class_reply_1 = {}      #reply的类别

dict_civil_instances_1 = easy_case1(dict_civil_instances_1, easy_1,error_1) #处理完easy类剩下的内容
#dict_civil_instances_1 = easy_case2(dict_civil_instances_1, easy_1_plus)
print("简易类：")
print("一审中驳回数目：" + str(len(easy_1.keys())) + "  一审中撤回类数目：0"+ "  一审中剩余数目：" + str(len(dict_civil_instances_1.keys()))+"   error: "+ str(len(error_1)))

format_1 = r"(?<=要求被告)[^。]*|(?<=请求判令)[^。]*|(?<=请求判决)[^。]*|(?<=请求人民法院判决)[^。]*|(?<=请求人民法院判令)[^。]*|(?<=要求：)[^。]*|(?<=诉讼请求：)[^。]*|(?<=请求法院判令)[^。]*"
main(dict_civil_instances_1, '原告诉称', '裁判结果', complex_appeal_1, class_appeal_1, class_reply_1, format_1,error_11,error_111)

pop_list_1a = []
#pop_list_1b =[]
solve_error_x(error_1,error_11,complex_appeal_1,class_appeal_1,class_reply_1,pop_list_1a,"原告诉称",format_1)
#solve_error_x(error_1,error_11,complex_appeal_1,class_appeal_1,class_reply_1,pop_list_1b)

match_len_1 = len(dict_civil_instances_1.keys()) - len(error_111) - len(error_11) -len(error_1)
rate_1 = float(100*match_len_1 / len(dict_civil_instances_1.keys()))



###二审
easy_2 = {}     #驳回类
easy_2_plus ={} #撤回类
error_2 = {}    #easy中没有裁判结果
error_22 ={}   #complex中不匹配
error_222 = {} # complex中没有原告诉称或裁判结果
complex_appeal_2 = {}   #complex的appeal内容
complex_reply_2 = {}    #complex的reply内容
class_appeal_2 = {}     #appeal的类别
class_reply_2 = {}

dict_civil_instances_2 = easy_case1(dict_civil_instances_2, easy_2,error_2)
dict_civil_instances_2 = easy_case2(dict_civil_instances_2, easy_2_plus,error_2)
print("二审中驳回数目：" + str(len(easy_2.keys())) + "  二审中撤回类数目：" + str(len(easy_2_plus.keys())) + "  二审中剩余数目：" + str(len(dict_civil_instances_2))+"   error: "+ str(len(error_2)))



format_2 = r"请求二审法院[^。]*|请求撤销原审判决[^。]*|请求撤销一审判决[^。]*|(?<=其上诉的请求是：)[^。]*|(?<=综上，)[^。]*|(?<=上诉请求：)[^。]*|(?<=上诉请求二审法院：)[^。]*|(?<=请求：)[^。]*|(?<=提起上诉称：)[^。]*|(?<=不服原审判决上诉称)[^。]*|(?<=向本院提起上诉，请求)[^。]*|综上[^。]*"
main(dict_civil_instances_2, "上诉人诉称", "裁判结果", complex_appeal_2, class_appeal_2, class_reply_2, format_2, error_22, error_222)


pop_list_2a = []
pop_list_2b = []
solve_error_x(error_2,error_22,complex_appeal_2,class_appeal_2,class_reply_2,pop_list_2a,"上诉人诉称",format_2)
solve_error_xxx(error_222,error_22,complex_appeal_2,class_appeal_2,class_reply_2,pop_list_2b)

match_len_2 = len(dict_civil_instances_2.keys()) - len(error_222) - len(error_22) -len(error_2)
rate_2 = float(100*match_len_2 / len(dict_civil_instances_2.keys()))
###再审
easy_3 = {}     #驳回类
easy_3_plus ={} #撤回类
error_3 = {}    #easy中没有裁判结果
error_33 ={}   #complex中不匹配
error_333 = {} # complex中没有原告诉称或裁判结果
complex_appeal_3= {}   #complex的appeal内容
complex_reply_3 = {}    #complex的reply内容
class_appeal_3 = {}     #appeal的类别
class_reply_3 = {}

dict_civil_instances_3 = easy_case1(dict_civil_instances_3, easy_3, error_3)
dict_civil_instances_3 = easy_case2(dict_civil_instances_3, easy_3_plus,error_3)

print("再审中驳回数目：" + str(len(easy_3.keys())) + "  再审中撤回类数目：" + str(len(easy_3_plus.keys()))+ "  在审中剩余数目：" + str(len(dict_civil_instances_3))+"    error: "+ str(len(error_3)))


format_3 = r"(?<=申请再审称)[^。]*|(?<=综上，请求)[^。]*|(?<=综上请求)[^。]*|(?<=增加再审请求)[^。]*|(?<=申请再审，要求)[^。]*|(?<=申请再审请求：)[^。]*"
main(dict_civil_instances_3, "申诉人诉称", "裁判结果", complex_appeal_3, class_appeal_3, class_reply_3, format_3, error_33, error_333)

pop_list_3a = []
pop_list_3b = []
solve_error_x(error_3,error_33,complex_appeal_3,class_appeal_3,class_reply_3,pop_list_3a,"申诉人诉称",format_3)
solve_error_xxx(error_333,error_33,complex_appeal_3,class_appeal_3,class_reply_3,pop_list_3b)

match_len_3 = len(dict_civil_instances_3.keys()) - len(error_333) - len(error_33) - len(error_3)
rate_3 = float(100*match_len_3 / len(dict_civil_instances_3.keys()))

###实验结果
rate_11 = float( 100*(all_1 - len(error_111) - len(error_11) - len(error_1))/all_1 )
rate_22 = float( 100*(all_2 - len(error_222) - len(error_22) - len(error_2))/all_2 )
rate_33 = float( 100*(all_3 - len(error_222) - len(error_33) - len(error_3))/all_3 )
rate_all = float( 100*((all_1 - len(error_111) - len(error_11) - len(error_1))+(all_2 - len(error_222) - len(error_22) - len(error_2))+(all_3 - len(error_222) - len(error_33) - len(error_3)))/(all_1+all_2+all_3) )

print("复杂类：")
print("一审中匹配数目：" + str(match_len_1) + " 匹配率：" + str(rate_1)+"%")
print("二审中匹配数目：" + str(match_len_2) + " 匹配率：" + str(rate_2)+"%")
print("再审中匹配数目：" + str(match_len_3) + " 匹配率：" + str(rate_3)+"%")
print("总体：")
print("一审匹配率：" + str(rate_11)+"%")
print("二审匹配率："+ str(rate_22)+"%")
print("再审匹配率："+ str(rate_33)+"%")
print("民事总体匹配率："+ str(rate_all)+"%")

