from DataRead import *
from FeatureExtract import *
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# import os
# os.environ["PATH"] += os.pathsep + 'C:/Users/80942.DESKTOP-QHPCKRM/Downloads/release/bin/'
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
from sklearn import tree
from sklearn.externals.six import StringIO
# import pydot


''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(x, l, test_size = 0.2)

''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)

''' 把决策树结构写入文件 '''
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph[0].write_pdf("tree2.pdf")#写入pdf

# with open("tree.pdf", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)
    
''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)

'''测试结果的打印'''
answer = clf.predict(x_test)
#print(x_train)
print(answer)
#print(y_train)



 
recall = recall_score(answer,y_test)
precision = precision_score(answer,y_test)
f1 = f1_score(answer,y_test)
print('The #accuracy is:',np.mean( answer == y_test))
print('The #recall is:',recall) 
print('The #precision is:',precision) 
print('The #f1_score is:',f1) 
