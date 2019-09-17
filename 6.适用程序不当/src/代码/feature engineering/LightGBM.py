from DataRead import reas ,text ,l
from FeatureExtract import x
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
# 加载数据
print('Load data...')

X_train,X_test,y_train,y_test =train_test_split(x,l,test_size=0.2)


print('Start training...')
# 创建模型，训练模型
lgbm= lgb.LGBMClassifier()
lgbm.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric='l1',early_stopping_rounds=5)
lgb.create_tree_digraph(lgbm, tree_index=1)
import matplotlib.pyplot as plt
import matplotlib
fig2 = plt.figure(figsize=(20, 20))
ax = fig2.subplots()
lgb.plot_tree(lgbm._Booster, tree_index=1, ax=ax)
plt.show()   

print('Start predicting...')
# 测试机预测
y_pred = lgbm.predict(X_test, num_iteration=lgbm.best_iteration_)

# feature importances
print('Feature importances:', list(lgbm.feature_importances_))

recall = recall_score(y_pred,y_test)
precision = precision_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test)
# 模型评估
print('The #accuracy is:',np.mean( y_pred == y_test))
print('The #recall is:',recall) 
print('The #precision is:',precision) 
print('The #f1_score is:',f1) 
