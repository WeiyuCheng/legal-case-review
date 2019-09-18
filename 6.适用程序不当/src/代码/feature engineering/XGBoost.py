from DataRead import *
from FeatureExtract import *
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(x, l, test_size=test_size, random_state=seed)
# X_train,X_test,y_train,y_test =train_test_split(x,l,test_size=0.2)

print('Start training...')
# 创建模型，训练模型

model = XGBClassifier()
model.fit(X_train, y_train)
eval_set = [(X_test, y_test)]
y_pred = model.predict(X_test)

print(model.feature_importances_)
print(y_pred)


 
recall = recall_score(y_pred,y_test)
precision = precision_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test)
print('The #accuracy is:',np.mean( y_pred == y_test))
print('The #recall is:',recall) 
print('The #precision is:',precision) 
print('The #f1_score is:',f1) 
