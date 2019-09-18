from DataRead import *
from FeatureExtract import *
import jieba
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
svc = SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

recall = recall_score(y_pred,y_test)
precision = precision_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test)
print(y_pred)
print('The #accuracy is:',np.mean( y_pred == y_test))
print('The #recall is:',recall) 
print('The #precision is:',precision) 
print('The #f1_score is:',f1) 
