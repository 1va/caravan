"""
Description:    Supervised clasification of caravans from the RGB profiles
Author:         Iva
Date:           07/01/2016
Python version: 2.7
"""
import numpy as np

# load the data:
RGB_coords = np.genfromtxt('RGBprofiles/RGB_coords.csv', delimiter=',', skip_header= False, dtype='float')
RGB_profiles = np.genfromtxt('RGBprofiles/RGB_profiles.csv', delimiter=',', skip_header= False, dtype='float')
y = np.array(RGB_coords[:,0], dtype='int8').ravel()
n = y.shape[0]

# smooth the histograms using linear kernel:
from  scipy.signal import convolve
k = 10           # bandwith size (half) - important parameter to optimize through cross-validation! => 9 ?
kernel = np.array(range(1,k+1)+range(k-1,0,-1),dtype='float32')/(k*k)

mm = RGB_profiles.shape[1]/3
m = mm-2*k+2
X = np.zeros([n,3*m], dtype='float32')
for i in range(n):
    for j in range(3):
       X[i,(j*m):((j+1)*m)] = convolve(RGB_profiles[i,(j*mm):((j+1)*mm)], kernel, mode='valid')

# now we have data ready as X & y, next validation split:
from sklearn.cross_validation import StratifiedKFold, cross_val_predict
cv = StratifiedKFold(y, 10, shuffle=True, random_state=0)

#helper functions for scoring
def cont_table(predictions,labels,toplabel=1):
  """Return list contingency table (as a list) of predictions and labels."""
  predP = predictions == toplabel
  trueP = labels == toplabel
  return np.sum(predP*trueP),np.sum(predP*(1-trueP)), np.sum((1-predP)*trueP),np.sum((1-predP)*(1-trueP))
def print_score(predictions, labels, toplabel=1):
  """Return the F1 score based on dense predictions and labels."""
  TP, FP, FN, TN  = cont_table(predictions, labels, toplabel)
  precision = TP/(TP+FP+.01)
  recall = TP/(TP+FN+.01)
  f1 = 2*precision*recall/(precision+recall+.01)*100
  accu = (TP+TN)/(TP+FP+FN+TN+.01)*100
  print('Accuracy: %.1f%%, F1-score: %.1f%% and contingency table: (TP = %d, FP = %d, FN = %d, TN = %d).' % (accu, f1, TP, FP, FN, TN))


# preprocessing by PCA?
from sklearn.decomposition import PCA
pca = PCA(n_components=.95).fit(X)
X_pca = pca.transform(X)
print('PCA - number of components needed to explain 95%% variability: %d.' % X_pca.shape[1])
print(pca.explained_variance_ratio_)

# Logistic regression with L2 regularization
from sklearn.linear_model import LogisticRegression
log_regr = LogisticRegression(penalty='l2')
log_regr.fit(X,y)
coef = abs(log_regr.coef_).ravel()
feat = np.argsort(-coef)[1:12]
print('Logistic Regression - feature importance by color: red= %.3f, green = %.3f, blue = %.3f' %
      (sum(coef[0:m]), sum(coef[m:(2*m)]), sum(coef[(2*m):(3*m)])))
print(feat, coef[feat])
y_log_regr_full = cross_val_predict(log_regr, X, y, cv=cv) #, scoring='f1')
y_log_regr_pca = cross_val_predict(log_regr, X_pca, y, cv=cv) #, scoring='f1')
print_score(y_log_regr_full ,y)
print_score(y_log_regr_pca ,y)

# Random forrest with entropy criterion
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy')
rf.fit(X,y)
feat = np.argsort(-rf.feature_importances_)[1:12]
print('Random Forest - feature importance by color: red= %.3f, green = %.3f, blue = %.3f' %
      (sum(rf.feature_importances_[0:m]), sum(rf.feature_importances_[m:(2*m)]), sum(rf.feature_importances_[(2*m):(3*m)])))
print(feat, rf.feature_importances_[feat])
y_rf_full = cross_val_predict(rf, X, y, cv=cv)
y_rf_pca = cross_val_predict(rf, X_pca, y, cv=cv)
print_score(y_rf_full ,y)
print_score(y_rf_pca ,y)

# Support vector machine
from sklearn.svm import SVC
svm = SVC()
#qda = GaussianNB()
#gnb.fit(X,y)
print('Support vector machine: ')
y_svm_full = cross_val_predict(svm, X, y, cv=cv)
y_svm_pca = cross_val_predict(svm, X_pca, y, cv=cv)
print_score(y_svm_full, y)
print_score(y_svm_pca, y)

'''
from sklearn.grid_search import GridSearchCV
param_grid = [ {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1.0], 'kernel': ['linear','rbf']} ]
svmmodel = GridSearchCV(SVC(), param_grid, cv=3)
svmmodel.fit(X, y)
print(svmmodel.best_params_)
'''

# Quadratic discriminant analysis ( = Gaussian naive Bayes + nontrivial covariance)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
qda = QuadraticDiscriminantAnalysis()
#qda = GaussianNB()
#gnb.fit(X,y)
print('Quadratic discriminant analysis: ')
y_qda_full = cross_val_predict(qda, X, y, cv=cv)
y_qda_pca = cross_val_predict(qda, X_pca, y, cv=cv)
print_score(y_qda_full, y)
print_score(y_qda_pca, y)
