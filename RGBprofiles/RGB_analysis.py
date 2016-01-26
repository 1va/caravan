"""
Description:    Supervised clasification of caravans from the RGB profiles
Author:         Iva
Date:           07/01/2016
Python version: 2.7
"""
import numpy as np
from time import ctime
SEED=4567
k = 8           # linear kernel bandwidth - optimized through cross-validation => 8 (results comperable for values cca 5-13)

# load the data:
RGB_coords = np.genfromtxt('RGBprofiles/RGB_coords.csv', delimiter=',', skip_header= False, dtype='float')
RGB_profiles = np.genfromtxt('RGBprofiles/RGB_profiles.csv', delimiter=',', skip_header= False, dtype='float')
y = np.array(RGB_coords[:,0], dtype='int8').ravel()
n = y.shape[0]
mm = RGB_profiles.shape[1]/3

# center the histograms (preprocesing option, default=False):
'''
for i in range(n):
  X = RGB_profiles[i,]
  shift = round((mm-1)/2. - 1.0*sum(X*np.concatenate((range(mm),range(mm),range(mm))))/sum(X))
  newX = np.zeros(X.shape)
  newX[max(0,-shift):min(3*mm,3*mm-shift)] = X[max(0, shift):min(3*mm,3*mm+shift)]
  RGB_profiles[i,] = newX  
'''

# smooth the histograms using linear kernel:
from  scipy.signal import convolve
kernel = np.array(range(1,k+1)+range(k-1,0,-1),dtype='float32')/(k*k)

mm = RGB_profiles.shape[1]/3
m = mm-2*k+2
X = np.zeros([n,3*m], dtype='float32')
for i in range(n):
    for j in range(3):
       X[i,(j*m):((j+1)*m)] = convolve(RGB_profiles[i,(j*mm):((j+1)*mm)], kernel, mode='valid')
print('           ',ctime())
print('*** Starting analysis of RGB profiles (smooth using linear kernel of bandwidth %d).' % k)

# now we have data ready as X & y, next validation split:
from sklearn.cross_validation import StratifiedKFold, cross_val_predict
cv = StratifiedKFold(y, 10, shuffle=True, random_state=SEED)

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
'''
print(pca.explained_variance_ratio_)
'''
# Logistic regression with L2 regularization
from sklearn.linear_model import LogisticRegression
log_regr = LogisticRegression(penalty='l2', random_state=SEED)
log_regr.fit(X,y)
coef = abs(log_regr.coef_).ravel()
feat = np.argsort(-coef)[1:12]
print('*** Logistic Regression (l2 penalty) - feature importance by color: red= %.3f, green = %.3f, blue = %.3f' %
      (sum(coef[0:m]), sum(coef[m:(2*m)]), sum(coef[(2*m):(3*m)])))
print(feat, coef[feat])
y_log_regr_full = cross_val_predict(log_regr, X, y, cv=cv) #, scoring='f1')
y_log_regr_pca = cross_val_predict(log_regr, X_pca, y, cv=cv) #, scoring='f1')
print_score(y_log_regr_full ,y)
print_score(y_log_regr_pca ,y)

# Random forrest with entropy criterion
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy', random_state=SEED)
rf.fit(X,y)
feat = np.argsort(-rf.feature_importances_)[1:12]
print('*** Random Forest (entropy criterion) - feature importance by color: red= %.3f, green = %.3f, blue = %.3f' %
      (sum(rf.feature_importances_[0:m]), sum(rf.feature_importances_[m:(2*m)]), sum(rf.feature_importances_[(2*m):(3*m)])))
print(feat, rf.feature_importances_[feat])
y_rf_full = cross_val_predict(rf, X, y, cv=cv)
y_rf_pca = cross_val_predict(rf, X_pca, y, cv=cv)
print_score(y_rf_full ,y)
print_score(y_rf_pca ,y)

# Quadratic discriminant analysis ( = added covariance to Gaussian Naive Bayes ( = added diagonal variance to Linear Discriminant Analysis))
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
qda = QuadraticDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis()
gnb = GaussianNB()
#gnb.fit(X,y)
print('*** Quadratic Discriminant Analysis: ')
y_qda_full = cross_val_predict(qda, X, y, cv=cv)
y_qda_pca = cross_val_predict(qda, X_pca, y, cv=cv)
print_score(y_qda_full, y)
print_score(y_qda_pca, y)
print('*** Linear Discriminant Analysis: ')
y_lda_full = cross_val_predict(lda, X, y, cv=cv)
y_lda_pca = cross_val_predict(lda, X_pca, y, cv=cv)
print_score(y_lda_full, y)
print_score(y_lda_pca, y)
print('*** Gaussian Naive Bayes: ')
y_gnb_full = cross_val_predict(gnb, X, y, cv=cv)
y_gnb_pca = cross_val_predict(gnb, X_pca, y, cv=cv)
print_score(y_gnb_full, y)
print_score(y_gnb_pca, y)

# Linear Support vector machine
from sklearn.svm import SVC, LinearSVC
'''
lsvm = LinearSVC(C=1, penalty='l2', random_state=SEED)
lsvm.fit(X,y)
coef = abs(lsvm.coef_).ravel()
feat = np.argsort(-coef)[1:12]
print('*** Linear Support Vector Machine (l2 penalty) - feature importance by color: red= %.3f, green = %.3f, blue = %.3f' %
      (sum(coef[0:m]), sum(coef[m:(2*m)]), sum(coef[(2*m):(3*m)])))
print(feat, coef[feat])
y_lsvm_full = cross_val_predict(lsvm, X, y, cv=cv)
y_lsvm_pca = cross_val_predict(lsvm, X_pca, y, cv=cv)
print_score(y_lsvm_full, y)
print_score(y_lsvm_pca, y)
'''
lsvm = LinearSVC(C=1, penalty='l1', random_state=SEED, dual=False)
lsvm.fit(X,y)
coef = abs(lsvm.coef_).ravel()
feat = np.argsort(-coef)[1:12]
print('*** Linear Support Vector Machine (l1 penalty) - feature importance by color: red= %.3f, green = %.3f, blue = %.3f' %
       (sum(coef[0:m]), sum(coef[m:(2*m)]), sum(coef[(2*m):(3*m)])))
print(feat, coef[feat])
y_lsvm_full = cross_val_predict(lsvm, X, y, cv=cv)
y_lsvm_pca = cross_val_predict(lsvm, X_pca, y, cv=cv)
print_score(y_lsvm_full, y)
print_score(y_lsvm_pca, y)

'''
# Kernel Support vector machine
svm = SVC(C=1, kernel='rbf', random_state=SEED, penalty='l1')
y_svm_full = cross_val_predict(svm, X, y, cv=cv)
y_svm_pca = cross_val_predict(svm, X_pca, y, cv=cv)
print_score(y_svm_full, y)
print_score(y_svm_pca, y)
'''
# time consuming svm gridsearch
print('*** Parameter grid search for SVM (linear and rbf, by C -> accuracy)')
from sklearn.grid_search import GridSearchCV
param_grid = [ {'C': [.01, .1, 1, 10], 'penalty': ['l1','l2']} ]
svmmodel = GridSearchCV(LinearSVC(random_state=SEED,dual=False), param_grid, cv=cv, scoring='accuracy')
svmmodel.fit(X, y)
print(svmmodel.grid_scores_)
param_grid = [ {'C': [.01, .1, 1, 10]}]
svmmodel = GridSearchCV(SVC(random_state=SEED), param_grid, cv=cv, scoring='accuracy')
svmmodel.fit(X, y)
print(svmmodel.grid_scores_)

print('           ',ctime())

