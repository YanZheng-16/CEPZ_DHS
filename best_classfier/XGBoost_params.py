##XGBoost分类调参
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import scipy.io as scio
import numpy as np
import pandas as pd

# read in the data
dataFile = r'D:\python\DHS\特征\CEPZ.mat'
data = scio.loadmat(dataFile)
X=data['CEPZ']
labelFile=r'D:\pseKNC\DHS\label.mat'
label=scio.loadmat(labelFile)
y=label['label']
rankFile = r'D:\pseKNC\DHS\indices.mat'



# model
##{'n_estimators': [100,200,300,400,500,600,700,800,900]}
##{'max_depth': [1,2,3,4,5,6,7,8,9,10],'min_child_weight': [1,2,3,4,5,6,7,8,9,10]}
##{'gamma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]}
##{'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
##{'reg_alpha': [0.05, 0.1,0.5, 1, 2, 3], 'reg_lambda': [0.05, 0.1,0.5, 1, 2, 3]}
##{'learning_rate': [0.01,0.05,0.07,0.1,0.2,0.5,1]}


if __name__ == '__main__':

    cv_params ={'n_estimators': [100,200,300,400,500,600,700,800,900]}
    other_params = {'learning_rate':0.01,'n_estimators': 500, 'max_depth': 2, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.7, 'colsample_bytree': 0.6, 'gamma': 0.01, 'reg_alpha': 0.05, 'reg_lambda': 0.1}

    model = xgb.XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X, y)
    evalute_result = optimized_GBM.cv_results_
    
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
