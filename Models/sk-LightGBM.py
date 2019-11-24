##基于Scikit-learn接口的LightGBM分类
import lightgbm as lgb
import scipy.io as scio
from matplotlib import pyplot as plt
import numpy as np
from sklearn.externals import joblib

dataFile = r'D:\python\DHS\特征\CEPZ.mat'
data = scio.loadmat(dataFile)
X=data['X']
labelFile=r'D:\pseKNC\DHS\label.mat'
label=scio.loadmat(labelFile)
y=label['label']
rankFile = r'D:\pseKNC\DHS\indices.mat'
indices= scio.loadmat(rankFile)
indices=indices['indices']

Sn=[];Sp=[];Acc=[];MCC=[];

for k in range(1,6):
    # 五折
    X_train=[];X_test=[];y_train=[];y_test=[];
    for i in range(len(indices)):
        if indices[i]==k:
            y_test.append(y[i])
            X_test.append(X[i])
        if indices[i]!=k:
            y_train.append(y[i])
            X_train.append(X[i])
    X_train=np.array(X_train);X_test=np.array(X_test)
    y_train=np.array(y_train);y_test=np.array(y_test)
    # 训练模型
    gbm = lgb.LGBMClassifier(learning_rate=0.05,n_estimators=100, num_leaves=10)
    gbm.fit(X_train, y_train)
##    joblib.dump(gbm,'LightGBM-CEPZ.model')
    ans = gbm.predict(X_test)

    # 评价
    TP=0;FN=0;FP=0;TN=0
    for i in range(len(y_test)):
        if ans[i]==1 and y_test[i]==1:
            TP+=1
        if ans[i]==0 and y_test[i]==1:
            FN+=1
        if ans[i]==1 and y_test[i]==0:
            FP+=1
        if ans[i]==0 and y_test[i]==0:
            TN+=1
    Sn.append(TP/(TP+FN))
    Sp.append(TN/(TN+FP))
    Acc.append((TP+TN)/(TP+TN+FP+FN))
    MCC.append((TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))**0.5)
    
total=np.row_stack((Sn,Sp,Acc,MCC))
print(total.mean(axis=1))
