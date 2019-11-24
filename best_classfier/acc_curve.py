import scipy.io as scio
import xgboost as xgb
from matplotlib import pyplot as plt
import numpy as np
from sklearn.externals import joblib

dataFile = r'D:\python\DHS\特征\CEPZ_98.mat'
data = scio.loadmat(dataFile)
feature=data['CEPZ_98']

labelFile=r'D:\pseKNC\DHS\label.mat'
label=scio.loadmat(labelFile)
y=label['label']
rankFile = r'D:\pseKNC\DHS\indices.mat'
indices= scio.loadmat(rankFile)
indices=indices['indices']



acc=[];
model=joblib.load('XGBoost-CEPZ.model')

for k in range(1,98):
    X = feature[:,0:k]
    for j in range(1,6):
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
        
        XGB=model.fit(X_train, y_train)
        ans = XGB.predict(X_test)
        TP=0;TN=0;ACC=[];
        for j in range(len(y_test)):
            if ans[j]==1 and y_test[j]==1:
                TP+=1
            if ans[j]==0 and y_test[j]==0:
                TN+=1
        ACC.append((TP+TN)/len(y_test))
    acc.append(np.mean(ACC))
    
print("max_acc=",max(acc),"index=",np.argmax(acc))
scio.savemat('acc_98.mat', {'acc':acc })
