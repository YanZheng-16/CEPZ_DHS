##基于Scikit-learn接口的XGBoost分类
import scipy.io as scio
import xgboost as xgb
from matplotlib import pyplot as plt
import numpy as np
from sklearn.externals import joblib

#scio.savemat('saveddata.mat', {'xi': test_label})
dataFile = r'D:\python\DHS\特征\CEPZ.mat'
data = scio.loadmat(dataFile)
X=data['CEPZ']
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
    params = {'learning_rate':0.01,'n_estimators': 500, 'max_depth': 2, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.7, 'colsample_bytree': 0.6, 'gamma': 0.01, 'reg_alpha': 0.5, 'reg_lambda': 1}
    model = xgb.XGBClassifier(**params)
    XGB=model.fit(X_train, y_train)
    # joblib.dump(XGB,'XGBoost-CEPZ.model')
    ans = XGB.predict(X_test)

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


#A=XGB.feature_importances_
#scio.savemat('importance.mat', {'importance':A })
