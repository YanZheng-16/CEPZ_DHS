import scipy.io as scio
import xgboost as xgb
from matplotlib import pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import roc_curve,auc

dataFile = r'D:\python\DHS\特征\Feng.mat'
data = scio.loadmat(dataFile)
feature=data['CEPZ']

labelFile=r'D:\python\DHS\特征\Feng.mat'
label=scio.loadmat(labelFile)
y=label['label']

rankFile = r'D:\python\DHS\特征\Feng.mat'
indices= scio.loadmat(rankFile)
indices=indices['indices']

pred=[];y_=[];

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

    model=joblib.load('XGBoost-CEPZ.model')
    XGB=model.fit(X_train, y_train)
    pred_proba = XGB.predict_proba(X_test)[:,1]
    pred.append(pred_proba);y_.append(y_test);

y_pred=np.reshape(pred,(np.size(pred),1))
y_label=np.reshape(y_,(np.size(y_),1))

fpr, tpr, thresholds = roc_curve(y_label, y_pred)
roc_auc = auc(fpr,tpr)


plt.figure(1) # 创建图表1
plt.title('Receiver Operating Characteristic Curve')# give plot a title
plt.xlabel('False Positive Rate')# make axis labels
plt.ylabel('True Positive rate')
plt.plot(fpr, tpr, color='coral')
plt.show()
