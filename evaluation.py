import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import numpy as np

modelused = ['DT','SVM','BP','GRU','LSTM','RGRU']
indicators = ['IN','WN','IWN']
for mo in modelused:
    for ind in indicators:
        regress_path = "./output/{}_regress_{}.csv".format(mo,ind)
        class_path = './output/{}_{}.csv'.format(mo,ind)
        df_cl = pd.read_csv(class_path)
        y1 = df_cl["predict"]
        y2 = df_cl["label"]
        AUC = roc_auc_score(y1,y2)
        print("class:\t {}-{}\t auc:{}".format(mo, ind, AUC), end="\n")
        acc = accuracy_score(y1,y2)
        print("class:\t {}-{}\t acc:{}".format(mo, ind, acc), end="\n")


        df_re = pd.read_csv(regress_path)
        y1 = df_re["predict"]
        y2 = df_re["label"]

        RMSE = np.sqrt(mean_squared_error(y1, y2))
        print("regress:\t {}-{}\t RMSE:{}".format(mo, ind, RMSE), end="\n")
        MAPELoss = (np.abs(y1 - y2) / (np.abs(y2) + 1e-2)).mean()
        print("regress:\t {}-{}\t MAPELoss:{}".format(mo, ind, MAPELoss), end="\n")
        SMAPELoss = (np.abs(y1 - y2) / (np.abs(y1) + np.abs(y2) + 1e-2)).mean()
        print("regress:\t {}-{}\t SMAPELoss:{}".format(mo, ind, SMAPELoss), end="\n")
