# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from preprocessing import EmotionAnalysis, OriginalData
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ea = EmotionAnalysis(positive_words_path="data/positive.txt",
                     negative_words_path="data/negative.txt",
                     stop_words_path="data/stopwords.txt")
od = OriginalData(oil_data_path="data/crudeoil.xlsx",
                  chinese_news_path="data/china5e_news.csv",
                  out_chinese_news_emotion_path="data/chinese_news_emotion.csv")

# Calculate emotional characteristics
od.make_news_emotion(ea)

# -----------------------Class-----------------------
# # 中国基本面
# dataset_C = od.transform(data_source="zh")
# # 美国基本面
# dataset_C = od.transform(data_source="en")
# 中国基本面 + 美国涨跌
dataset_C = od.transform_zh_with_en_label()

# Split the data set
train_C, test_C = train_test_split(dataset_C, test_size=0.2, random_state=777)
train_C = [[[j.numpy().tolist() for j in i] for i in item] for item in train_C]
test_C = [[[j.numpy().tolist() for j in i] for i in item] for item in test_C]

inputs_C, _, labels_C = list(zip(*train_C))
inputs_C = [[c for b in a for c in b[0]] for a in inputs_C]
labels_C = [x[0] for x in labels_C]
C = SVC()
C.fit(inputs_C, labels_C)
inputs_test_C, _, labels_test_C = list(zip(*test_C))
inputs_test_C = [[c for b in a for c in b[0]] for a in inputs_test_C]
labels_test_C = [x[0] for x in labels_test_C]
y_hat_C = C.predict(inputs_test_C)

df = pd.DataFrame({'predict': y_hat_C, 'label': labels_test_C})
df.to_csv("output/SVM_IWN.csv", index=False)

# -----------------------Regress-----------------------
od.train_type = "regress"
# # 中国基本面
# dataset_D = od.transform(data_source="zh")
# # 美国基本面
# dataset_D = od.transform(data_source="en")
# 中国基本面 + 美国涨跌
dataset_D = od.transform_zh_with_en_label()
train_D, test_D = train_test_split(dataset_D, test_size=0.2, random_state=777)
train_D = [[[j.numpy().tolist() for j in i] for i in item] for item in train_D]
test_D = [[[j.numpy().tolist() for j in i] for i in item] for item in test_D]

inputs_D, _, labels_D = list(zip(*train_D))
inputs_D = [[c for b in a for c in b[0]] for a in inputs_D]
labels_D = [x[0] for x in labels_D]

D = SVR()
D.fit(inputs_D, labels_D)
inputs_test_D, _, labels_test_D = list(zip(*test_D))
inputs_test_D = [[c for b in a for c in b[0]] for a in inputs_test_D]
labels_test_D = [x[0] for x in labels_test_D]
y_hat_R_D = D.predict(inputs_test_D)

df = pd.DataFrame({'predict': y_hat_R_D, 'label': labels_test_D})
df.to_csv("output/SVM_regress_IWN.csv", index=False)