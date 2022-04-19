# -*- coding: utf-8 -*-

import os
import torch

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from model import BPNN
from preprocessing import EmotionAnalysis, OriginalData


# -----------------------Training parameters-----------------------

num_epochs = 50
GPU_id = "cuda:0"   # Specifies the graphics card
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
lr_init = 0.001      # Initial learning rate
weights_path = "pkl/"  # Model save path
weights_save_interval = 5   # Model save interval
random_seed = 273

# -----------------------Pytorch results reproduce-----------------------

# os.environ['PYTHONHASHSEED'] = str(random_seed)
# np.random.seed(random_seed)
# random.seed(random_seed)

torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False

# -----------------------Make train and val data-----------------------

ea = EmotionAnalysis(positive_words_path="data/positive.txt",
                     negative_words_path="data/negative.txt",
                     stop_words_path="data/stopwords.txt")
od = OriginalData(oil_data_path="data/crudeoil.xlsx",
                  chinese_news_path="data/china5e_news.csv",
                  out_chinese_news_emotion_path="data/chinese_news_emotion.csv")

# Calculate emotional characteristics
od.make_news_emotion(ea)

# 中国基本面
# dataset = od.transform(data_source="zh")
# 美国基本面
# dataset = od.transform(data_source="en")
# 中国基本面 + 美国涨跌
dataset = od.transform_zh_with_en_label()

# Split the data set
train, val = train_test_split(dataset, test_size=0.1, random_state=777)

inputs, _, labels = list(zip(*train))
inputs = [[[c for b in a for c in b[0]]] for a in inputs]
input_size = len(inputs[0][0])
# -----------------------Model initialization-----------------------

model = BPNN(input_size=input_size, output_size=32, classification=True)
model = model.to(device)

# -----------------------Loss function、optimizer、learning rate decay-----------------------

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# -----------------------Train-----------------------

model.train(mode=True)
print("Training...")

for epoch in range(num_epochs):
    iteration = 0
    total_loss = 0.0
    preds_train = []
    for input, label in zip(inputs, labels):
        iteration += 1
        optimizer.zero_grad()
        input = torch.tensor(input)
        label = label.to(device)
        input = input.to(device)
        out = model(input)
        loss = criterion(out, label)
        _, out_binary = torch.max(out, 1)
        preds_train.append(out_binary.cpu().tolist()[0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch: {}\tLoss: {}".format(epoch, total_loss / iteration), end="\t")
    print("train_accuracy_score: {}".format(accuracy_score([x.tolist()[0] for x in labels], preds_train)))
    exp_lr_scheduler.step()

    if epoch % weights_save_interval == 0:
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
        torch.save(model.state_dict(), weights_path + "epoch_{}.pkl".format(epoch))


# -----------------------Inference-----------------------

inputs_val, _, labels_val = list(zip(*val))
inputs_val = [[[c for b in a for c in b[0]]] for a in inputs_val]
preds_val = []
labels_val = [x.tolist()[0] for x in labels_val]
model.eval()

for input_val in inputs_val:
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        input_val = torch.tensor(input_val)
        input_val = input_val.to(device)
        out = model(input_val)
        _, out_val = torch.max(out, 1)
        preds_val.append(out_val.cpu().tolist()[0])
df = pd.DataFrame({'predict': preds_val, 'label': labels_val})
df.to_csv('./output/BP_IWN.csv', index=False)
print("-------------------------------")
print("val accuracy_score: {}".format(accuracy_score(labels_val, preds_val)))
print("val auc: {}".format(roc_auc_score(labels_val, preds_val)))
