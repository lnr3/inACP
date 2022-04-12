import lightgbm
import numpy as np
import pandas as pd
import src.getFeatures as getFeatures
from sklearn.metrics import roc_curve,accuracy_score,auc,matthews_corrcoef,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

i = 0

def getMetrics(y_true, y_pred, y_proba):
    ACC = accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    CM = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = CM.ravel()
    Sn = tp / (tp + fn)
    Sp = tn / (tn + fp)
    FPR, TPR, thresholds_ = roc_curve(y_true, y_proba)
    AUC = auc(FPR, TPR)

    Results = np.array([ACC, MCC, Sn, Sp, AUC]).reshape(-1, 5)
    Metrics_ = pd.DataFrame(Results, columns=["ACC", "MCC", "Sn", "Sp", "AUC"])

    return Metrics_

file = open('dataset/principal.fasta', 'r')
data = []
tag = []
for line in file:
    line = line.strip('\n')
    if i & 1:
        data.append(line)
    else:
        if line[len(line) - 1] == '0':
            tag.append(0)
        else:
            tag.append(1)
    i += 1

column = []
categories = []
for i in range(0, 50):
    categories.append('residue' + str(i))
    column.append('residue' + str(i))
    column.append('alpha' + str(i))
    column.append('beta' + str(i))
    column.append('coil' + str(i))
    column.append('rsa' + str(i))

for i in range(0, 121):
    column.append('ssa' + str(i))
column.append('label')
test = []
for sequence in data:
    i = 0
    single = [0] * 250
    for ch in sequence:
        if ch == 'A':
            single[i] = 1
        elif ch == 'R':
            single[i] = 2
        elif ch == 'N':
            single[i] = 3
        elif ch == 'D':
            single[i] = 4
        elif ch == 'C':
            single[i] = 5
        elif ch == 'Q':
            single[i] = 6
        elif ch == 'E':
            single[i] = 7
        elif ch == 'G':
            single[i] = 8
        elif ch == 'H':
            single[i] = 9
        elif ch == 'I':
            single[i] = 10
        elif ch == 'L':
            single[i] = 11
        elif ch == 'K':
            single[i] = 12
        elif ch == 'M':
            single[i] = 13
        elif ch == 'F':
            single[i] = 14
        elif ch == 'P':
            single[i] = 15
        elif ch == 'S':
            single[i] = 16
        elif ch == 'T':
            single[i] = 17
        elif ch == 'W':
            single[i] = 18
        elif ch == 'Y':
            single[i] = 19
        elif ch == 'V':
            single[i] = 20
        i += 5
    test.append(single)

import re
secondary = open('dataset/principal.txt', 'r')
i = -1
j = 0
prev = ''
for line in secondary:
    line = line.strip('\n')
    if len(line) < 1 or (line[0] != 'E' and line[0] != 'B'):
        continue
    vec = re.split(r'[ ]+', line)
    if vec[2] != prev:
        i += 1
        j = 0
    prev = vec[2]
    test[i][j + 1] = float(vec[-3])
    test[i][j + 2] = float(vec[-2])
    test[i][j + 3] = float(vec[-1])
    test[i][j + 4] = float(vec[-6])
    j += 5

Test = "./dataset/principal.fasta"
Features, seqs = getFeatures.DRLF_Embed(Test, "dataset/testFeature.csv")
i = 0
for f in Features:
    test[i].extend(f)
    i += 1

load = lightgbm.Booster(model_file='lgbm_model.txt')
pred1 = load.predict(test)

load = lightgbm.Booster(model_file='rf_model.txt')
pred2 = load.predict(test)

import torch
import torch.nn as nn

i = 0

file = open("dataset/principal.fasta", "r")
data = []
label = []
for line in file:
    line = line.strip('\n')
    if i & 1:
        data.append(line)
    else:
        if line[len(line) - 1] == '0':
            label.append(0)
        else:
            label.append(1)
    i += 1

x = torch.zeros(len(data), 50, 20 + 4, dtype=torch.float32)
i = 0
for sequence in data:
    for j in range(len(sequence)):
        ch = sequence[j]
        if ch == 'A':
            x[i][j][0] = 1
        elif ch == 'R':
            x[i][j][1] = 1
        elif ch == 'N':
            x[i][j][2] = 1
        elif ch == 'D':
            x[i][j][3] = 1
        elif ch == 'C':
            x[i][j][4] = 1
        elif ch == 'Q':
            x[i][j][5] = 1
        elif ch == 'E':
            x[i][j][6] = 1
        elif ch == 'G':
            x[i][j][7] = 1
        elif ch == 'H':
            x[i][j][8] = 1
        elif ch == 'I':
            x[i][j][9] = 1
        elif ch == 'L':
            x[i][j][10] = 1
        elif ch == 'K':
            x[i][j][11] = 1
        elif ch == 'M':
            x[i][j][12] = 1
        elif ch == 'F':
            x[i][j][13] = 1
        elif ch == 'P':
            x[i][j][14] = 1
        elif ch == 'S':
            x[i][j][15] = 1
        elif ch == 'T':
            x[i][j][16] = 1
        elif ch == 'W':
            x[i][j][17] = 1
        elif ch == 'Y':
            x[i][j][18] = 1
        elif ch == 'V':
            x[i][j][19] = 1
    i += 1

import re
secondary = open("dataset/principal.txt", "r")
i = -1
j = 0
prev = ''
for line in secondary:
    line = line.strip('\n')
    if len(line) < 1 or (line[0] != 'E' and line[0] != 'B'):
        continue
    vec = re.split(r"[ ]+", line)
    if vec[2] != prev:
        i += 1
        j = 0
    prev = vec[2]
    x[i][j][20] = float(vec[-3])
    x[i][j][21] = float(vec[-2])
    x[i][j][22] = float(vec[-1])
    x[i][j][23] = float(vec[-6])

    j += 1

batch_size = 64
learning_rate = 0.0005

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv = nn.Sequential(
            nn.ZeroPad2d(padding=(0, 16, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(3, 20), stride=(1, 20)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 2), stride=(1, 1)),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3072, 128),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, data):
        x = self.conv(data)
        output = self.fc(x.view(data.shape[0], -1))
        return output

model = torch.load('model_{}_{}.pkl'.format(batch_size, learning_rate))
model.eval()

pred3 = model(torch.unsqueeze(x, dim=1))
y_pred = []
y_pred_prob = []
for i in range(pred3.shape[0]):
    x = []
    y = []
    p = 0
    if pred1[i] > 0.5:
        x.append(pred1[i])
    else:
        y.append(pred1[i])
    if pred2[i] > 0.5:
        x.append(pred2[i])
    else:
        y.append(pred2[i])
    if pred3[i][1].item() > 0.5:
        x.append(pred3[i][1].item())
    else:
        y.append(pred3[i][1].item())
    if len(x) > 1:
        p = sum(x) / len(x)
    else:
        p = sum(y) / len(y)
    #p = (pred1[i] + pred2[i] + pred3[i][1].item()) / 3
    #p = 0.6 * pred1[i] + 0.2 * pred2[i] + 0.2 * pred3[i][1].item()
    y_pred.append(1 if p > 0.5 else 0)
    y_pred_prob.append(p)
print(getMetrics(label, y_pred, y_pred_prob))
