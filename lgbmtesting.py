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

tmp = []
for i in range(len(test)):
    l = test[i]
    tmp.append(l)
df = pd.DataFrame(tmp, columns=column)

load = lightgbm.Booster(model_file='lgbm_model.txt')
pred = load.predict(df[column], categorical_feature=categories)
label = []
for p in pred:
    if p > 0.5:
        label.append(1)
    else:
        label.append(0)
print(getMetrics(tag, label, pred))
