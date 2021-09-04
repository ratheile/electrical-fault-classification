# This is a vscode jupyter notebook (#%% magic comment)

#%% system imports
%load_ext autoreload
%autoreload 2

import os
import importlib
from enum import Enum

# data processing / linalg
import numpy as np
import pandas as pd

# datascience
import scipy.signal as sps

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import PolynomialFeatures  # TODO: look at polynomial features

from sklearn.svm import SVC
# from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

# plotting
import matplotlib.pyplot as plt
from plots import plot_class_dataset
import seaborn as sns

#%%
path = "/home/shafall/datasets/efd"
df1 = pd.read_csv(f"{path}/classData.csv")
df = df1
df['fault_type'] = df['G'].astype('str') + df['C'].astype('str') + df['B'].astype('str') + df['A'].astype('str')

numerical_cols = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
faults = {
  '0000': 'OK',
  '1001': 'LG',
  '0110': 'LL', # 0011 does not exist, assume it is 0110
  '1011': 'LLG',
  '0111': 'LLL',
  '1111': 'LLLG'
}

faults_as_int = {
  '0000': 0,
  '1001': 1,
  '0110': 2, # 0011 does not exist, assume it is 0110
  '1011': 3,
  '0111': 4,
  '1111': 5 
}

df1['fault_name'] = df['fault_type'].map(faults)
df1['fault_id'] = df['fault_type'].map(faults_as_int)
df2 = pd.read_csv(f"{path}/detect_dataset.csv")

pd.DataFrame([])



# %% Dataset 1
fig, axes = plt.subplots(3, 1, figsize=(15, 10))
ax = axes[0]
ds = df2
ax.set_title("currents")
ax.plot(ds["Ia"],'r', label='Ia')
ax.plot(ds["Ib"],'b', label='Ib')
ax.plot(ds["Ic"],'g', label='Ic')
ax.legend(loc="upper right")
ax = axes[1]
ax.set_title("voltages")
ax.plot(ds["Va"],'r', label='Va')
ax.plot(ds["Vb"],'b', label='Vb')
ax.plot(ds["Vc"],'g', label='Vc')
ax.legend(loc="upper right")
ax = axes[2]
ax.set_title("binary error")
ax.plot(ds['Output (S)'],'b')
fig.tight_layout()
fig.savefig('plots/binary_dataset.png')
# %%

# %% Dataset 1
fig, axes = plt.subplots(3, 1, figsize=(15, 10))
ds = df1
ax = axes[0]
ax.set_title("currents")
ax.plot(ds["Ia"],'r', label='Ia')
ax.plot(ds["Ib"],'b', label='Ib')
ax.plot(ds["Ic"],'g', label='Ic')
ax.legend(loc="upper right")
ax = axes[1]
ax.set_title("voltages")
ax.plot(ds["Va"],'r', label='Va')
ax.plot(ds["Vb"],'b', label='Vb')
ax.plot(ds["Vc"],'g', label='Vc')
ax.legend(loc="upper right")
ax = axes[2]
ax.set_title("error class label")
ax.plot(ds['fault_name'])
fig.tight_layout()
fig.savefig('plots/class_dataset.png')
# %% Takes a long time to compute

#%% ----------------------------- Spectral Data Analysis -----------------------------

# f, t, Sxx = sps.spectrogram(df1['Vb'], 20000)
f, t, Zxx = sps.stft(df1['Vb'], 2e4, nperseg=20)
#%%
id_fmax = 16
plt.pcolormesh(t, f[0:id_fmax], np.abs(Zxx)[0:id_fmax], shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#%% ----------------------------- Single Class -----------------------------

def fit_performance(classifier, X_train, y_train, X_test, y_test):
  classifier.fit(X_train,y_train)
  y_preds = classifier.predict(X_test)
  print(f'method: {type(classifier)}')
  print("accuracy_score: ", accuracy_score(y_preds, y_test))
  print("precision_score", precision_score(y_preds, y_test))
  print("recall_score", recall_score(y_preds, y_test))
  print(confusion_matrix(y_preds,y_test))


X = df2[numerical_cols]
y = df2['Output (S)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42)
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)


dattrs = [X_train_scaled, y_train, X_test_scaled, y_test]
dattrs_unscaled = [X_train, y_train, X_test, y_test]

fit_performance(RandomForestClassifier(), *dattrs)
fit_performance(DecisionTreeClassifier(), *dattrs)
fit_performance(LogisticRegression(), *dattrs_unscaled)
fit_performance(LogisticRegression(), *dattrs)
fit_performance(SVC(), *dattrs)
fit_performance(KNeighborsClassifier(), *dattrs)
fit_performance(MLPClassifier(), *dattrs)


#%% -----------------------------  Multi Class -----------------------------

X = df1[numerical_cols]
y = df1['fault_id']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
scaled = StandardScaler()
scaled = MinMaxScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)


def fit_performance2(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train,y_train)
    y_preds = classifier.predict(X_test)
    print(f'method: {type(classifier)}')
    print("accuracy_score: ", accuracy_score(y_preds, y_test))
    print("precision_score", precision_score(y_preds, y_test, average = 'macro'))
    print("recall_score", recall_score(y_preds, y_test, average = 'macro'))
    print("f1_score", f1_score(y_preds, y_test, average='macro'))
    print(cm:=confusion_matrix(y_preds,y_test,))
    print("--------------------------------------")
    return cm


dattrs = [X_train_scaled, y_train, X_test_scaled, y_test]
dattrs_unscaled = [X_train, y_train, X_test, y_test]


rfccm = fit_performance2(RandomForestClassifier(), *dattrs)
fit_performance2(DecisionTreeClassifier(), *dattrs)
fit_performance2(SVC(kernel='linear'), *dattrs)
fit_performance2(SVC(kernel='poly'), *dattrs)
fit_performance2(SVC(kernel='rbf'), *dattrs)
fit_performance2(SVC(kernel='sigmoid'), *dattrs)
fit_performance2(MLPClassifier(), *dattrs)
fit_performance2(GaussianNB(), *dattrs)
fit_performance2(KNeighborsClassifier(), *dattrs)


#%%
sns.heatmap(rfccm, annot=True, xticklabels=faults.values(), yticklabels=faults.values())
plt.savefig('plots/rfc_heatmap.png')

# %%
clf = RandomForestClassifier().fit(X_train_scaled, y_train)
X_scaled = scaled.transform(X)
y_pred = clf.predict(X_scaled)
fig = plot_class_dataset(X_scaled[:,0:3], X_scaled[:, 3:6], y_pred, y)
fig.savefig('plots/rfc_results.png')

# %%
clf = MLPClassifier().fit(X_train_scaled, y_train)
X_scaled = scaled.transform(X)
y_pred = clf.predict(X_scaled)
fig = plot_class_dataset(X_scaled[:,0:3], X_scaled[:, 3:6], y_pred, y)
fig.savefig('plots/mlpc_results.png')



#%% -----------------------------  Analyze Logs -----------------------------
metrics = pd.read_csv('./logs/mlp/0/metrics.csv')
train_loss = metrics[['train_loss', 'step', 'epoch']][~np.isnan(metrics['train_loss'])]
val_loss = metrics[['val_loss', 'epoch']][~np.isnan(metrics['val_loss'])]
test_loss_end = metrics['test_loss'].iloc[-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
axes[0].set_title('Train loss per batch')
axes[0].plot(train_loss['step'], train_loss['train_loss'])
axes[1].set_title('Validation loss per epoch')
axes[1].plot(val_loss['epoch'], val_loss['val_loss'], color='orange')
plt.show(block = True)

print('MSE:')
print(f"Train loss: {train_loss['train_loss'].iloc[-1]:.3f}")
print(f"Val loss:   {val_loss['val_loss'].iloc[-1]:.3f}")
print(f'Test loss:  {test_loss_end:.3f}')

