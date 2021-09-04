
from enum import Enum

# data processing / linalg
import numpy as np
import pandas as pd

# datascience
from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  confusion_matrix,
  f1_score
)

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# plotting
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader as GeometricDataLoader

pl.seed_everything(1) # help with reproduceability

from dataloader import  (
  EfdGraphDataModule,
  DataSetType,
  EfdVectorDataModule,
  TimeGraphDataset
)

from gcn import GraphLevelGNN
from mlp import MLP, MLPFeatureMapping
from lstm import LstmClassifier
from plots import plot_class_dataset


def report_scores(y_pred, y_true):
  score_avg = 'macro'
  print("accuracy_score: ", accuracy_score(y_pred, y_true))
  print("precision_score", precision_score(y_pred, y_true, average=score_avg))
  print("recall_score", recall_score(y_pred, y_true, average=score_avg))
  print("f1_score", f1_score(y_pred, y_true, average=score_avg))
  print(confusion_matrix(y_pred,y_true))


def train_mlp():
  # model
  params = dict(
      feature_mapping = MLPFeatureMapping.DIRECT,
      n_classes=6,
      n_features=8,
      batch_size=128, 
      criterion=nn.CrossEntropyLoss(),
      hidden_size=8,
      dropout=0.2,
      learning_rate=0.001,
  )

  model = MLP(**params)

  # data
  slen = 1
  edm = EfdVectorDataModule(
    DataSetType.CLASSES,
    seq_len=slen,
    batch_size=params['batch_size'],
    test_ratio=0.2,
    val_ratio=0.2,
    shuffle=True
  )

  # training
  trainer_params = dict(
      precision=32,
      max_epochs = 100,
      # limit_train_batches=0.5,
      gpus=1,
      # log_row_interval=1,
      log_every_n_steps=1,
      progress_bar_refresh_rate=2,
  )

  csv_logger = pl.loggers.CSVLogger('./logs', name='mlp', version='0'),
  tb_logger = pl.loggers.TensorBoardLogger('logs_tb')
  trainer = pl.Trainer(
    logger=tb_logger,
    **trainer_params
  )

  trainer.fit(model, datamodule=edm)
  trainer.test(model, datamodule=edm)

  model.eval()

  n_classes = params['n_classes']
  n_feat = params['n_features']
  ds = edm.complete_dataset()

  ys_hat = torch.zeros((len(ds), n_classes))
  for i, sample in enumerate(ds): # move along window dimension
    x,y = sample
    x = x.view(1, n_feat) # batch size 1
    ys_hat[i,:] = model(x)

  # Plot
  s = lambda t: t[(slen-1):]
  y_pred = torch.exp(ys_hat).argmax(dim=1)

  #%%
  X_orig = ds.X_orig
  y_true = ds.y_orig
  fig = plot_class_dataset(s(X_orig[:,0:3]), s(ds.X_orig[:, 3:6]), y_pred, s(y_true))
  fig.savefig('plots/mlp_results.png')
  report_scores(y_pred, s(y_true))



  return model, params, edm


def train_gcm():
  # model

  n_nodes = 4
  params = dict(
      c_in=2,
      c_hidden=64, # 2 8 16 64
      c_out=2,
      
      num_layers=3,
      layer_name="GCN",
      dp_rate_linear=0.3,
      dp_rate=0.1,

      learning_rate=1e-2,
  )

  model = GraphLevelGNN(**params)

  # data
  slen = 64
  edm = EfdGraphDataModule(
    DataSetType.BINARY,
    seq_len=slen,
    batch_size=32,
    test_ratio=0.2,
    val_ratio=0.2,
    shuffle=True
  )

  # training
  trainer_params = dict(
      precision=32,
      max_epochs = 100,
      # limit_train_batches=0.5,
      gpus=1,
      # log_row_interval=1,
      log_every_n_steps=1,
      progress_bar_refresh_rate=2,
  )

  csv_logger = pl.loggers.CSVLogger('./logs', name='gnn', version='0'),
  tb_logger = pl.loggers.TensorBoardLogger('logs_tb')
  trainer = pl.Trainer(
    logger=tb_logger,
    **trainer_params
  )

  trainer.fit(model, datamodule=edm)
  trainer.test(model, datamodule=edm)

  # evaluate the whole sequence
  model.eval()

  n_classes = params['c_out']
  ds: TimeGraphDataset = edm.complete_dataset()


  ys_hat = torch.zeros((len(ds), n_classes))
  for i, sample in enumerate(ds): # move along window dimension
    sample.batch = torch.zeros(n_nodes*slen).long()
    ys_hat[i,:] = model(sample)


  # Plot
  s = lambda t: t[(slen-1):]
  y_pred = torch.exp(ys_hat).argmax(dim=1)

  #%%
  X_orig = ds.X_orig
  y_true = ds.y_orig
  fig = plot_class_dataset(s(X_orig[:,0:3]), s(X_orig[:,4:7]), y_pred, s(y_true))
  fig.savefig('plots/gcm_results.png')
  report_scores(y_pred, s(y_true))

  return model, params, edm


def train_lstm():
  #%% -----------------------------  LSTM -----------------------------
  # model
  params = dict(
      seq_len = 64, # 128 did not improve the result by much
      n_classes=6,
      n_features=8,
      batch_size=64, 
      criterion=nn.CrossEntropyLoss(),
      hidden_size=16,
      num_layers=1,
      dropout=0.3,
      learning_rate=0.001,
  )

  trainer_params = dict(
      precision=32,
      max_epochs = 100,
      # limit_train_batches=1.5,
      gpus=1,
      # log_row_interval=1,
      log_every_n_steps=1,
      progress_bar_refresh_rate=2,
  )

  model = LstmClassifier(**params)

  # data
  edm = EfdVectorDataModule(
    DataSetType.CLASSES,
    seq_len=params['seq_len'],
    batch_size=params['batch_size'],
    test_ratio=0.2,
    val_ratio=0.2,
    shuffle=False
  )

  # training
  csv_logger = pl.loggers.CSVLogger('./logs', name='lstm', version='0'),
  tb_logger = pl.loggers.TensorBoardLogger('logs_tb')
  trainer = pl.Trainer(
    logger=tb_logger,
    **trainer_params
  )

  trainer.fit(model, datamodule=edm)
  trainer.test(model, datamodule=edm)

  # %% evaluate the whole sequence
  model.eval()

  slen = params['seq_len']
  nfeat = params['n_features']
  n_classes = params['n_classes']
  ds = edm.complete_dataset()

  ys_hat = torch.zeros((len(ds), n_classes))
  for i, sample in enumerate(ds): # move along window dimension
    x,y = sample
    x = x.view(1,slen, nfeat) # batch size 1
    ys_hat[i,:] = model(x)

  # Plot
  s = lambda t: t[(slen-1):]
  y_pred = torch.exp(ys_hat).argmax(dim=1)

  #%%
  X_orig = ds.X_orig
  y_true = ds.y_orig
  fig = plot_class_dataset(s(X_orig[:,0:3]), s(ds.X_orig[:, 3:6]), y_pred, s(y_true))
  fig.savefig('plots/lstm_results.png')
  report_scores(y_pred, s(y_true))

  return model, params, edm


def main():
  # model, params, edm = train_gcm()
  # model, params, edm = train_lstm()
  model, params, edm = train_mlp()


if __name__ == "__main__":
    main()