import os
import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
import torchmetrics

from enum import Enum, auto


class MLPFeatureMapping(Enum):
  FOURIER = auto()
  DIRECT = auto()

def fourier_mapping(x, B):
  if B is None:
    return x
  else:
    x_proj = (2. * np.pi * x) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLP(pl.LightningModule):
  def __init__(
    self, 
    feature_mapping,
    n_features, 
    n_classes,
    hidden_size, 
    batch_size,
    dropout, 
    learning_rate,
    criterion
  ):

    super(MLP, self).__init__()

    self.n_features = n_features
    self.n_classes = n_classes
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.dropout = dropout
    self.criterion = criterion
    self.learning_rate = learning_rate
    self.feature_mapping = feature_mapping

    self.f1_metric = torchmetrics.F1(average='macro', num_classes=n_classes)

    mapping_size = 256
    B_gauss = torch.randn((mapping_size, n_features)) * 10
    self.register_buffer("B_gauss", B_gauss)

    self.layers = nn.Sequential(
      nn.Linear(n_features, hidden_size),
      nn.ReLU(),   
      nn.Dropout(dropout), 
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(), 
      nn.Dropout(dropout), 
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, 32),
      nn.ReLU(),
      nn.Linear(32, n_classes)
    )
    

  def forward(self, x):
    if self.feature_mapping == MLPFeatureMapping.FOURIER:
      x = fourier_mapping(x, self.B_gauss)
    return self.layers(x)
  
  def compute_loss(self, batch):
    x, y = batch
    x = x.view(x.size(0), -1)
    y_hat = self(x)
    loss = self.criterion(y_hat, y)

    if self.n_classes == 1:
        preds = (y_hat > 0).float()
    else:
        preds = y_hat.argmax(dim=-1)

    acc = (preds == y).sum().float() / preds.shape[0]
    f1 = self.f1_metric(preds, y)
    return loss, acc, f1

  def training_step(self, batch, batch_idx):
    loss, acc, _ = self.compute_loss(batch)
    self.log('train_loss', loss)
    self.log('train_acc', acc)
    return loss

  def test_step(self, batch, batch_idx):
    loss, acc, f1 = self.compute_loss(batch)
    self.log('test_loss', loss)
    self.log('test_acc', acc)
    self.log('test_f1', f1)
    return loss
  
  def validation_step(self, batch, batch_idx):
    loss, acc, _ = self.compute_loss(batch)
    return (loss, acc)

  def validation_epoch_end(self, outs):
    # outs is a list of whatever you returned in `validation_step`
    outs_arr = torch.tensor(outs)
    loss = outs_arr[:,0].mean()
    acc  = outs_arr[:,1].mean()
    self.log("val_loss", loss)
    self.log("va_acc", acc)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(
      self.parameters(), 
      lr=self.learning_rate,
      weight_decay=1e-5
    )
    return optimizer


