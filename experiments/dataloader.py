import os
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from enum import Enum, auto

import torch
from torch import Tensor 
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pytorch_lightning as pl
import torch_geometric.utils as tgu
from torch_geometric.data import Data, DataLoader as GeometricDataLoader
import matplotlib.pyplot as plt

from typing import Optional

nl_to_tensor = lambda nl: torch.tensor(nl).t().contiguous()

def unfold_filter_dataset(X, y, idx, seq_len):
  X_orig = torch.tensor(X).float()
  y = torch.tensor(y).long()
  X = X_orig.unfold(0, seq_len, 1).permute(0,2,1) # switch last 2 axis

  # we filter X, y by idx after we generated the sequences
  # this might not be good because it leaks train / val data into the test set
  # however the dataset is very small so it might not yield enough sequences
  # without this procedure
  X= X[idx]

  # we take the label at the end of the
  # sequence because we do not want to "look ahead"
  y = y[idx+(seq_len-1)] 
  return X, y

class TimeseriesDataset(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self,
      X: np.ndarray,
      y: np.ndarray,
      idx,
      seq_len: int = 1):

        super(TimeseriesDataset, self).__init__()
        self.seq_len = seq_len
        self.idx = idx
        self.X_orig = X
        self.y_orig = y

        self.X, self.y = unfold_filter_dataset(X,y,idx,seq_len)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        item = (self.X[index], self.y[index])
        return item



def error_graph_model():

  # attempt to model faults
  gx_ok = nl_to_tensor([
    [0,3]
  ])

  gx_lg = nl_to_tensor([
    # lg connections
    [0,3],
    [1,3],
    [2,3],
  ])

  gx_ll = nl_to_tensor([
    # ll connections
    [0,0],
    [0,1],
    [0,2],
    [1,0],
    [1,1],
    [1,2],
    [2,0],
    [2,1],
    [2,2],
  ])

  gx_llg = nl_to_tensor([
    [0,3]
  ])
  gx_lll = nl_to_tensor([
    [0,3]
  ])
  gx_lllg = nl_to_tensor([
    [0,3]
  ])

  error_graphs = {
    0:gx_ok,
    1:gx_lg,
    2:gx_ll, # 0011 does not exist, assume it is 0110
    3:gx_llg,
    4:gx_lll,
    5:gx_lllg 
  }
  return error_graphs

def  fc_graph_model():
  # fully connected test graph
  fc_graph = nl_to_tensor([
    [0,0],
    [0,1],
    [0,2],
    [0,3],
    [1,0],
    [1,1],
    [1,2],
    [1,3],
    [2,0],
    [2,1],
    [2,2],
    [1,3],
  ])

  fc_graph = tgu.to_undirected(fc_graph)

  # return error_graphs[y]
  return fc_graph

class GraphDataset(Dataset):
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self,
      X: np.ndarray,
      y: np.ndarray,
      idx,
      seq_len: int = 1):

        super(GraphDataset, self).__init__()

        self.seq_len = seq_len
        self.idx = idx
        self.X_orig = X
        self.y_orig = y

        self.X, self.y = unfold_filter_dataset(X,y,idx,seq_len)
        self.gm = fc_graph_model()


    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):

        x = self.X[index]
        y = self.y[index]

        # we need to reformat x to fit the graph
        # we have the features like this ['Ia', 'Ib', 'Ic', Ig, 'Va', 'Vb', 'Vc', Vg]

        # TODO: modify for sequences > 1
        x_graph = x.reshape(2,4).t().contiguous() # [num_nodes, num_node_features]

        item = Data(
          x=x_graph,
          y=y,
          edge_index=self.gm
        )
        return item




class TimeGraphDataset(Dataset):

    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self,
      X: np.ndarray,
      y: np.ndarray,
      idx,
      seq_len: int = 1):

        super(TimeGraphDataset, self).__init__()

        self.seq_len = seq_len
        self.idx = idx
        self.X_orig = X
        self.y_orig = y

        self.X, self.y = unfold_filter_dataset(X,y,idx,seq_len)
        self.gm = self.time_graph()


    def __len__(self):
        return len(self.idx)

    def time_graph(self):
      sl = self.seq_len
      phases = 3
      id_node = np.arange(sl * phases)
      node_type = id_node % phases
      n_arr = zip(id_node[:-phases], node_type[:-phases]) # remove last layer from being processed

      edges = []
      for id_n, type in n_arr:
        if type == 0 or type == 1: 
          edges.append([id_n, id_n+phases+1]) # connect to the next phase
        if type == 2:
          edges.append([id_n, id_n+1]) # connect to first
      # we have the nodes along time

      return nl_to_tensor(edges)

    def __getitem__(self, index):

        x = self.X[index]
        y = self.y[index]

        # we need to reformat x to fit the graph
        # we have the features like this ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

        i = x[:,0:4].flatten()
        v = x[:,4:8].flatten()
        x_graph = torch.stack((i,v), dim=1) # [num_nodes, num_node_features]

        item = Data(
          x=x_graph,
          y=y,
          edge_index=self.gm
        )
        return item





class DataSetType(Enum):
  BINARY = auto()
  CLASSES = auto()

class OutputFormatType(Enum):
  VECTOR = auto()
  GRAPH = auto()
  

def load_dataset(type: DataSetType):
  path = "/home/shafall/datasets/efd"

  if type == DataSetType.BINARY:
    df = pd.read_csv(f"{path}/detect_dataset.csv")
    df['y'] = df['Output (S)']
    
  elif type == DataSetType.CLASSES:
    df = pd.read_csv(f"{path}/classData.csv")

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

    df['fault_type'] = df['G'].astype('str') + df['C'].astype('str') + df['B'].astype('str') + df['A'].astype('str')
    df['fault_name'] = df['fault_type'].map(faults)
    df['fault_id'] = df['fault_type'].map(faults_as_int)
    df['y'] = df['fault_id']
  
  return df




class EfdVectorDataModule(pl.LightningDataModule):
  def __init__(self,
      dataset: DataSetType,
      shuffle,
      seq_len,
      batch_size,
      test_ratio,
      val_ratio) -> None:
      super().__init__()

      numerical_cols = ['Ia', 'Ib', 'Ic', 'Ig', 'Va', 'Vb', 'Vc', 'Vg']
      df = load_dataset(dataset)
      y = df['y'].values # as np arrays


      # Additional Features
      # Sum of current should be zero
      df['Ig'] = (df['Ia'] + df['Ib'] + df['Ic']) 
      df['Vg'] = np.zeros(len(df))

      X = df[numerical_cols].values # as np arrays

      # Balanced Power System
      # 1. Phase voltages should have equal magnitude
      # 2. Equal displaced phases 
      # 3. with counter clockwise rotation
      

      # Derive the symetrical components
      # (zero sequence, pos, neg)
      

      # encoder decoder lstm
      # https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/

      # avoid look ahead bias!
      # https://stats.stackexchange.com/questions/346907/splitting-time-series-data-into-train-test-validation-sets

      idx = np.arange(len(X)-(seq_len-1)) # subtract the invalid boundary

      idx_train, idx_test, = train_test_split(
        idx, test_size=test_ratio, shuffle=True
      )
      idx_test, idx_val  = train_test_split(
        idx_test, test_size=val_ratio, shuffle=True
      )

      scaler = StandardScaler()
      scaler.fit(X[idx_train]) # fit scaler on training set to not cheat
      X_scaled = scaler.transform(X)  # ... but transform all of the data

      self.df = df
      self.X_scaled = X_scaled
      self.y = y

      sort = not shuffle
      s_f = lambda x: np.sort(x) if sort else x
      self.idx_train = s_f(idx_train)
      self.idx_test = s_f(idx_test) 
      self.idx_val = s_f(idx_val)

      # https://medium.com/keita-starts-data-science/time-series-split-with-scikit-learn-74f5be38489e
      self.seq_len = seq_len
      self.batch_size = batch_size
      self.num_workers = 5

  def complete_dataset(self):
    idx = np.arange(len(self.X_scaled)-(self.seq_len-1)) # subtract the invalid boundary
    return TimeseriesDataset(self.X_scaled, self.y, idx, seq_len=self.seq_len)


  def init_loader(self, ds, y, idx, seq_len):
    return DataLoader(
      TimeseriesDataset(ds, y, idx, seq_len),
      batch_size = self.batch_size,
      shuffle = False,
      num_workers = self.num_workers
    )

  def train_dataloader(self):
    return self.init_loader(self.X_scaled, self.y, self.idx_train, self.seq_len)

  def val_dataloader(self): 
    return self.init_loader(self.X_scaled, self.y, self.idx_val, self.seq_len)

  def test_dataloader(self):
    return self.init_loader(self.X_scaled, self.y, self.idx_test, self.seq_len)


class EfdGraphDataModule(pl.LightningDataModule):
  def __init__(self,
      dataset: DataSetType,
      shuffle,
      seq_len,
      batch_size,
      test_ratio,
      val_ratio) -> None:

      super().__init__()

      numerical_cols = [
        'Ia', 'Ib', 'Ic', 'Ig',
        'Va', 'Vb', 'Vc', 'Vg'
      ]

      df = load_dataset(dataset)

      # Add the ground node
      #TODO: prob. wrong, check 3phase current math
      df['Ig'] = -(df['Ia'] + df['Ib'] + df['Ic']) 
      df['Vg'] = np.zeros(len(df))

      y = df['y'].values # as np arrays
      X = df[numerical_cols].values # as np arrays

      # encoder decoder lstm
      # https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/

      # avoid look ahead bias!
      # https://stats.stackexchange.com/questions/346907/splitting-time-series-data-into-train-test-validation-sets

      idx = np.arange(len(X)-(seq_len-1)) # subtract the invalid boundary

      idx_train, idx_test, = train_test_split(
        idx, test_size=test_ratio, shuffle=True
      )
      idx_test, idx_val  = train_test_split(
        idx_test, test_size=val_ratio, shuffle=True
      )

      scaler = StandardScaler()
      scaler.fit(X[idx_train]) # fit scaler on training set to not cheat
      X_scaled = scaler.transform(X)  # ... but transform all of the data

      self.df = df
      self.X_scaled = X_scaled
      self.y = y

      sort = not shuffle
      s_f = lambda x: np.sort(x) if sort else x
      self.idx_train = s_f(idx_train)
      self.idx_test = s_f(idx_test) 
      self.idx_val = s_f(idx_val)

      # https://medium.com/keita-starts-data-science/time-series-split-with-scikit-learn-74f5be38489e
      self.seq_len = seq_len
      self.batch_size = batch_size
      self.num_workers = 5

  def complete_dataset(self): 
    idx = np.arange(len(self.X_scaled)-(self.seq_len-1)) # subtract the invalid boundary
    return TimeGraphDataset(self.X_scaled, self.y, idx, seq_len=self.seq_len)


  def init_loader(self, ds, y, idx, seq_len):
    return GeometricDataLoader(
      TimeGraphDataset(ds, y, idx, seq_len),
      batch_size = self.batch_size,
      shuffle = False,
      num_workers = self.num_workers
    )

  def train_dataloader(self):
    return self.init_loader(self.X_scaled, self.y, self.idx_train, self.seq_len)

  def val_dataloader(self): 
    return self.init_loader(self.X_scaled, self.y, self.idx_val, self.seq_len)

  def test_dataloader(self):
    return self.init_loader(self.X_scaled, self.y, self.idx_test, self.seq_len)

