import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class FCView(nn.Module):
    r"""
    Pytorch view abstraction as nn.module
    """
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        n_b = x.data.size(0)
        x = x.view(n_b, -1)
        return x

    def __repr__(self):
        return 'view(nB, -1)'

class LSTMModel(nn.Module):
  def __init__(
    self, 
    n_features, 
    n_classes,
    hidden_size, 
    num_layers, 
    dropout, 
  ):
    
    super(LSTMModel, self).__init__()

    # neural layers
    # TODO: initialize hidden state
    self.lstm = nn.LSTM(
      input_size=n_features, 
      hidden_size=hidden_size,
      num_layers=num_layers, 
      dropout=dropout, 
      # we expect the first input dimension to be the batch size
      # could also use torch.transpose(tensor_name, 0, 1)
      # https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
      batch_first=True 
    )

    self.linear = nn.Sequential(
      nn.Linear(hidden_size, 64),
      nn.LeakyReLU(inplace=True),
      nn.Linear(64, n_classes)
    )



  # used for inference only
  def forward(self, x):
    """
    # x: inputs of size: [batch_size, seq_len, input_size]
    """
    # we dont need any packing because all sequences have the same length:
    # https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fecV

    #  (h_t itermediate hidden state, h_c long term cell state) inplicitly initialized / saved  
    lstm_out, _ = self.lstm(x) # returns out, hidden
    # out has shape [max_seq_len - context_size + 1, batch_size, lstm_size]
    # class_space = self.hidden2class(lstm_out.view(len(x), -1))
    linear_in = lstm_out[:, -1]
    class_space = self.linear(linear_in)
    return class_space 




class CNN1DModel(nn.Module):
  def __init__(
    self, 
    n_features, 
    n_classes,
    hidden_size, 
    dropout, 
    num_layers, 
  ):

    super(CNN1DModel, self).__init__()

    self.layer1 = self.conv1d(
      in_channels=n_features,
      out_channels=hidden_size,
      dropout=dropout,
      c1d_args=dict(kernel_size=3, stride=2),
      bn_args={}
    )

    self.layer2 = self.conv1d(
      in_channels=hidden_size,
      out_channels= 8 * hidden_size,
      dropout=dropout,
      c1d_args=dict(kernel_size=3, stride=2),
      bn_args={},
      final_layer=False
    )

    self.linear = nn.Sequential(
      nn.AdaptiveMaxPool1d(1),
      FCView(),
      nn.Linear(in_features=8 * hidden_size, out_features=100),
      nn.Linear(in_features=100, out_features=n_classes),
    )

  def conv1d(self,
    in_channels,
    out_channels,
    dropout,
    c1d_args,
    bn_args,
    final_layer=False):

    drlu = [nn.Dropout(dropout), nn.LeakyReLU(True)]

    return  nn.Sequential( 
      nn.Conv1d(in_channels=in_channels, out_channels=out_channels, **c1d_args),   
      nn.BatchNorm1d(num_features=out_channels, **bn_args),
      nn.Dropout(dropout),
      nn.LeakyReLU(True),
      # nn.MaxPool1d(kernel_size=2),  
      *(drlu if not final_layer else [])
    )  

  def forward(self, x):
      x = x.permute(0,2,1)
      x = self.layer1(x)
      x = self.layer2(x) 
      x = self.linear(x)
      return x




class GRUModel(nn.Module):
  def __init__(
    self, 
    n_features, 
    n_classes,
    hidden_size, 
    num_layers, 
    dropout, 
  ):
    super(GRUModel, self).__init__()

    # neural layers
    # TODO: initialize hidden state
    self.gru = nn.GRU(
      input_size=n_features, 
      hidden_size=hidden_size,
      num_layers=num_layers, 
      dropout=dropout, 
      # we expect the first input dimension to be the batch size
      # could also use torch.transpose(tensor_name, 0, 1)
      # https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
      batch_first=True 
    )

    self.linear = nn.Sequential(
      nn.Linear(hidden_size, 64),
      nn.LeakyReLU(inplace=True),
      nn.Linear(64, n_classes)
    )



  # used for inference only
  def forward(self, x):
    """
    # x: inputs of size: [batch_size, seq_len, input_size]
    """
    # we dont need any packing because all sequences have the same length:
    # https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fecV

    #  (h_t itermediate hidden state, h_c long term cell state) inplicitly initialized / saved  
    lstm_out, _ = self.gru(x) # returns out, hidden
    # out has shape [max_seq_len - context_size + 1, batch_size, lstm_size]
    # class_space = self.hidden2class(lstm_out.view(len(x), -1))
    linear_in = lstm_out[:, -1]
    class_space = self.linear(linear_in)
    return class_space 

class LstmClassifier(pl.LightningModule):

  def __init__(
    self, 
    n_classes,
    seq_len, 
    batch_size,
    learning_rate,
    criterion,
    **kwargs
  ):

    super(LstmClassifier, self).__init__()

    self.n_classes = n_classes
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.criterion = criterion
    self.learning_rate = learning_rate

    self.lstm = GRUModel(n_classes=n_classes, **kwargs)

    self.f1_metric = torchmetrics.F1(average='macro', num_classes=n_classes)
  
  def forward(self, x):
    return self.lstm(x)


  def compute_loss(self, batch):
    x, y = batch
    y_hat = self(x)
    loss = self.criterion(y_hat, y)
    preds = y_hat.argmax(dim=-1)
    acc = (preds == y).sum().float() / preds.shape[0]
    f1 = self.f1_metric(preds, y)
    return loss, acc, f1


  def training_step(self, batch, batch_idx):
    loss, acc, _ = self.compute_loss(batch)
    self.log('train_loss', loss)
    self.log('train_acc', acc)
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
  
  def test_step(self, batch, batch_idx):
    loss, acc, f1 = self.compute_loss(batch)
    self.log('test_loss', loss)
    self.log('test_acc', acc)
    self.log('test_f1', f1)
    return loss 

  def configure_optimizers(self):
    # return torch.optim.SGD(self.parameters(), lr=0.1)
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)