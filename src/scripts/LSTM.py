"""
Handles the LSTM objects in case we wanted to run it onto our computers.
This was made on a Google Colab and was essentially copied and pasted to scripts.
It's highly recommended that this is ran on an L4 instance rather than a computer.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

class DLConsumptionForecaster(nn.Module):
  """
  Deep Learning Weather Forecaster for M2 Project
  """
  def __init__(self, features, hidden: int, num_layers: int = 2, bidirectional: bool = False):
    super(DLConsumptionForecaster, self).__init__()
    self.model = nn.LSTM(
        features,
        hidden_size=hidden,
        num_layers=num_layers,
        batch_first=True,
        dropout=0.3,
        bidirectional=bidirectional
        )
    D = 1 if bidirectional is False else 2
    self.outputs = nn.Linear(
        hidden*D,
        1
    )

  def forward(self, x: torch.Tensor):
    x, *_ = self.model(x)
    x = x[:, -1, :]
    return self.outputs(x)

  @torch.no_grad()
  def predict(self, x):
    return self(x)
  
class ConsumptionTrainerConfig:
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)

  def __getattr__(self, name):
    return None
  
class ConsumptionForecasterTrainer:
  def __init__(self, model, train_data, val_data, test_data, config):
    self.device = config.device
    self.model = model
    self.model.to(self.device)
    self.train_data = train_data
    self.val_data = val_data
    self.test_data = test_data

    self.optimizer = config.optimizer(self.model.parameters(), config.learning_rate)
    self.scaler = config.scaler
    self.epochs = config.epochs
    self.loss_fn = config.loss_fn

  def _train_loop(self, epoch: int):
    self.model.train()
    train_loader = tqdm(self.train_data, desc=f"Training: Epoch {epoch}/{self.epochs}", leave=False)
    for contexts, targets in train_loader:
      contexts, targets = contexts.to(self.device), targets.to(self.device)
      with torch.autocast(self.device):
        outputs = self.model(contexts)
        loss = self.loss_fn(outputs, targets)

      self.optimizer.zero_grad()
      self.scaler.scale(loss).backward()
      self.scaler.step(self.optimizer)
      self.scaler.update()

      # self.global_step += 1
      # if (self.global_step%50==0) and (self.summary_writer is not None):
      #   self._log_loss(loss, 'Training')

      train_loader.set_postfix(Raw=f'{loss.item():.2e}', Normalized=f'{loss.item() / len(contexts):.2e}')
    return None

  @torch.no_grad()
  def _val_loop(self, epoch: int) -> None:
    self.model.eval()
    val_loader = tqdm(self.val_data, desc=f"  Validation: Epoch {epoch}/{self.epochs}", leave=False)
    val_loss = 0.0
    for contexts, targets in val_loader:
      contexts, targets = contexts.to(self.device), targets.to(self.device)
      with torch.autocast(self.device):
        outputs = self.model(contexts)
        loss = self.loss_fn(outputs, targets)
      val_loss += loss.item()
      val_loader.set_postfix(Raw=f'{loss.item():.2e}', Normalized=f'{loss.item() / len(contexts):.2e}')
    # val_loss /= len(val_loader)
    # if self.summary_writer is not None:
    #   self._log_loss(val_loss, 'Validation', epoch)
    #   i = torch.randint(0, len(contexts), [1]).item()
    return None


  def train(self):
    epoch_load = tqdm(range(self.epochs), leave=False)
    for i in epoch_load:
      self._train_loop(i+1)
      self._val_loop(i+1)
    return None

  def evaluate(self):
    test_loss = 0.0
    for contexts, targets in self.test_data:
      contexts, targets = contexts.to(self.device), targets.to(self.device)
      with torch.autocast(self.device):
        outputs = self.model(contexts)
        loss = self.loss_fn(outputs, targets)
      test_loss += loss.item()
    return test_loss / len(self.test_data)

  def get_model(self):
    return self.model
  
class ConsumptionDataset(Dataset):
  def __init__(self, df: np.ndarray, seq_length: int):
    sequences, targets = self._get_sequences(df, seq_length)
    self.sequences = sequences
    self.targets = targets

  def __len__(self):
    assert len(self.sequences) == len(self.targets)
    return len(self.sequences)

  def __getitem__(self, idx):
    return self.sequences[idx], self.targets[idx]

  def _get_sequences(self, df: pd.DataFrame, seq_length: int):
    sequences = []
    targets = []
    for i in tqdm(range(0, len(df) - seq_length)):
      seq = df[i:i+seq_length, :-1]
      seq = torch.Tensor(seq)
      tar = df[i+seq_length, -1]
      tar = torch.Tensor([tar])
      sequences.append(seq)
      targets.append(tar)
    return sequences, targets