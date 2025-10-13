import numpy as np
import torch.nn as nn
import torch
from rl_agent.utils import get_regularization

class BaseModel(nn.Module):
  def __init__(self, reader, params):
    super().__init__()
    self.display_name = "BaseModel"
    self.reader = reader
    self.model_path = params['model_path']
    self.loss_type = params['loss_type']
    self.l2_coef = params['l2_coef']
    self.device = params['device']
    self.sigmoid = nn.Sigmoid()
    self._define_params(reader, params)

  def get_regularization(self, *modules):
    return get_regularization(*modules)

  def do_forward_and_loss(self, feed_dict: dict) -> dict:
    '''
    Used during training to compute predictions and the loss.
    '''
    out_dict = self.get_forward(feed_dict)
    out_dict['loss'] = self.get_loss(feed_dict, out_dict)
    return out_dict

  def forward(self, feed_dict: dict, return_prob=True) -> dict:
    '''
      Used during evaluation/prediction to generate predictions and probabilities
    '''
    out_dict = self.get_forward(feed_dict)
    if return_prob:
      out_dict['probs'] = self.sigmoid(out_dict['preds'])
    return out_dict

  def wrap_batch (self, batch):
    '''
    Build feed_dict from batch data and move data to self.device
    '''
    for k, val in batch.items():
      if type(val).__module__ == np.__name__:
        batch[k] = torch.from_numpy(val)
      elif torch.is_tensor(val):
        batch[k] = val
      elif type(val) is list:
        batch[k] = torch.tensor(val)
      else:
        continue # No compatiable type
      if batch[k].type() == 'torch.DoubleTensor':
        batch[k] = batch[k].type(torch.FloatTensor)
      batch[k] = batch[k].to(self.device)
    return batch

  def save_checkpoint(self):
    torch.save({
        "model_state_dict": self.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
    }, self.model_path + ".checkpoint")

  def load_checkpoint(self, model_path, with_optimizer=True):
    checkpoint = torch.load(model_path + ".checkpoint",
                            map_location=self.device)
    self.load_state_dict(checkpoint["model_state_dict"])
    if with_optimizer:
      self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.model_path = model_path

  def _define_params(self, reader, params):
    pass

  def get_forward(self, feed_dict: dict) -> dict:
    pass

  def get_loss(self, feed_dict: dict, out_dict: dict) -> dict:
    pass