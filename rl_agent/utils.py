import random
import numpy as np
import torch
from torch.distributions import Categorical


# @title Support function

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def padding_and_clip(sequence, max_len, padding_direction = 'left'):
    if len(sequence) < max_len:
        sequence = [0] * (max_len - len(sequence)) + sequence if padding_direction == 'left' else sequence + [0] * (max_len - len(sequence))
    sequence = sequence[-max_len:] if padding_direction == 'left' else sequence[:max_len]
    # print(f"sequence{sequence}")
    return sequence

def get_regularization(*modules):
  """
  L2 regularization
  """
  reg = 0.0
  for m in modules:
      for p in m.parameters():
          if p.requires_grad:
              reg += torch.sum(p ** 2)
  return reg

def wrap_batch(batch, device):
  """
  Build feed_dict from batch data and move data to device
  """
  for k,val in batch.items():
    if type(val).__module__ == np.__name__:
        batch[k] = torch.from_numpy(val)
    elif torch.is_tensor(val):
        batch[k] = val
    elif type(val) is list:
        batch[k] = torch.tensor(val)
    else:
        continue
    if batch[k].type() == "torch.DoubleTensor":
        batch[k] = batch[k].float()
    batch[k] = batch[k].to(device)
  return batch

def sample_categorical_action(action_prob, candidate_ids, slate_size,
                              with_replacement=True, batch_wise=False,
                              return_idx=False):
  '''
  @input:
  - action_prob: (B, L)
  - candidate_ids: (B, L) or (1, L)
  - slate_size: K
  - with_replacement: sample with replacement
  - batch_wise: do batch wise candidate selection
  '''
  if with_replacement:
    # (K, B)
    indices = Categorical(action_prob).sample(sample_shape = (slate_size,))
    # (B, K)
    indices = torch.transpose(indices, 0, 1)
  else:
    indices = torch.cat([torch.multinomial(prob, slate_size, replacement=False).view(1, -1) \
                         for prob in action_prob], dim = 0)
  action = torch.gather(candidate_ids, 1, indices) if batch_wise else candidate_ids[indices]
  if return_idx:
    return action.detach(), indices.detach()
  else:
    return action.detach()


##################
#   Learning     #
##################

class LinearScheduler(object):
  def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
    self.schedule_timesteps = schedule_timesteps
    self.final_p = final_p
    self.initial_p = initial_p

  def value(self, t):
    '''
    see Schedule.value
    '''
    fraction = min(float(t) / self.schedule_timesteps, 1.0)
    return self.initial_p + fraction * (self.final_p - self.initial_p)


def dot_scorer(action_emb, item_emb, item_dim):
    '''
    @input:
    - action_emb: (B, i_dim)
    - item_emb: (B, L, i_dim) or (1, L, i_dim)
    @output:
    - score: (B, L)
    '''
    # Ensure item_emb has shape (B, L, i_dim)
    # action_emb: (B, i_dim) --> (B, 1, i_dim)
    # item_emb: (B, L, i_dim) --> (B, i_dim, L)
    return (action_emb.unsqueeze(1) @ item_emb.transpose(-1, -2)).squeeze(1)  # (B, L)
