import torch.nn as nn
import torch

from rl_agent.model.DNN import DNN
from rl_agent.utils import get_regularization

class GeneralCritic(nn.Module):
  def __init__(self, policy, params):
    super().__init__()
    self.state_dim = policy.state_dim
    self.action_dim = policy.action_dim
    self.net = DNN(self.state_dim + self.action_dim, params['critic_hidden_dims'], 1,
                   dropout_rate=params['critic_dropout_rate'], do_batch_norm=True)

  def forward(self, feed_dict):
    '''
    @input:
    - feed_dict: {'state_emb': (B, state_dim), 'action_emb': (B, action_dim)}
    '''
    state_emb = feed_dict['state_emb']
    action_emb = feed_dict['action_emb'].view(-1, self.action_dim)

    Q = self.net(torch.cat((state_emb, action_emb), dim = -1)).view(-1)

    reg = get_regularization(self.net)
    return {'q': Q, 'reg': reg}