import torch.nn as nn

from rl_agent.model.DNN import DNN
from rl_agent.utils import get_regularization

class QValueNetwork(nn.Module):
  def __init__(self, policy, params):
    super().__init__()
    self.state_dim = policy.state_dim
    self.net = DNN(self.state_dim, params['critic_hidden_dims'], params['n_item'],
                   dropout_rate=params['critic_dropout_rate'], do_batch_norm=True)

  def forward(self, feed_dict):
    state_emb = feed_dict['state_emb']
    Q = self.net(state_emb)

    reg = get_regularization(self.net)
    return {'q': Q, 'reg': reg}