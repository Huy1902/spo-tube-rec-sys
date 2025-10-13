import torch.nn as nn

class DNN(nn.Module):
  def __init__(self, in_dim, hidden_dims, out_dim=1, dropout_rate= 0.,
               do_batch_norm=True):
    super(DNN, self).__init__()
    self.in_dim = in_dim
    layers = []

    for hidden_dim in hidden_dims:
      linear_layer = nn.Linear(in_dim, hidden_dim)

      layers.append(linear_layer)
      in_dim = hidden_dim
      layers.append(nn.ReLU())
      if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
      if do_batch_norm:
        layers.append(nn.LayerNorm(hidden_dim))

    # Prediction layer
    last_layer = nn.Linear(in_dim, out_dim)
    layers.append(last_layer)

    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    x = x.view(-1, self.in_dim)
    logit = self.layers(x)
    return logit
