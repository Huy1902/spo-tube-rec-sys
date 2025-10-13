import torch.nn as nn
import torch

from rl_agent.utils import get_regularization, dot_scorer


class SASRec(nn.Module):
  def __init__(self, environment, params):
    super().__init__()
    self.n_layer = params['sasrec_n_layer']
    self.d_model = params['sasrec_d_model']
    self.n_head = params['sasrec_n_head']
    self.dropout_rate = params['sasrec_dropout']
    self.d_forward = params['sasrec_d_forward']

    # item space
    self.item_space = environment.action_space['item_id'][1]
    self.item_dim = environment.action_space['item_feature'][1]
    self.maxlen = environment.observation_space['history'][1]
    self.state_dim = self.d_model
    self.action_dim = self.d_model

    # policy network modules
    self.item_map = nn.Linear(self.item_dim, self.d_model)
    self.pos_emb = nn.Embedding(self.maxlen, self.d_model)
    self.pos_emb_getter = torch.arange(self.maxlen, dtype = torch.long)
    self.emb_dropout = nn.Dropout(self.dropout_rate)
    self.emb_norm = nn.LayerNorm(self.d_model)
    self.attn_mask = ~torch.tril(torch.ones((self.maxlen, self.maxlen), dtype=torch.bool))
    encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                               nhead=self.n_head,
                                               dim_feedforward= self.d_forward,
                                               dropout=self.dropout_rate,
                                               batch_first = True
                                               )
    self.transformer = nn.TransformerEncoder(encoder_layer= encoder_layer,
                                             num_layers = self.n_layer)

  def score(self, action_emb, item_emb, do_softmax=True):
    '''

    :param action_emb:
    :param item_emb:
    :param do_softmax:
    :return:
    probability of action
    '''
    item_emb = self.item_map(item_emb)
    output = dot_scorer(action_emb, item_emb, self.d_model)
    if do_softmax:
      return torch.softmax(output, dim=-1)
    else:
      return output

  def get_scorer_parameters(self):
    return self.item_map.parameters()

  def encode_state(self, feed_dict):
    user_history = feed_dict['history_features']
    # (1, H, d_model)
    # for item in feed_dict.items():
    #   print(item)
    # print("user_history device:", user_history.device)
    # print("self.pos_emb_getter device:", self.pos_emb_getter.device)
    # print("self.pos_emb device", self.pos_emb.device)

    pos_emb = self.pos_emb(self.pos_emb_getter.to(user_history.device)).view(1, self.maxlen, self.d_model)

    # (B, H, d_model)
    history_item_emb = self.item_map(user_history).view(-1, self.maxlen, self.d_model)
    history_item_emb = self.emb_norm(self.emb_dropout(history_item_emb + pos_emb))

    # (B, H, d_model)
    output_seq = self.transformer(history_item_emb, mask = self.attn_mask.to(user_history.device))

    return {'output_seq': output_seq, 'state_emb': output_seq[:, -1, :]}

  def forward(self, feed_dict):
    '''
    @input
    - feed_dict: {'user_profile': (B, user_dim),
                  'history_features': (B, H, item_dim),
                  'history_mask': (B),
                  'candicate_features': (B, L, item_dim) or (1, L, item_dim)
                  }
    @model
    - user_profile --> user_emb (B, 1, f_dim)
    - hisotry_items --> history_item_emb (B, H, f_dim)
    - (Q:user_emb, K&V: history_item_emb) --(multi-head attn) --> user_state(B, 1, feature_dim)
    - user_state --> action_prob (B, n_item)
    @note
    - Remove user_profile in this setting
    '''

    hist_enc = self.encode_state(feed_dict)

    # user embedding (B, 1, d_model)
    user_state = hist_enc['state_emb'].view(-1, self.d_model)

    # action embedding (B, d_model)
    action_emb = user_state

    # regularization
    reg = get_regularization(self.item_map, self.transformer)

    out_dict = {
        'action_emb': action_emb,
        'state_emb': user_state,
        'seq_emb': hist_enc['output_seq'],
        'reg': reg
    }
    return out_dict