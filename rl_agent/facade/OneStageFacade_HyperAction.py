import numpy as np
import torch

from rl_agent.utils import wrap_batch, get_regularization, sample_categorical_action
from rl_agent.facade.OneStageFacade import OneStageFacade

class OneStageFacade_HyperAction(OneStageFacade):
  def __init__(self, environment, actor, critic, params):
    super().__init__(environment, actor, critic, params)

  def apply_policy(self, observation, policy_model, epsilon = 0,
                   do_explore = False, do_softmax = True):
    feed_dict = wrap_batch(observation, device=self.device)
    # print(feed_dict.device)
    out_dict = policy_model(feed_dict)
    if do_explore:
      action_emb = out_dict['action_emb']
      # explore and exploit + clamping
      if np.random.rand() < epsilon:
        action_emb = torch.clamp(torch.rand_like(action_emb) * self.noise_var, -1, 1)
      else:
        action_emb = action_emb + torch.clamp(torch.rand_like(action_emb) * self.noise_var, -1, 1)

      out_dict['action_emb'] = action_emb

    # Z latent space
    out_dict['Z'] = out_dict['action_emb']

    if 'candidate_ids' in feed_dict:
      # (B, L, item_dim)
      out_dict['candidate_features']  = feed_dict['candidate_features']
      # (B, L)
      out_dict['candidate_ids'] = feed_dict['candidate_ids']
      batch_wise = True
    else:
      # (1, L, item_dim)
      out_dict['candidate_features'] = self.candidate_features.unsqueeze(0)
      #(L, )
      out_dict['candidate_ids'] = self.candidate_iids
      batch_wise = False

    # action pron (B, L)
    action_prob = policy_model.score(out_dict['action_emb'],
                                      out_dict['candidate_features'],
                                      do_softmax=do_softmax)

    # two types of greedy selection
    if np.random.rand() >= self.topk_rate:
      # greedy random
      action, indices = sample_categorical_action(action_prob, out_dict['candidate_ids'],
                                                  self.slate_size, with_replacement=False,
                                                  batch_wise=batch_wise,
                                                  return_idx=True)
    else:
      # indices on action_prob
      _, indices = torch.topk(action_prob, k = self.slate_size, dim = 1)
      # print(indices.shape)
      # print(self.candidate_features.shape)
      # top k action:
      # (B, slate_size)
      if batch_wise:
        action = torch.gather(out_dict['candidate_ids'], 1, indices).detach()
      else:
        action = out_dict['candidate_ids'][indices].detach()

    # (B, K)
    out_dict['action'] = action
    # (B, K, item_dim)
    out_dict['action_features'] = self.candidate_features[indices]
    # (B, K)
    out_dict['action_prob'] = torch.gather(action_prob, 1, indices)
    # (B, L)
    out_dict['candidate_prob'] = action_prob

    return out_dict

  def infer_hyper_action(self, observation, policy_output, actor):
    '''
    Inverse function A -> Z
    '''
    # (B, K)
    A = policy_output['action']

    # (B, K, item_dim)
    item_embs = self.candidate_features[A - 1]

    # (B, K, kernel_dim)
    Z = torch.mean(actor.item_map(item_embs).view(A.shape[0], A.shape[1], -1), dim = 1)
    return {
        'Z': Z,
        'action_emb': Z,
        'state_emb': policy_output['state_emb']
    }

  def apply_critic(self, observation, policy_output, critic_model):
    feed_dict = {
        'state_emb': policy_output['state_emb'],
        'action_emb': policy_output['action_emb']
    }
    critic_output = critic_model(feed_dict)
    return critic_output