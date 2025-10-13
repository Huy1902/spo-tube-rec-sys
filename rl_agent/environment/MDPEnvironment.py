import torch
from torch.utils.data import DataLoader
import numpy as np
import copy

from rl_agent.utils import wrap_batch, padding_and_clip
from rl_agent.environment.BaseEnv import BaseEnv
from rl_agent.reader.MDPDataReader import MDPDataReader
from rl_agent.model.MDPUserRespond import MDPUserResponse


class MDPEnvironment(BaseEnv):
  def __init__(self, params):
    super().__init__(params)
    self.reader = MDPDataReader(params)
    self.device = params['device']
    self.n_worker = params['n_worker']
    self.user_response_model = MDPUserResponse(self.reader, params)
    checkpoint = torch.load(params['model_path'] + ".checkpoint", map_location=self.device)
    self.user_response_model.load_state_dict(checkpoint["model_state_dict"])
    self.user_response_model.to(self.device)
    self.n_worker = params['n_worker']

    # spaces
    stats = self.reader.get_statistics()
    self.action_space = {'item_id': ('nomial', stats['n_item']),
                         'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
    self.observation_space = {'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}

  def reset(self, params = {'batch_size': 1, 'empty_history': True}):
      self.empty_history_flag = params['empty_history'] if 'empty_history' in params else True
      BS = params['batch_size']
      if 'sample' in params:
          sample_info = params['sample']
      else:
          self.batch_iter = iter(DataLoader(self.reader, batch_size = BS, shuffle = True,
                                            pin_memory = True, num_workers = self.n_worker))
          sample_info = next(self.batch_iter)
          sample_info = wrap_batch(sample_info, device = self.user_response_model.device)
      self.current_observation = {
          # 'user_profile': sample_info['user_profile'],  # (B, user_dim)
          'history': sample_info['history'],  # (B, H)
          'history_features': sample_info['history_features'], # (B, H, item_dim)
          'cumulative_reward': torch.zeros(BS).to(self.user_response_model.device),
          'min_reward': torch.full((BS,), float('inf'), device=self.user_response_model.device),
          'temper': torch.ones(BS).to(self.user_response_model.device) * self.initial_temper,
          'step': torch.zeros(BS).to(self.user_response_model.device),
      }
      self.reward_history = [0.]
      self.step_history = [0.]
      return copy.deepcopy(self.current_observation)


  def sample_user(self, n_user, empty_history = False):
    '''
    Sample random users and their history
    '''
    random_rows = np.random.randint(0, len(self.reader.data['train']), n_user)
    return self.pick_user(random_rows, empty_history)

  def pick_user(self, rows, empty_history = False):
    '''
    Pick users and their history
    '''
    raw_portrait = [self.reader.user_meta[self.reader.data['train']['user_id'][rowid]]
                    for rowid in rows]

    portrait = np.array(raw_portrait)

    history = []
    history_features = []
    for rowid in rows:
      H = [] if empty_history else eval(f"{self.reader.data['train']['user_mid_history'][rowid]}")
      H = padding_and_clip(H, self.reader.max_seq_len)
      history.append(H)

      history_features.append(self.reader.get_item_list_meta(H).astype(float))
    return {'user_profile': portrait,
          'history': history,
          'history_features': np.array(history_features)}

  def step(self, step_dict):
    '''
    @input:
    - step_dict: {'action': (B, slate_size),
                    'action_features': (B, slate_size, item_dim) }
    '''
    # actions (exposures)
    action = step_dict['action'] # (B, slate_size), should be item ids only
    action_features = step_dict['action_features']
    batch_data = {
        # 'user_profile': self.current_observation['user_profile'],
        'history_features': self.current_observation['history_features'],
        'exposure_features': action_features
    }
    # URM forward
    with torch.no_grad():
        output_dict = self.user_response_model(batch_data)
        # response = torch.bernoulli(output_dict['probs']) # (B, slate_size)
        probs_under_temper = output_dict['probs'] # * prob_scale
        response = torch.bernoulli(probs_under_temper).detach() # (B, slate_size)

        # reward (B,)
        immediate_reward = self.reward_func(response).detach()

        self.current_observation['min_reward'] = torch.min(immediate_reward, self.current_observation['min_reward'])

        # (B, H+slate_size)
        H_prime = torch.cat((self.current_observation['history'], action), dim = 1)
        # (B, H+slate_size, item_dim)
        H_prime_features = torch.cat((self.current_observation['history_features'], action_features), dim = 1)
        # (B, H+slate_size)
        F_prime = torch.cat((torch.ones_like(self.current_observation['history']), response), dim = 1).to(torch.long)
        # vector, vector
        row_indices, col_indices = (F_prime == 1).nonzero(as_tuple=True)
        # (B,), the number of positive iteraction as history length
        L = F_prime.sum(dim = 1)

        # user history update
        offset = 0
        newH = torch.zeros_like(self.current_observation['history'])
        newH_features = torch.zeros_like(self.current_observation['history_features'])
        for row_id in range(action.shape[0]):
            right = offset + L[row_id]
            left = right - self.reader.max_seq_len
            newH[row_id] = H_prime[row_id, col_indices[left:right]]
            newH_features[row_id] = H_prime_features[row_id,col_indices[left:right],:]
            offset += L[row_id]
        self.current_observation['history'] = newH
        self.current_observation['history_features'] = newH_features
        self.current_observation['cumulative_reward'] += immediate_reward

        # temper update for leave model
        temper_down = (-immediate_reward+1) * response.shape[1] + 1
#             temper_down = -(torch.sum(response, dim = 1) - response.shape[1] - 1)
#             temper_down = torch.abs(torch.sum(response, dim = 1) - response.shape[1] * self.temper_sweet_point) + 1
        self.current_observation['temper'] -= temper_down
        # leave signal
        done_mask = self.current_observation['temper'] < 1
        # step update
        self.current_observation['step'] += 1

        # update rows where user left
#             refresh_rows = done_mask.nonzero().view(-1)
#             print(f"#refresh: {refresh_rows}")
        if done_mask.sum() > 0:
            final_rewards = self.current_observation['cumulative_reward'][done_mask].detach().cpu().numpy()
            final_steps = self.current_observation['step'][done_mask].detach().cpu().numpy()
            self.reward_history.append(final_rewards[-1])
            self.step_history.append(final_steps[-1])
            # sample new users to fill in the blank
            new_sample_flag = False
            try:
                sample_info = next(self.iter)
                if sample_info['history'].shape[0] != done_mask.shape[0]:
                    new_sample_flag = True
            except:
                new_sample_flag = True
            if new_sample_flag:
                self.iter = iter(DataLoader(self.reader, batch_size = done_mask.shape[0], shuffle = True,
                                            pin_memory = True, num_workers = self.n_worker))
                sample_info = next(self.iter)
            sample_info = wrap_batch(sample_info, device = self.user_response_model.device)
            for obs_key in ['history', 'history_features']:
                self.current_observation[obs_key][done_mask] = sample_info[obs_key][done_mask]
            self.current_observation['cumulative_reward'][done_mask] *= 0
            self.current_observation['min_reward'][done_mask] = float('inf')
            self.current_observation['temper'][done_mask] *= 0
            self.current_observation['temper'][done_mask] += self.initial_temper
        self.current_observation['step'][done_mask] *= 0
#         print(f"step: {self.current_observation['step']}")
    return copy.deepcopy(self.current_observation), immediate_reward, done_mask, {'response': response}


  def stop(self):
    self.iter = None

  def get_new_iterator(self, B):
    return iter(DataLoader(self.reader, batch_size = B, shuffle = True,
                              pin_memory = True, num_workers = self.n_worker))
