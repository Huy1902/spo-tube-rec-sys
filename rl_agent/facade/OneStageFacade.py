import torch
import numpy as np



class OneStageFacade:
    def __init__(self, environment, actor, critic, params):
        super().__init__()
        self.device = params['device']
        self.env = environment
        self.actor = actor
        self.critic = critic

        self.slate_size = params['slate_size']
        self.noise_var = params['noise_var']
        self.noise_decay = params['noise_var'] / params['n_iter'][-1]
        self.q_laplace_smoothness = params['q_laplace_smoothness']
        self.topk_rate = params['topk_rate']
        self.empty_start_rate = params['empty_start_rate']

        self.n_item = self.env.action_space['item_id'][1]

        # (N)
        self.candidate_iids = np.arange(0, self.n_item)

        # (N, item_dim)
        self.candidate_features = torch.FloatTensor(
            self.env.reader.get_item_list_meta(self.candidate_iids)).to(self.device)
        self.candidate_iids = torch.tensor(self.candidate_iids).to(self.device)

        # replay buffer is initialized in initialize_train()
        self.buffer_size = params['buffer_size']
        self.start_timestamp = params['start_timestamp']

    def initialize_train(self):
        '''
        Procedures before training
        '''
        self.buffer = {
            # "user_profile": torch.zeros(self.buffer_size, self.env.reader.portrait_len),
            "history": torch.zeros(self.buffer_size, self.env.reader.max_seq_len).to(torch.long),
            "next_history": torch.zeros(self.buffer_size, self.env.reader.max_seq_len).to(torch.long),
            "state_emb": torch.zeros(self.buffer_size, self.actor.state_dim),
            "action_emb": torch.zeros(self.buffer_size, self.actor.action_dim),
            "action": torch.zeros(self.buffer_size, self.slate_size, dtype=torch.long),
            "reward": torch.zeros(self.buffer_size),
            "min_reward": torch.zeros(self.buffer_size),
            "feedback": torch.zeros(self.buffer_size, self.slate_size),
            "done": torch.zeros(self.buffer_size, dtype=torch.bool)
        }

        for k, v in self.buffer.items():
            self.buffer[k] = v.to(self.device)
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0
        self.is_training_available = False

    def reset_env(self, initial_params={'batch_size': 1}):
        '''
        Reset user response environment
        '''
        initial_params['empty_history'] = True if np.random.rand() < self.empty_start_rate else False
        initial_observation = self.env.reset(initial_params)
        return initial_observation

    def env_step(self, policy_output):
        action_dict = {
            'action': policy_output['action'],
            'action_features': policy_output['action_features']
        }
        observation, reward, done, info = self.env.step(action_dict)
        return observation, reward, done, info

    def stop_env(self):
        self.env.stop()

    def get_episode_report(self, n_recent=10):
        recent_rewards = self.env.reward_history[-n_recent:]
        recent_steps = self.env.step_history[-n_recent:]
        epsiode_report = {
            'average_total_reward': np.mean(recent_rewards),
            'reward_variance': np.var(recent_rewards),
            'max_total_reward': np.max(recent_rewards),
            'min_total_reward': np.min(recent_rewards),
            'average_n_step': np.mean(recent_steps),
            'max_n_step': np.max(recent_steps),
            'min_n_step': np.min(recent_steps),
            'buffer_size': self.current_buffer_size
        }
        return epsiode_report

    def apply_critic(self, observation, policy_output, critic_model):
        feed_dict = {
            'state_emb': policy_output['state_emb'],
            'action_emb': policy_output['action_emb']
        }
        critic_output = critic_model(feed_dict)
        return critic_output

    def apply_policy(self, observation, policy_model, epsilon=0,
                     do_explore=False, do_softmax=True):
        '''
        @input:
        - observation: input of policy model
        - policy_model
        - epsilon: greedy epsilon, effective only when do_explore == True
        - do_explore: exploration flag, True if adding noise to action
        - do_softmax: output softmax score
        '''
        emb = policy_model(observation)
        action_prob = policy_model.score(emb['action_emb'], self.candidate_features, do_softmax=True)
        indices = torch.multinomial(action_prob, self.slate_size, replacement=False)
        action = self.candidate_iids[indices].detach()
        z = action_prob == 0
        z = z.float() * 1e-8
        out_dict = dict()
        out_dict['action_features'] = self.candidate_features[indices]
        out_dict['action_prob' ] = action_prob + z # stabilize
        out_dict['action'] = action
        out_dict['state_emb'] = emb['state_emb']
        out_dict['action_emb'] = emb['action_emb']
        return out_dict

    def sample_buffer(self, batch_size):
        '''
        @output:
        - observation
        - policy output
        - reward
        - done_mask
        - next_observation
        '''
        indices = np.random.randint(0, self.current_buffer_size, size=batch_size)
        H, N, S, HA, A, R, F, D, MR = self.read_buffer(indices)
        observation = {
            # 'user_profile': U,
            'history_features': H,
            'min_reward': MR
        }
        policy_output = {
            'state_emb': S,
            'action_emb': HA,
            'action': A
        }
        reward = R
        done_mask = D
        next_observation = {
            # 'user_profile': U,
            'history_features': N,
            'min_reward': MR,
            'previous_feedback': F
        }
        return observation, policy_output, reward, done_mask, next_observation

    # def sample_raw_data(self, batch_size):
    #   '''
    #   Sample supervise data from raw training data
    #   '''
    #   batch = self.env.sample_user(batch_size)

    def update_buffer(self, observation, policy_output, reward, done_mask,
                      next_observation, info):
        # Overwrite old entries in buffer
        if self.buffer_head + reward.shape[0] >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + \
                      [i for i in range(reward.shape[0] - tail)]
        else:
            indices = [self.buffer_head + i for i in range(reward.shape[0])]

        # update buffer
        # self.buffer["user_profile"][indices] = observation['user_profile']
        self.buffer["history"][indices] = observation['history']
        self.buffer["min_reward"][indices] = observation['min_reward']
        self.buffer["next_history"][indices] = next_observation['history']
        self.buffer["state_emb"][indices] = policy_output['state_emb']
        self.buffer["action"][indices] = policy_output['action']
        self.buffer["action_emb"][indices] = policy_output['action_emb']
        self.buffer["reward"][indices] = reward
        self.buffer["feedback"][indices] = info['response']
        self.buffer["done"][indices] = done_mask

        # update buffer pointer
        self.buffer_head = (self.buffer_head + reward.shape[0]) % self.buffer_size
        self.n_stream_record += reward.shape[0]
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)

        # available training when sufficient sample buffer
        if self.n_stream_record >= self.start_timestamp:
            self.is_training_available = True

    def read_buffer(self, indices):
        # U = self.buffer['user_profile'][indices]
        # (L, item_dim)
        H = self.candidate_features[self.buffer["history"][indices] - 1]
        N = self.candidate_features[self.buffer["next_history"][indices] - 1]
        S = self.buffer["state_emb"][indices]
        HA = self.buffer["action_emb"][indices]
        A = self.buffer["action"][indices]
        R = self.buffer["reward"][indices]
        F = self.buffer["feedback"][indices]
        D = self.buffer["done"][indices]
        MR = self.buffer['min_reward'][indices]
        return H, N, S, HA, A, R, F, D, MR

    def extract_behavior_data(self, observation, policy_output, next_observation):
        '''
        Extract supervised data from RL samples
        '''
        observation = {
            # "user_profile": observation['user_profile'],
            "history_features": observation['history_features']
        }
        exposed_items = policy_output['action']
        exposure = {
            "ids": exposed_items,
            "features": self.candidate_features[exposed_items - 1]
        }
        user_feedback = next_observation["previous_feedback"]
        return observation, exposure, user_feedback