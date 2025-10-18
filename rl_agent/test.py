import os
import numpy as np
import torch
import pandas as pd

from model.QValueNetwork import QValueNetwork
from facade.OneStageFacade import OneStageFacade
from environment.RewardFunction import mean_with_cost
from environment.MDPEnvironment import MDPEnvironment
from model.SASRec import SASRec
from rl_agent.facade.OneStageFacade_HyperAction import OneStageFacade_HyperAction
from rl_agent.model.GeneralCritic import GeneralCritic
from utils import set_random_seed

path_to_data = "dataset/mdp/"
path_to_output = "output/mdp/"


cuda = 0
if cuda >= 0 and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    torch.cuda.set_device(cuda)
    device = f"cuda:{cuda}"
else:
    device = "cpu"

print(device)

item_info = np.load(os.path.join(path_to_data, "item_info.npy"))
item_ids = np.load(os.path.join(path_to_data, "item_ids.npy"))
train = pd.read_csv(os.path.join(path_to_data, "train.csv"), sep="@")
test = pd.read_csv(os.path.join(path_to_data, "test.csv"), sep="@")

params = dict()
params['train'] = train
params['val'] = test
params['item_meta'] = item_info
params['item_ids'] = item_ids
params['n_worker'] = 0
params['max_seq_len'] = 50

params['loss_type'] = 'bce'
params['device'] = device
params['l2_coef'] = 0.001
params['lr'] = 0.0003
params['feature_dim'] = 16
params['hidden_dims'] = [256]
params['attn_n_head'] = 2
params['batch_size'] = 128
params['seed'] = 26
params['epoch'] = 2
params['dropout_rate'] = 0.2
params['model_path'] = os.path.join(path_to_output,
                          f"env/mdp_user_env_lr{params['lr']}_reg{params['l2_coef']}.model")
set_random_seed(params['seed'])

params['loss_type'] = 'bce'
params['device'] = device
params['l2_coef'] = 0.001
params['lr'] = 0.0003
params['feature_dim'] = 16
params['hidden_dims'] = [256]
params['attn_n_head'] = 2
params['batch_size'] = 128
params['epoch'] = 2
params['dropout_rate'] = 0.2
params['max_step'] = 20
params['initial_temper'] = 20
params['reward_function'] = mean_with_cost
params['sasrec_n_layer'] = 2
params['sasrec_d_model'] = 32
params['sasrec_n_head'] = 4
params['sasrec_dropout'] = 0.1
params['sasrec_d_forward'] = 64
params['critic_hidden_dims'] = [256, 64]
params['critic_dropout_rate'] = 0.2
params['n_iter']= [50000]
params['slate_size'] = 10
params['noise_var'] = 0.1
params['q_laplace_smoothness'] = 0.5
params['topk_rate'] = 0
params['empty_start_rate'] = 0
params['buffer_size'] = 100000
params['start_timestamp'] = 2000
params['gamma'] = 0.9
params['train_every_n_step']= 1
params['initial_greedy_epsilon'] = 0
params['final_greedy_epsilon'] = 0
params['elbow_greedy'] = 0.1
params['check_episode'] = 10
params['with_eval'] = False

params['episode_batch_size'] = 32
params['batch_size'] = 64
params['actor_lr'] = 0.01
params['critic_lr'] = 0.01
params['actor_decay'] = 0.00001
params['critic_decay'] = 0.00001
params['target_mitigate_coef'] = 0.01
params['behavior_lr'] = 0.0003
params['behavior_decay'] = 0.00001
params['hyper_actor_coef'] = 0.1
params['advantage_bias'] = 0
params['entropy_coef'] = 0.0001

params['alpha'] = 0.02

params['n_item'] = 3952

env = MDPEnvironment(params)
policy = SASRec(env, params)
# user = env.sample_user(5)
# print(user['user_profile'].shape)
# print(len(user['history'][0]))
# print(len(user['history_features'][0][0]))


# (N)
candidate_iids = torch.tensor(np.arange(1, env.action_space['item_id'][1] + 1))

# (N, item_dim)
item_emb = torch.FloatTensor(env.reader.get_item_list_meta(candidate_iids)).to(device)

observation = env.reset()

## test environment + policy

# print(observation)

emb = policy(observation)
# print(emb.keys())

action_prob = policy.score(emb['action_emb'], item_emb, do_softmax=True)
# print(action_prob.shape)
_, indices = torch.topk(action_prob, k = 10, dim = 1)
# print(indices)
# print(candidate_iids.shape)

action = candidate_iids[indices].detach()

# print(action)
action_dict = {
    'action': action,
    'action_features': item_emb[indices]
}

next_observation, immediate_reward, done_mask, info = env.step(action_dict)
# while not done_mask[0].bool():
#     print(next_observation['cumulative_reward'])
#     emb = policy(next_observation)
#     action_prob = policy.score(emb['action_emb'], item_emb, do_softmax=True)
#     _, indices = torch.topk(action_prob, k=10, dim=1)
#     action = candidate_iids[indices].detach()
#     next_observation, immediate_reward, done_mask, info = env.step(action_dict)

critic = GeneralCritic(policy, params)
facade = OneStageFacade_HyperAction(env, policy, critic, params)

policy_output = facade.apply_policy(next_observation, policy)
# print((policy_output['action_prob'] == 1e-8).sum().item())
next_prob = policy_output['action_prob']
next_log_prob = torch.log(next_prob)


# critic1 = QValueNetwork(policy, params)
# critic2 = QValueNetwork(policy, params)
next_state_emb = policy(next_observation)['state_emb']
# print(next_observation.keys())
feed_dict = dict()
feed_dict['state_emb'] = next_state_emb
# next_q1 = critic1(feed_dict)
# next_q2 = critic2(feed_dict)
# next_q = torch.min(next_q1['q'], next_q2['q'])
# log_alpha = torch.zeros(1, requires_grad=True, device=device)
# alpha = log_alpha.exp()
# print(observation.keys())
# print(next_prob.shape)
# print(next_q.shape)
# print(next_log_prob.shape)
# next_v = (next_prob * (next_q - alpha * next_log_prob)).sum(-1).unsqueeze(-1)
# print(alpha * next_log_prob)
# print(next_q)
# print(next_q - alpha*next_log_prob)
#
# print(next_prob.shape)
# print(next_q.shape)
#
# next_policy_output = facade.apply_policy(next_observation, policy)
# next_prob = next_policy_output['action_prob']
# next_log_prob = torch.log(next_prob)
# next_state_emb = policy(next_observation)['state_emb']
# # print(next_observation.keys())
# feed_dict = dict()
# feed_dict['state_emb'] = next_state_emb
# next_q1 = critic1(feed_dict)
# next_q2 = critic2(feed_dict)
# next_q = torch.min(next_q1['q'], next_q2['q'])
# next_v = (next_prob * (next_q - alpha * next_log_prob)).sum(-1).unsqueeze(-1)
# print(immediate_reward, done_mask)
# print(next_v)
# target_q = immediate_reward + 0.9 * (~done_mask) * next_v
# print(target_q)