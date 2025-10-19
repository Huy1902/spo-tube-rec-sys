import os
import wandb
import torch
import numpy as np
import pandas as pd

from rl_agent.environment.RewardFunction import mean_with_cost
from rl_agent.environment.MDPEnvironment import MDPEnvironment
from rl_agent.model.SASRec import SASRec
from rl_agent.model.GeneralCritic import GeneralCritic
from rl_agent.facade.OneStageFacade_HyperAction import OneStageFacade_HyperAction
from rl_agent.agent.HAC import HAC
from rl_agent.utils import set_random_seed

path_to_data = os.path.join(os.getcwd(), "dataset/mdp/")
path_to_output = os.path.join(os.getcwd(), "output/mdp/")
# print(path_to_data, path_to_output)
wandb.login(key =os.getenv("WANDB_LOGIN"))
cuda = 0
if cuda >= 0 and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    torch.cuda.set_device(cuda)
    device = f"cuda:{cuda}"
else:
    device = "cpu"
item_info = np.load(os.path.join(path_to_data, "item_info.npy"))
item_ids = np.load(os.path.join(path_to_data, "item_ids.npy"))
train = pd.read_csv(os.path.join(path_to_data, "train.csv"), sep="@")
test = pd.read_csv(os.path.join(path_to_data, "test.csv"), sep="@")

params = dict()

params['train'] = train
params['val'] = test
params['item_meta'] = item_info
# params['user_meta'] = user_info
params['item_ids'] = item_ids

params['n_worker'] = 4
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
params['n_iter'] = [50000]
params['slate_size'] = 10
params['noise_var'] = 0.1
params['q_laplace_smoothness'] = 0.5
params['topk_rate'] = 1
params['empty_start_rate'] = 0
params['buffer_size'] = 100000
params['start_timestamp'] = 2000
params['gamma'] = 0.9
params['train_every_n_step'] = 1
params['update_target_every_n_step'] = 100
params['initial_greedy_epsilon'] = 0
params['final_greedy_epsilon'] = 0
params['elbow_greedy'] = 0.1
params['check_episode'] = 10
params['with_eval'] = False

params['episode_batch_size'] = 32
params['batch_size'] = 64
params['actor_lr'] = 0.00001
params['critic_lr'] = 0.001
params['actor_decay'] = 0.00001
params['critic_decay'] = 0.00001
params['target_mitigate_coef'] = 0.01
params['behavior_lr'] = 0.00005
params['behavior_decay'] = 0.00001
params['hyper_actor_coef'] = 0.1

config = params.copy()
config.pop("train", None)
config.pop("val", None)
config.pop("item_meta", None)
config.pop("item_ids", None)
# config.pop("user_meta", None)

for seed in [7, 11, 19]:
    params['seed'] = seed
    set_random_seed(params['seed'])
    params['save_path'] = os.path.join(path_to_output, f"agent/mdp_model_seed{params['seed']}")
    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="23020082-uet",
        # Set the wandb project where this run will be logged.
        project="SpoTubeRecSys",
        # Track hyperparameters and run metadata.
        config=config
    )
    os.makedirs(os.path.dirname(params['save_path']), exist_ok=True)

    env = MDPEnvironment(params)

    policy = SASRec(env, params)
    policy.to(device)

    critic = GeneralCritic(policy, params)
    critic.to(device)

    facade = OneStageFacade_HyperAction(env, policy, critic, params)

    agent = HAC(facade, params)

    agent.train()
    wandb.finish()