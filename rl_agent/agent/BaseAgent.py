from rl_agent.utils import LinearScheduler
from time import time
from tqdm import tqdm
import numpy as np
import wandb

import torch
class BaseAgent:
    def __init__(self, facade, params):
        self.device = params['device']
        self.gamma = params['gamma']
        self.n_iter = [0] + params['n_iter']
        self.train_every_n_step = params['train_every_n_step']
        self.check_episode = params['check_episode']
        self.save_path = params['save_path']
        self.facade = facade
        self.check_episode = params['check_episode']
        self.exploration_scheduler = LinearScheduler(int(sum(self.n_iter) * params['elbow_greedy']),
                                                     params['final_greedy_epsilon'],
                                                     params['initial_greedy_epsilon'])
        self.episode_batch_size = params['episode_batch_size']

    def train(self):
        if len(self.n_iter) > 2:
            self.load()

        t = time()
        start_time = t
        print("Run procedure before training")
        self.action_before_train()

        print("Start training")
        observation = self.facade.reset_env({
            'batch_size': self.episode_batch_size,
        })
        step_offset = sum(self.n_iter[:-1])
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            observation = self.run_episode_step(i, self.exploration_scheduler.value(i),
                                                observation, True)
            if i % self.train_every_n_step == 0:
                self.step_train()

            if i % self.check_episode == 0:
                t_ = time()
                # print(f"Episode step {i}, time diff {t_ - t}, total time dif {t - start_time})")
                self.log_iteration(i)
                t = t_
                if i % (3 * self.check_episode) == 0:
                    self.save()

        self.action_after_train()

    def action_before_train(self):
        pass

    def action_after_train(self):
        self.facade.stop_env()

    def get_report(self):
        episode_report = self.facade.get_episode_report(10)
        train_report = {k: np.mean(v[-10:]) for k, v in self.training_history.items()}
        return episode_report, train_report

    def log_iteration(self, step):
        episode_report, train_report = self.get_report()
        wandb.log(episode_report | train_report)
        log_str = f"step: {step} @ episode report: {episode_report} @ step loss: {train_report}\n"
        with open(self.save_path + ".report", 'a') as outfile:
            outfile.write(log_str)
        return log_str

    def test(self):
        self.load()
        self.facade.initialize_train()

        t = time()
        start_time = t

        print("Start testing")
        observation = self.facade.reset_env({
            'batch_size': self.episode_batch_size,
        })
        step_offset = sum(self.n_iter[:-1])
        with torch.no_grad():
            for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
                observation = self.run_episode_step(i, self.exploration_scheduler.value(i),
                                                    observation, True)
                if i % self.check_episode == 0:
                    t_ = time()
                    episode_report = self.facade.get_episode_report(10)
                    log_str = f"step: {i} @ episode report: {episode_report}\n"
                    # wandb.log(episode_report)
                    with open(self.save_path + "_eval.report", 'a') as outfile:
                        outfile.write(log_str)
                    # print(f"Episode step {i}, time diff {t_ - t}, total time dif {t - start_time})")
                    # print(log_str)
                    t = t_

    #######################################
    #           Abstract function         #
    #######################################
    def run_episode_step(self, *episode_args):
        pass

    def step_train(self):
        pass

    def save(self):
        pass

    def load(self):
        pass