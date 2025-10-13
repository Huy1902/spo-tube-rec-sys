import copy
import torch
import numpy as np
import torch.nn.functional as F

from rl_agent.agent.BaseAgent import BaseAgent

class SAC(BaseAgent):
    def __init__(self, facade, params):
        super().__init__(facade, params)
        self.actor = facade.actor
        self.critic1 = copy.deepcopy(facade.critic)
        self.critic2 = copy.deepcopy(facade.critic)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)


        self.entropy_target = 0.98 * (-np.log(1 / params['n_item']))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        self.update_target_freq = params['update_target_every_n_step']
        self.update_counter= 0

        self.tau = params['target_mitigate_coef']
        self.batch_size = params['batch_size']
        self.episode_batch_size = params['episode_batch_size']

        self.training_history = {"critic_loss": [], "actor_loss": [], 'alpha_loss': [], 'alpha': [], 'target_q': [], 'q1': [], 'q': []}

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params['actor_lr'],
                                                weight_decay=params['actor_decay'])
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=params['critic_lr'],
                                                  weight_decay=params['critic_decay'])
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=params['critic_lr'],
                                                  weight_decay=params['critic_decay'])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=params['lr'])

    def action_before_train(self):
        self.facade.initialize_train()
        initial_epsilon = 1.0
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        prepare_step = 0
        while not self.facade.is_training_available:
            observation = self.run_episode_step(0, initial_epsilon, observation, True)
            prepare_step += 1
        print(f"Total {prepare_step} prepare steps")

    def run_episode_step(self, *episode_args):
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            policy_output = self.facade.apply_policy(observation, self.actor, do_explore=True)
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)

        # Critic target
        with torch.no_grad():
            next_policy_output = self.facade.apply_policy(next_observation, self.actor)
            next_prob = next_policy_output['action_prob']
            next_log_prob = torch.log(next_prob)
            next_state_emb = self.actor(next_observation)['state_emb']
            feed_dict = dict()
            feed_dict['state_emb'] = next_state_emb # (B, state_emb_dim)
            # print(next_state_emb.shape)
            next_q1 = self.critic1_target(feed_dict)
            next_q2 = self.critic2_target(feed_dict)
            # print(next_q1['q'].shape)
            next_q = torch.min(next_q1['q'], next_q2['q'])
            # print(next_q.shape, (self.alpha * next_log_prob).shape)
            next_v = (next_prob * (next_q - self.alpha * next_log_prob)).sum(-1)
            target_q = reward + self.gamma * (~done_mask) * next_v
            current_state_emb = self.actor(observation)['state_emb']


        # Current Q estimates
        feed_dict = dict()
        feed_dict['state_emb'] = current_state_emb
        q1 = self.critic1(feed_dict)['q'].gather(-1, policy_output['action'] - 1).mean(-1)
        q2 = self.critic2(feed_dict)['q'].gather(-1, policy_output['action'] - 1).mean(-1)

        # print(q1.shape, q2.shape, target_q.shape)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        # Actor update
        probs = self.facade.apply_policy(observation, self.actor)['action_prob']
        log_probs = torch.log(probs)
        with torch.no_grad():
            q1 = self.critic1(feed_dict)['q']
            q2 = self.critic2(feed_dict)['q']
            q = torch.min(q1, q2)

        actor_loss = (probs * (self.alpha.detach() * log_probs - q)).sum(-1).mean()
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        log_probs = (probs * log_probs).sum(-1)
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.entropy_target)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        self.update_counter += 1
        # Target update
        if self.update_counter % self.update_target_freq == 0:
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

        self.training_history['critic_loss'].append(0.5 * (critic1_loss + critic2_loss).item())
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history.setdefault('alpha_loss', []).append(alpha_loss.item())
        self.training_history.setdefault('alpha', []).append(self.alpha.item())
        self.training_history['target_q'].append(target_q.mean().item())
        self.training_history['q1'].append(q1.mean().item())

        return {"step_loss": (0.5 * (critic1_loss + critic2_loss).item(), actor_loss.item(), alpha_loss.item())}

    def save(self):
        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.critic1.state_dict(), self.save_path + "_critic1")
        torch.save(self.critic2.state_dict(), self.save_path + "_critic2")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")
        torch.save(self.critic1_optimizer.state_dict(), self.save_path + "_critic1_optimizer")
        torch.save(self.critic2_optimizer.state_dict(), self.save_path + "_critic2_optimizer")

    def load(self):
        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.critic1.load_state_dict(torch.load(self.save_path + "_critic1", map_location=self.device))
        self.critic2.load_state_dict(torch.load(self.save_path + "_critic2", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.critic1_optimizer.load_state_dict(
            torch.load(self.save_path + "_critic1_optimizer", map_location=self.device))
        self.critic2_optimizer.load_state_dict(
            torch.load(self.save_path + "_critic2_optimizer", map_location=self.device))
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)