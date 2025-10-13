import copy

import torch
import torch.nn.functional as F

from rl_agent.agent.BaseAgent import BaseAgent
from rl_agent.utils import wrap_batch


class DDPG(BaseAgent):
  def __init__(self, facade, params):
    super().__init__(facade, params)
    self.actor = facade.actor
    self.actor_target = copy.deepcopy(self.actor)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = params['actor_lr'], weight_decay = params['actor_decay'])

    self.critic = facade.critic
    self.critic_target = copy.deepcopy(self.critic)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = params['critic_lr'], weight_decay = params['critic_decay'])

    self.episode_batch_size = params['episode_batch_size']
    self.tau = params['target_mitigate_coef']
    self.actor_lr = params['actor_lr']
    self.critic_lr = params['critic_lr']
    self.actor_decay = params['actor_decay']
    self.critic_decay = params['critic_decay']

    self.batch_size = params['batch_size']

    with open(self.save_path + ".report", 'w') as outfile:
      pass

  def action_before_train(self):
    '''
    - facade setup
      - buffer setup
    - run random episodes to build-up the initial buffer
    '''
    self.facade.initialize_train()
    # print("Facade Parameters:")
    # for param, value in vars(self.facade).items():
    #     print(f"{param}: {value}")
    prepare_step = 0
    # random explore before training
    initial_epsilon = 1.0
    observation = self.facade.reset_env({
        'batch_size': self.episode_batch_size,
    })
    while not self.facade.is_training_available:
      observation = self.run_episode_step(0, initial_epsilon, observation, True)
      # print(observation)
      prepare_step += 1

    # training records
    self.training_history = {"critic_loss": [], "actor_loss": []}

    print(f"Total {prepare_step} prepare steps")

  def run_episode_step(self, *episode_args):
    '''
    One step of interaction
    '''
    episode_iter, epsilon, observation, do_buffer_update = episode_args
    with torch.no_grad():
      # sample action
      policy_output = self.facade.apply_policy(observation, self.actor, epsilon,
                                               do_explore=True)

      # apply action on environment and update replay buffer
      next_observation, reward, done, info = self.facade.env_step(policy_output)

      # update replay buffer
      if do_buffer_update:
        self.facade.update_buffer(observation, policy_output, reward, done,
                                  next_observation, info)
    return next_observation

  def step_train(self):
    observation , policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.params['batch_size'])

    critic_loss, actor_loss = self.get_ddpg_loss(observation, policy_output, reward,
                                                  done_mask, next_observation)
    self.training_history["critic_loss"].append(critic_loss.item())
    self.training_history["actor_loss"].append(actor_loss.item())

    # Update the frozen target models
    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
      target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    return {'step_loss': (self.training_history['actor_loss'][-1],
                          self.training_history['critic_loss'][-1])}

  def get_ddpg_loss(self, observation, policy_output, reward, done_mask, next_observation,
                    do_actor_update = True, do_critic_update = True):
    # Get current Q estimate
    current_critic_output = self.facade.apply_critic(observation,
                                                     wrap_batch(policy_output, device=self.device),
                                                     self.critic)
    current_Q = current_critic_output['q']

    # Compute the target Q value
    next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)
    target_critic_output = self.facade.apply_critic(next_observation, next_policy_output,
                                                    self.critic_target)

    target_Q = target_critic_output['q']
    target_Q = reward + self.gamma * (done_mask * target_Q).detach()

    # compute critic loss
    # minimize current_Q predict and target_Q predict
    critic_loss = F.mse_loss(current_Q, target_Q).mean()

    if do_critic_update and self.critic_lr > 0:
      # Optimize the critic
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

    # compute actor loss
    policy_output = self.facade.apply_policy(observation, self.actor)
    critic_output = self.facade.apply_critic(observation, policy_output, self.critic)

    # Maximize Q value
    actor_loss = -critic_output['q'].mean()

    if do_actor_update and self.actor_lr > 0:
      # Optimize the actor
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()
    return critic_loss, actor_loss

  def save(self):
    torch.save(self.critic.state_dict(), self.save_path + "_critic")
    torch.save(self.critic_optimizer.state_dict(), self.save_path + "_critic_optimizer")
    torch.save(self.actor.state_dict(), self.save_path + "_actor")
    torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")

  def load(self):
    self.critic.load_state_dict(torch.load(self.save_path + "_critic", map_location=self.device))
    self.critic_optimizer.load_state_dict(torch.load(self.save_path + "_critic_optimizer", map_location=self.device))
    self.critic_target = copy.deepcopy(self.critic)

    self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
    self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
    self.actor_target = copy.deepcopy(self.actor)