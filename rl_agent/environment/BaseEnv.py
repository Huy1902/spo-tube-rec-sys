class BaseEnv:
  def __init__(self, params):
    super().__init__()
    self.reward_func = params['reward_function']
    self.max_step_per_episode = params['max_step']
    self.initial_temper = params["initial_temper"]

  def reset(self, paras):
    pass
  def step(self, action):
    pass