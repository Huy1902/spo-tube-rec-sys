import torch

def mean_with_cost(feedback, zero_reward_cost=0.1):
  B, L = feedback.shape
  cost = torch.zeros_like(feedback)
  cost[feedback == 0] = -zero_reward_cost
  reward = torch.mean(feedback + cost, dim=-1)
  return reward


def nsw(avg_r, min_r, lambda_nsw=1e-4, epsilon=1e-8):
    r_vec = torch.stack([avg_r, min_r + lambda_nsw], dim=-1)
    r_vec = torch.clamp(r_vec, min=epsilon)
    return torch.sum(torch.log(r_vec), dim=-1)