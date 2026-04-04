import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gymnasium as gym
from dataclasses import dataclass
import minigrid
from minigrid.wrappers import FlatObsWrapper
from typing import Optional


class StickyActionWrapper(gym.Wrapper):
    """Forces the environment to repeat the chosen action for 4 frames."""
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

@dataclass
class Args:
    exp_name: str = "Robust RL"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    env_id: str = "MountainCar-v0"
    total_timesteps: int = 1000000
    learning_rate: float = 1e-4
    num_envs: int = 8
    num_steps: int = 512
    anneal_lr: bool = True
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    start_alpha: float = 0.05
    max_alpha: float = 0.5
    start_eta: float = 0.0
    lambda_threshold: float = -750.0
    nu_alpha: float = 0.001
    nu_eta: float = 0.005
    eval_episodes: int = 10



def make_env(env_id, idx, capture_video, run_name):
    """Generates the environment and applies standard wrappers."""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=1000)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, max_episode_steps=1000)
        env = StickyActionWrapper(env, repeat=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes neural network layers with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Standard CleanRL discrete action Agent."""
    def __init__(self, envs):
        super().__init__()
        self.reward_sign = 1.0 # Default to protagonist (+1)
        
        # Determine input size from the flattened observation space
        obs_shape = int(np.array(envs.single_observation_space.shape).prod())
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        x = x.reshape(x.shape[0], -1) 
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.reshape(x.shape[0], -1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def update_agent(agent, optimizer, obs, actions, logprobs, rewards, values, dones, next_obs, next_done, args, writer, global_step, agent_name="agent"):
    """
    Calculates advantages and runs the PPO update loop for a single agent.
    """
    device = obs.device
    
    # Bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    # Flatten the batch
    b_obs = obs.reshape((-1,) + obs.shape[2:]) 
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + actions.shape[2:])
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # Record specific agent metrics to Tensorboard and WandB
    writer.add_scalar(f"losses/{agent_name}_value_loss", v_loss.item(), global_step)
    writer.add_scalar(f"losses/{agent_name}_policy_loss", pg_loss.item(), global_step)
    writer.add_scalar(f"losses/{agent_name}_entropy", entropy_loss.item(), global_step)
    writer.add_scalar(f"losses/{agent_name}_explained_variance", explained_var, global_step)
    writer.add_scalar(f"losses/{agent_name}_approx_kl", approx_kl.item(), global_step)

    return b_returns