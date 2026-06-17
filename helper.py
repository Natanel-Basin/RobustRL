import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gymnasium as gym
from dataclasses import dataclass

@dataclass
class Args:
    # ==================================================
    # --- CleanRL general arguments ---
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 5
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RobustRL"
    """the wandb's project name"""
    wandb_entity: str = "models"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Walker2d-v5"
    """the id of the environment""" 
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.015
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    # ==================================================

    # Our args
    lambda_threshold: float = 2700
    """minimal performance required"""
    lr_alpha: float = 0
    """learning rate for alpha updates"""
    actor_adv_ratio: int = 10
    """two-timescale (Tessler et al. Alg. 5): N protagonist updates per 1 adversary update (article uses 10 for PR-MDP, 1 for NR-MDP)"""
    barrier_t: float = 0.2
    """barrier parameter t for log barrier method"""
    start_alpha: float = 0.05
    """start value of alpha"""
    max_alpha: float = 0.4
    """maximum value of alpha"""
    alpha_step_limit: float = 5e-2
    """maximum step size for alpha updates"""
    alpha_method: str = "controller"
    """how to update alpha: 'controller' (dual ascent on V-lambda, recommended) or 'barrier' (corrected log-barrier)"""
    v_ema_beta: float = 0.95
    """EMA smoothing for the raw episodic-return estimate V used in the alpha update"""
    beta: float = 0.05
    """probability of choosing a random action in the beta test"""
    eval_episodes: int = 20
    """number of episodes to evaluate the protagonist agent after training"""
    gpus: str = ""
    """comma-separated GPU ids for parallel per-seed evaluation in test.py (e.g. 0,1,2,3)"""


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

def make_env(env_id, idx, capture_video, run_name, gamma, test_mode=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=1000)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, max_episode_steps=1000)
            
        # Move RecordEpisodeStatistics here so it records the raw, unnormalized score!
        env = gym.wrappers.RecordEpisodeStatistics(env)
            
        if env_id == "MountainCar-v0":
            env = StickyActionWrapper(env, repeat=4)
        elif env_id == "Walker2d-v5":
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, 
                lambda obs: np.clip(obs, -10, 10), 
                observation_space=env.observation_space
            )
            if not test_mode:
                env = gym.wrappers.NormalizeReward(env, gamma=gamma)
                env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def get_obs_norm_stats(envs):
    rms_list = []
    for env in envs.envs:
        try:
            rms_list.append(env.get_wrapper_attr('obs_rms'))
        except AttributeError:
            continue
    rms_list = [r for r in rms_list if r is not None]
    if not rms_list:
        return None
    counts = np.array([float(r.count) for r in rms_list])
    means = np.stack([np.asarray(r.mean, dtype=np.float64) for r in rms_list])
    varss = np.stack([np.asarray(r.var, dtype=np.float64) for r in rms_list])
    total = counts.sum()
    mean = (counts[:, None] * means).sum(0) / total
    var = (counts[:, None] * (varss + means ** 2)).sum(0) / total - mean ** 2
    return mean, var


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
        
        self.is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)

        # Determine input size from the flattened observation space
        obs_shape = int(np.array(envs.single_observation_space.shape).prod())
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        if self.is_continuous:
            action_dim = np.prod(envs.single_action_space.shape)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, action_dim), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
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
        if self.is_continuous:
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = torch.distributions.Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
        else:
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class Actor(nn.Module):
    """Policy-only network (no critic). Used for the NR-MDP-style architecture
    where the protagonist and the adversary each have their own actor but share
    a single critic."""
    def __init__(self, envs):
        super().__init__()
        self.is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
        obs_shape = int(np.array(envs.single_observation_space.shape).prod())

        if self.is_continuous:
            action_dim = np.prod(envs.single_action_space.shape)
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, action_dim), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(obs_shape, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            )

    def get_action(self, x, action=None):
        x = x.reshape(x.shape[0], -1)
        if self.is_continuous:
            action_mean = self.actor_mean(x)
            action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
            probs = torch.distributions.Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)
        else:
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy()


class Critic(nn.Module):
    """Single shared state-value network V(s). In the action-robust (NR-MDP) game
    the protagonist and adversary play a zero-sum game over one reward stream, so
    there is a single value function for both."""
    def __init__(self, envs):
        super().__init__()
        obs_shape = int(np.array(envs.single_observation_space.shape).prod())
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.critic(x)


def update_agent(agent, optimizer, obs, actions, logprobs, rewards, values, dones, controlled, next_obs, next_done, args, writer, global_step, agent_name="agent"):
    # `controlled` is a (num_steps, num_envs) float mask: 1.0 on the steps this
    # agent actually chose the executed action, 0.0 otherwise. The policy/entropy
    # loss is restricted to controlled steps (the stored action == executed action
    # there, so reward/next-state are valid), while the critic trains on all steps
    # since the state-value target is well defined regardless of who acted.
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
    b_mask = controlled.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            if getattr(agent, "is_continuous", False):
                b_action_batch = b_actions[mb_inds]
            else:
                b_action_batch = b_actions.long()[mb_inds]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_action_batch)
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            # Steps this agent actually controlled within the minibatch
            mb_mask = b_mask[mb_inds]
            mb_bool = mb_mask.bool()
            denom = mb_mask.sum().clamp(min=1.0)

            with torch.no_grad():
                old_approx_kl = (-logratio[mb_bool]).mean() if mb_bool.any() else torch.zeros((), device=device)
                approx_kl = (((ratio - 1) - logratio)[mb_bool]).mean() if mb_bool.any() else torch.zeros((), device=device)
                if mb_bool.any():
                    clipfracs += [((ratio[mb_bool] - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss (masked to controlled steps only)
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_mask).sum() / denom

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

            entropy_loss = (entropy * mb_mask).sum() / denom
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


def update_robust(prot_actor, adv_actor, critic, opt_prot, opt_adv,
                  obs, actions_prot, logprobs_prot, actions_adv, logprobs_adv,
                  rewards, values, dones, controllers, next_obs, next_done,
                  args, writer, global_step, update_adversary=True):
    """NR-MDP-style joint update with ONE shared critic.

    The reward stream is the protagonist's (+r). GAE/advantages and the value
    targets are computed ONCE from the shared critic. The critic is trained once
    (with the protagonist optimizer). Both actors reuse those advantages:
      - protagonist ascends +A on the steps it controlled  (mask = 1 - controllers)
      - adversary    ascends -A on the steps it controlled  (mask =     controllers)

    `update_adversary` implements the article's two-timescale (Tessler et al.,
    Alg. 5): the protagonist+critic are updated every call, but the adversary is
    updated only on the calls where this flag is True (once every N protagonist
    updates), keeping the adversary on the slow timescale.
    Returns the protagonist's flattened GAE returns (for the alpha barrier).
    """
    device = obs.device

    # --- GAE on the shared critic, protagonist-sign reward (computed once) ---
    with torch.no_grad():
        next_value = critic.get_value(next_obs).reshape(1, -1)
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

    # --- Flatten the batch ---
    b_obs = obs.reshape((-1,) + obs.shape[2:])
    b_logp_p = logprobs_prot.reshape(-1)
    b_logp_a = logprobs_adv.reshape(-1)
    b_act_p = actions_prot.reshape((-1,) + actions_prot.shape[2:])
    b_act_a = actions_adv.reshape((-1,) + actions_adv.shape[2:])
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_prot_mask = (1.0 - controllers).reshape(-1)   # 1.0 where protagonist acted
    b_adv_mask = controllers.reshape(-1)            # 1.0 where adversary acted

    if not getattr(prot_actor, "is_continuous", False):
        b_act_p = b_act_p.long()
        b_act_a = b_act_a.long()

    prot_params = list(prot_actor.parameters()) + list(critic.parameters())

    b_inds = np.arange(args.batch_size)
    approx_kl = torch.zeros((), device=device)
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb = b_inds[start:end]

            # advantages are shared; normalize once over the minibatch
            mb_adv = b_advantages[mb]
            if args.norm_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            # ---------- protagonist actor + shared critic ----------
            _, newlogp_p, ent_p = prot_actor.get_action(b_obs[mb], b_act_p[mb])
            ratio_p = (newlogp_p - b_logp_p[mb]).exp()
            mp = b_prot_mask[mb]
            denom_p = mp.sum().clamp(min=1.0)
            pg1 = -mb_adv * ratio_p
            pg2 = -mb_adv * torch.clamp(ratio_p, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss_p = (torch.max(pg1, pg2) * mp).sum() / denom_p
            ent_loss_p = (ent_p * mp).sum() / denom_p

            newvalue = critic.get_value(b_obs[mb]).view(-1)
            if args.clip_vloss:
                v_un = (newvalue - b_returns[mb]) ** 2
                v_cl = (b_values[mb] + torch.clamp(newvalue - b_values[mb], -args.clip_coef, args.clip_coef) - b_returns[mb]) ** 2
                v_loss = 0.5 * torch.max(v_un, v_cl).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

            loss_prot = pg_loss_p - args.ent_coef * ent_loss_p + args.vf_coef * v_loss
            opt_prot.zero_grad()
            loss_prot.backward()
            nn.utils.clip_grad_norm_(prot_params, args.max_grad_norm)
            opt_prot.step()

            # ---------- adversary actor (ascends the NEGATED advantage) ----------
            # Only on the slow-timescale calls (see `update_adversary`).
            if update_adversary:
                _, newlogp_a, ent_a = adv_actor.get_action(b_obs[mb], b_act_a[mb])
                ratio_a = (newlogp_a - b_logp_a[mb]).exp()
                ma = b_adv_mask[mb]
                denom_a = ma.sum().clamp(min=1.0)
                adv_neg = -mb_adv
                pa1 = -adv_neg * ratio_a
                pa2 = -adv_neg * torch.clamp(ratio_a, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_a = (torch.max(pa1, pa2) * ma).sum() / denom_a
                ent_loss_a = (ent_a * ma).sum() / denom_a

                loss_adv = pg_loss_a - args.ent_coef * ent_loss_a
                opt_adv.zero_grad()
                loss_adv.backward()
                nn.utils.clip_grad_norm_(adv_actor.parameters(), args.max_grad_norm)
                opt_adv.step()

            with torch.no_grad():
                mp_bool = mp.bool()
                if mp_bool.any():
                    approx_kl = (((ratio_p - 1) - (newlogp_p - b_logp_p[mb]))[mp_bool]).mean()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    writer.add_scalar("losses/critic_value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/protagonist_policy_loss", pg_loss_p.item(), global_step)
    writer.add_scalar("losses/protagonist_entropy", ent_loss_p.item(), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    writer.add_scalar("losses/protagonist_approx_kl", approx_kl.item(), global_step)
    if update_adversary:
        writer.add_scalar("losses/adversary_policy_loss", pg_loss_a.item(), global_step)
        writer.add_scalar("losses/adversary_entropy", ent_loss_a.item(), global_step)

    return b_returns