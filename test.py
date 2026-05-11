import os
import glob
import tyro
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

from helper import Args, Agent, make_env

def evaluate(agent, envs, args, device, param1_val, param2_val):
    env = envs.envs[0] 
    
    if "MountainCar" in args.env_id:
        env.unwrapped.force = param1_val
        env.unwrapped.gravity = param2_val
    elif "Walker2d" in args.env_id:
        env.unwrapped.model.opt.gravity[2] = param1_val
        if not hasattr(env, "original_body_mass"):
            env.original_body_mass = env.unwrapped.model.body_mass.copy()
        env.unwrapped.model.body_mass[:] = env.original_body_mass * param2_val

    total_eval_returns = []

    with torch.no_grad():
        for test_seed in range(args.eval_episodes):
            obs, _ = env.reset(seed=test_seed)
            
            episode_reward = 0.0
            done = False

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                
                if getattr(agent, "is_continuous", False):
                    action = agent.actor_mean(obs_tensor).detach().cpu().numpy()[0]
                else:
                    logits = agent.actor(obs_tensor)
                    action = torch.argmax(logits, dim=1).item()
                if getattr(agent, "is_continuous", False):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                """
                if args.env_id == "MountainCar-v0":
                    positions = next_obs[0]
                    velocities = next_obs[1]
                    potential_energy = (positions + 0.5) ** 2
                    kinetic_energy = (velocities * 10) ** 2
                    shaping_bonus = (potential_energy + kinetic_energy) * 10.0
                    win_bonus = 500.0 if positions >= 0.5 else 0.0
                    reward = win_bonus + shaping_bonus
                """
                episode_reward += reward
                obs = next_obs

            total_eval_returns.append(episode_reward)

    return np.mean(total_eval_returns)

def load_latest_model(agent, search_pattern, device):
    """Helper to find and load the latest .pt file for a given pattern."""
    available_models = glob.glob(search_pattern)
    if not available_models:
        print(f"ERROR: Could not find any saved models matching {search_pattern}!")
        exit()
    model_path = max(available_models, key=os.path.getctime)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    return agent

def add_slope(ax, x, y, label_prefix, color):
    # fit line y = ax + b
    p = Polynomial.fit(x, y, 1).convert()
    y_fit = p(x)

    slope = p.coef[1]

    ax.plot(x, y_fit, linestyle="--", color=color, alpha=0.8,
            label=f"{label_prefix} slope={slope:.2f}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, "test", args.gamma, test_mode=True) for i in range(args.num_envs)]
    )

    # Initialize and load Baseline Agent
    baseline_agent = Agent(envs).to(device)
    load_latest_model(baseline_agent, os.path.join("runs", "*", "baseline_agent.pt"), device)

    # Initialize and load Robust Agent
    robust_agent = Agent(envs).to(device)
    load_latest_model(robust_agent, os.path.join("runs", "*", "robust_protagonist.pt"), device)

    envs.set_attr('alpha', 0.0)

    # Custom bounds
    if "MountainCar" in args.env_id:
        default_param1 = 0.001
        default_param2 = 0.0025
        param1_name = "Force"
        param2_name = "Gravity"
    elif "Walker2d" in args.env_id:
        default_param1 = -9.81
        default_param2 = 1.0
        param1_name = "Gravity"
        param2_name = "Body Mass Multiplier"
    else:
        raise ValueError(f"Bounds not defined for environment: {args.env_id}")

    deviation = 0.15
    num_robust_values = 30

    # Generate data points
    param1_values = np.linspace(default_param1 * (1 - deviation), default_param1 * (1 + deviation), num_robust_values)
    param2_values = np.linspace(default_param2 * (1 - deviation), default_param2 * (1 + deviation), num_robust_values)

    # Storage arrays
    p1_base_scores, p1_rob_scores, p1_diff = [], [], []
    p2_base_scores, p2_rob_scores, p2_diff = [], [], []

    print(f"\n--- Testing Robustness for {param1_name} ---")
    for i, p1 in enumerate(param1_values):
        b_score = evaluate(baseline_agent, envs, args, device, param1_val=p1, param2_val=default_param2)
        r_score = evaluate(robust_agent, envs, args, device, param1_val=p1, param2_val=default_param2)
        p1_base_scores.append(b_score)
        p1_rob_scores.append(r_score)
        p1_diff.append(r_score - b_score)
        print(f"Step {i+1}/{len(param1_values)} | {param1_name}={p1:.4f} | Baseline Return: {b_score:.2f} | Robust Return: {r_score:.2f}")

    print(f"\n--- Testing Robustness for {param2_name} ---")
    for i, p2 in enumerate(param2_values):
        b_score = evaluate(baseline_agent, envs, args, device, param1_val=default_param1, param2_val=p2)
        r_score = evaluate(robust_agent, envs, args, device, param1_val=default_param1, param2_val=p2)
        p2_base_scores.append(b_score)
        p2_rob_scores.append(r_score)
        p2_diff.append(r_score - b_score)
        print(f"Step {i+1}/{len(param2_values)} | {param2_name}={p2:.4f} | Baseline Return: {b_score:.2f} | Robust Return: {r_score:.2f}")

    envs.close()

    # --- Create graphs ---
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Robustness Comparison: Standard PPO vs Robust Agent', fontsize=16, fontweight='bold')

    # --- Param1 Plot ---
    axs[0].plot(param1_values, p1_base_scores, marker='o', linewidth=2, label='Baseline PPO')
    axs[0].plot(param1_values, p1_rob_scores, marker='o', linewidth=2, label='Robust Agent')

    axs[0].axvline(x=default_param1, color='red', linestyle='--', alpha=0.5, label='Default')
    axs[0].set_title(f'Performance vs {param1_name}')
    axs[0].set_xlabel(f'{param1_name} Value')
    axs[0].set_ylabel('Average Return')
    add_slope(axs[0], param1_values, p1_base_scores, "Baseline", "blue")
    add_slope(axs[0], param1_values, p1_rob_scores, "Robust", "orange")
    axs[0].legend()

    # --- Param2 Plot ---
    axs[1].plot(param2_values, p2_base_scores, marker='o', linewidth=2, label='Baseline PPO')
    axs[1].plot(param2_values, p2_rob_scores, marker='o', linewidth=2, label='Robust Agent')

    axs[1].axvline(x=default_param2, color='red', linestyle='--', alpha=0.5, label='Default')
    axs[1].set_title(f'Performance vs {param2_name}')
    axs[1].set_xlabel(f'{param2_name} Value')
    axs[1].set_ylabel('Average Return')
    add_slope(axs[1], param2_values, p2_base_scores, "Baseline", "blue")
    add_slope(axs[1], param2_values, p2_rob_scores, "Robust", "orange")
    axs[1].legend()

    plt.tight_layout()
    plt.show()