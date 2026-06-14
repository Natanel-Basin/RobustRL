import os
import glob
import tyro
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # headless: render straight to a file, no display needed
import matplotlib.pyplot as plt

from helper import Args, Agent, Actor, make_env


def set_env_dynamics(env, args, param1_val, param2_val):
    """Apply the perturbed physics parameters to the (single) test env."""
    if "MountainCar" in args.env_id:
        env.unwrapped.force = param1_val
        env.unwrapped.gravity = param2_val
    elif "Walker2d" in args.env_id:
        env.unwrapped.model.opt.gravity[2] = param1_val
        if not hasattr(env, "original_body_mass"):
            env.original_body_mass = env.unwrapped.model.body_mass.copy()
        env.unwrapped.model.body_mass[:] = env.original_body_mass * param2_val


def evaluate(agent, envs, args, device, param1_val, param2_val, obs_stats=None):
    env = envs.envs[0]
    apply_obs_norm(env, obs_stats)
    set_env_dynamics(env, args, param1_val, param2_val)

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
                episode_reward += reward
                obs = next_obs

            total_eval_returns.append(episode_reward)

    return np.mean(total_eval_returns)

def beta_test(agent, envs, device, beta, num_episodes_beta_test, obs_stats=None):
    env = envs.envs[0]
    apply_obs_norm(env, obs_stats)
    total_eval_returns = []
    with torch.no_grad():
        for test_seed in range(num_episodes_beta_test):
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
                if np.random.random() < beta:
                    action = env.action_space.sample()
                elif getattr(agent, "is_continuous", False):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                obs = next_obs

            total_eval_returns.append(episode_reward)

    return total_eval_returns


def load_checkpoint(agent, model_path, device, state_key=None):
    """Load one checkpoint file into `agent`.

    Checkpoints are dicts: the baseline saves {"agent", "obs_mean", "obs_var"} and
    the trainer saves {"prot_actor", "adv_actor", "critic", "obs_mean", "obs_var"}.
    `state_key` selects which sub-state-dict to load. Older flat state_dicts (no
    obs stats) are still supported for backward compatibility.

    Returns (agent, obs_stats) where obs_stats is (mean, var) or None."""
    # weights_only=False: our checkpoints embed numpy obs-norm stats (obs_mean/var),
    # which PyTorch >=2.6 refuses to unpickle under the new weights_only=True default.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    obs_stats = None
    if isinstance(checkpoint, dict) and "obs_mean" in checkpoint:
        obs_stats = (checkpoint["obs_mean"], checkpoint["obs_var"])

    state = checkpoint
    if state_key is not None and isinstance(checkpoint, dict) and state_key in checkpoint:
        state = checkpoint[state_key]

    agent.load_state_dict(state)
    agent.eval()
    return agent, obs_stats


def evaluate_curves(agent, obs_stats, envs, args, device,
                    param1_values, param2_values, noise_values, default_param1, default_param2):
    """Run all three robustness sweeps for one agent and return the curves as
    numpy arrays: (param1, param2, noise-probability)."""
    p1 = np.array([evaluate(agent, envs, args, device, param1_val=v, param2_val=default_param2, obs_stats=obs_stats)
                   for v in param1_values])
    p2 = np.array([evaluate(agent, envs, args, device, param1_val=default_param1, param2_val=v, obs_stats=obs_stats)
                   for v in param2_values])
    # Restore default physics before the action-noise test (the mass sweep left them perturbed).
    set_env_dynamics(envs.envs[0], args, default_param1, default_param2)
    noise = np.array([np.mean(beta_test(agent, envs, device, float(p), args.eval_episodes, obs_stats=obs_stats))
                      for p in noise_values])
    return p1, p2, noise


def apply_obs_norm(env, obs_stats):
    """Restore the training-time observation normalization onto the test env and
    freeze it, so the policy sees the same normalized inputs it was trained on."""
    if obs_stats is None:
        return
    mean, var = obs_stats
    try:
        obs_rms = env.get_wrapper_attr('obs_rms')
    except AttributeError:
        return
    obs_rms.mean = np.asarray(mean, dtype=np.float64)
    obs_rms.var = np.asarray(var, dtype=np.float64)
    try:
        env.set_wrapper_attr('update_running_mean', False)  # stop re-estimating at test
    except AttributeError:
        pass

def plot_band(ax, x, scores, label, color):
    """Plot the per-seed mean as a line and shade +/-1 std across seeds.
    `scores` has shape (n_seeds, n_points)."""
    scores = np.asarray(scores)
    n = scores.shape[0]
    mean = scores.mean(axis=0)
    ax.plot(x, mean, marker='o', linewidth=2, color=color, label=f"{label} (n={n})")
    if n > 1:
        std = scores.std(axis=0, ddof=1)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, "test", args.gamma, test_mode=True) for i in range(args.num_envs)]
    )

    # Aggregate over ALL matching checkpoints in runs/ (one per training seed/run).
    baseline_paths = sorted(glob.glob(os.path.join("runs", "*", "baseline_agent.pt")))
    robust_paths = sorted(glob.glob(os.path.join("runs", "*", "robust_protagonist.pt")))
    if not baseline_paths or not robust_paths:
        print("ERROR: need at least one baseline_agent.pt and one robust_protagonist.pt under runs/")
        exit()
    print(f"Aggregating over {len(baseline_paths)} baseline and {len(robust_paths)} robust checkpoints")

    envs.set_attr('alpha', 0.0)

    # --- parameter robustness test ---
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

    num_robust_values = 50
    deviation = 0.35

    param1_values = np.linspace(default_param1 * (1 - deviation), default_param1 * (1 + deviation), num_robust_values)
    param2_values = np.linspace(default_param2 * (1 - deviation), default_param2 * (1 + deviation), num_robust_values)
    noise_values = np.linspace(0.0, 0.5, 11)

    def run_method(paths, build_agent, state_key, tag):
        """Evaluate every checkpoint of one method; return stacked curves (n_seeds, n_points)."""
        p1_all, p2_all, noise_all = [], [], []
        for j, path in enumerate(paths):
            print(f"\n[{tag} {j+1}/{len(paths)}] {path}")
            agent, obs_stats = load_checkpoint(build_agent(), path, device, state_key=state_key)
            c1, c2, cn = evaluate_curves(agent, obs_stats, envs, args, device,
                                         param1_values, param2_values, noise_values,
                                         default_param1, default_param2)
            p1_all.append(c1); p2_all.append(c2); noise_all.append(cn)
        return np.array(p1_all), np.array(p2_all), np.array(noise_all)

    base_p1, base_p2, base_noise = run_method(
        baseline_paths, lambda: Agent(envs).to(device), "agent", "baseline")
    rob_p1, rob_p2, rob_noise = run_method(
        robust_paths, lambda: Actor(envs).to(device), "prot_actor", "robust")

    envs.close()

    # --- Create graphs (mean +/- 1 std across seeds) ---
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Robustness Comparison: Standard PPO vs Robust Agent (mean +/- 1 std over seeds)',
                 fontsize=16, fontweight='bold')

    # --- Relative mass ---
    plot_band(axs[0], param2_values, base_p2, "Baseline PPO", "tab:blue")
    plot_band(axs[0], param2_values, rob_p2, "Robust Agent", "tab:orange")
    axs[0].set_title(f"Performance Under Perturbation of {param2_name}")
    axs[0].axvline(x=default_param2, color='red', linestyle='--', alpha=0.5, label='Default')
    axs[0].set_xlabel(f'{param2_name}')
    axs[0].set_ylabel('Average Return')
    axs[0].legend()

    # --- Action-noise probability (paper: 0 -> 0.5) ---
    plot_band(axs[1], noise_values, base_noise, "Baseline PPO", "tab:blue")
    plot_band(axs[1], noise_values, rob_noise, "Robust Agent", "tab:orange")
    axs[1].set_title('Performance Under Random Actions (Noise Probability)')
    axs[1].set_xlabel('Noise Probability')
    axs[1].set_ylabel('Average Return')
    axs[1].legend()

    # --- Gravity (extra; not tested in the paper) ---
    plot_band(axs[2], param1_values, base_p1, "Baseline PPO", "tab:blue")
    plot_band(axs[2], param1_values, rob_p1, "Robust Agent", "tab:orange")
    axs[2].set_title(f"Performance Under Perturbation of {param1_name}")
    axs[2].axvline(x=default_param1, color='red', linestyle='--', alpha=0.5, label='Default')
    axs[2].set_xlabel(f'{param1_name}')
    axs[2].set_ylabel('Average Return')
    axs[2].legend()

    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "robustness_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")