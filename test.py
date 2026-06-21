import os
import glob
import subprocess
import sys
import time
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


def _episode_returns(infos):
    """Pull completed-episode RAW returns out of a vector-env info dict
    (handles both gymnasium autoreset/info formats)."""
    out = []
    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "episode" in info:
                r = info["episode"]["r"]
                out.append(float(r.item() if hasattr(r, "item") else r))
    elif "episode" in infos and "_episode" in infos:
        rs = infos["episode"]["r"]
        for i, done in enumerate(infos["_episode"]):
            if done:
                r = rs[i]
                out.append(float(r.item() if hasattr(r, "item") else r))
    return out


def _policy_actions(agent, obs, device):
    """Deterministic batched actions for ALL sub-envs at once (obs: (num_envs, obs_dim))."""
    obs_tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32).to(device)
    if getattr(agent, "is_continuous", False):
        return agent.actor_mean(obs_tensor).cpu().numpy()
    logits = agent.actor(obs_tensor)
    return torch.argmax(logits, dim=1).cpu().numpy()


def evaluate(agent, envs, args, device, param1_val, param2_val, obs_stats=None):
    """Vectorized evaluation: steps ALL num_envs sub-envs at once and averages the
    raw episodic returns of the first >= eval_episodes that complete."""
    for env in envs.envs:
        apply_obs_norm(env, obs_stats)
        set_env_dynamics(env, args, param1_val, param2_val)

    is_cont = getattr(agent, "is_continuous", False)
    returns = []
    with torch.no_grad():
        obs, _ = envs.reset(seed=args.seed)
        while len(returns) < args.eval_episodes:
            action = _policy_actions(agent, obs, device)
            if is_cont:
                action = np.clip(action, envs.single_action_space.low, envs.single_action_space.high)
            obs, _, _, _, infos = envs.step(action)
            returns.extend(_episode_returns(infos))
    return float(np.mean(returns))


def beta_test(agent, envs, device, beta, num_episodes_beta_test, obs_stats=None):
    """Vectorized action-noise test: with prob beta a uniformly random action replaces
    the policy's action (per env, per step). Returns the list of episodic returns."""
    for env in envs.envs:
        apply_obs_norm(env, obs_stats)

    is_cont = getattr(agent, "is_continuous", False)
    n = len(envs.envs)
    returns = []
    with torch.no_grad():
        obs, _ = envs.reset(seed=0)
        while len(returns) < num_episodes_beta_test:
            action = _policy_actions(agent, obs, device)
            if is_cont:
                action = np.clip(action, envs.single_action_space.low, envs.single_action_space.high)
            mask = np.random.random(n) < beta
            if mask.any():
                rand = np.stack([envs.single_action_space.sample() for _ in range(n)])
                action = np.where(mask[:, None], rand, action) if is_cont else np.where(mask, rand, action)
            obs, _, _, _, infos = envs.step(action)
            returns.extend(_episode_returns(infos))
    return returns


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
    # Restore default physics on ALL sub-envs before the action-noise test
    # (the mass sweep perturbed every one of them).
    for env in envs.envs:
        set_env_dynamics(env, args, default_param1, default_param2)
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

def build_grids(args):
    """Perturbation sweep ranges, derived deterministically from env_id so the
    orchestrator and the per-seed workers compute identical x-axes."""
    if "MountainCar" in args.env_id:
        default_param1, default_param2 = 0.001, 0.0025
        param1_name, param2_name = "Force", "Gravity"
    elif "Walker2d" in args.env_id:
        default_param1, default_param2 = -9.81, 1.0
        param1_name, param2_name = "Gravity", "Body Mass Multiplier"
    else:
        raise ValueError(f"Bounds not defined for environment: {args.env_id}")
    num_robust_values = 30
    deviation = 0.5
    return {
        "param1_values": np.linspace(default_param1 * (1 - deviation), default_param1 * (1 + deviation), num_robust_values),
        "param2_values": np.linspace(default_param2 * (1 - deviation), default_param2 * (1 + deviation), num_robust_values),
        "noise_values": np.linspace(0.0, 0.5, 11),
        "default_param1": default_param1, "default_param2": default_param2,
        "param1_name": param1_name, "param2_name": param2_name,
    }


def make_test_env(args):
    return gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, "test", args.gamma, test_mode=True) for i in range(args.num_envs)])


def eval_checkpoint(path, kind, args, device, grids):
    """Evaluate one checkpoint on a fresh env; return (p1, p2, noise) curves."""
    envs = make_test_env(args)
    envs.set_attr('alpha', 0.0)
    if kind == "baseline":
        agent, state_key = Agent(envs).to(device), "agent"
    else:
        agent, state_key = Actor(envs).to(device), "prot_actor"
    agent, obs_stats = load_checkpoint(agent, path, device, state_key=state_key)
    curves = evaluate_curves(agent, obs_stats, envs, args, device,
                             grids["param1_values"], grids["param2_values"], grids["noise_values"],
                             grids["default_param1"], grids["default_param2"])
    envs.close()
    return curves


def run_parallel(tasks, gpus):
    """Evaluate each (path, kind) in its own subprocess, round-robined across `gpus`
    (one run per GPU). Returns {path: (p1, p2, noise)}. Re-invokes THIS script in
    worker mode via the TEST_WORKER_* environment variables (so each worker rebuilds
    the same config from the same CLI args and pins itself to one GPU)."""
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="test_curves_")
    base_cmd = [sys.executable] + sys.argv
    results, queue, running, free = {}, list(tasks), [], list(gpus)
    n = 0
    while queue or running:
        while queue and free:
            path, kind = queue.pop(0)
            gpu = free.pop(0)
            out = os.path.join(tmpdir, f"curves_{n}.npz"); n += 1
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, TEST_WORKER_CKPT=path,
                       TEST_WORKER_KIND=kind, TEST_WORKER_OUT=out)
            print(f"--> eval {kind} [GPU {gpu}] {path}")
            running.append((subprocess.Popen(base_cmd, env=env), path, kind, gpu, out))
        time.sleep(1)
        still = []
        for proc, path, kind, gpu, out in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, path, kind, gpu, out))
                continue
            free.append(gpu)  # release the GPU
            if rc == 0 and os.path.exists(out):
                d = np.load(out)
                results[path] = (d["p1"], d["p2"], d["noise"])
                print(f"<-- done {kind} {path}")
            else:
                print(f"WARNING: eval failed for {path} (exit {rc}) -- skipping")
        running = still
    return results


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    grids = build_grids(args)

    # --- Worker mode: evaluate a single checkpoint and save its curves, then exit. ---
    worker_ckpt = os.environ.get("TEST_WORKER_CKPT")
    if worker_ckpt:
        p1, p2, noise = eval_checkpoint(worker_ckpt, os.environ["TEST_WORKER_KIND"], args, device, grids)
        np.savez(os.environ["TEST_WORKER_OUT"], p1=p1, p2=p2, noise=noise)
        raise SystemExit(0)

    # --- Orchestrator: aggregate over ALL checkpoints in runs/ (one per training seed). ---
    baseline_paths = sorted(glob.glob(os.path.join("runs", "*", "baseline_agent.pt")))
    robust_paths = sorted(glob.glob(os.path.join("runs", "*", "robust_protagonist.pt")))
    if not baseline_paths or not robust_paths:
        print("ERROR: need at least one baseline_agent.pt and one robust_protagonist.pt under runs/")
        exit()
    print(f"Aggregating over {len(baseline_paths)} baseline and {len(robust_paths)} robust checkpoints")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    tasks = [(p, "baseline") for p in baseline_paths] + [(p, "robust") for p in robust_paths]

    if gpus:
        print(f"Evaluating seeds in parallel across GPUs {gpus} (one seed per GPU)")
        results = run_parallel(tasks, gpus)
    else:
        results = {}
        for j, (path, kind) in enumerate(tasks):
            print(f"[{j + 1}/{len(tasks)}] {kind} {path}")
            results[path] = eval_checkpoint(path, kind, args, device, grids)

    # Stack per-method curves (skipping any that failed), shape (n_seeds, n_points).
    def stack(paths, idx):
        return np.array([results[p][idx] for p in paths if p in results])

    base_p1, base_p2, base_noise = stack(baseline_paths, 0), stack(baseline_paths, 1), stack(baseline_paths, 2)
    rob_p1, rob_p2, rob_noise = stack(robust_paths, 0), stack(robust_paths, 1), stack(robust_paths, 2)
    if len(base_p1) == 0 or len(rob_p1) == 0:
        print("ERROR: no successful evaluations to plot.")
        exit()

    param1_values, param2_values, noise_values = grids["param1_values"], grids["param2_values"], grids["noise_values"]
    default_param1, default_param2 = grids["default_param1"], grids["default_param2"]
    param1_name, param2_name = grids["param1_name"], grids["param2_name"]

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
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "recreating_results_HalfCheetah")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "alpha=0.05.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")