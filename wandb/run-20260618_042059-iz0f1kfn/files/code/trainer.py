import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from helper import Args, make_env, Actor, Critic, update_robust, get_obs_norm_stats

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__protagonist__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name,
                   sync_tensorboard=True,
                   config=vars(args),
                   name=run_name,
                   monitor_gym=True,
                   save_code=True)
        
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, 
                                                 value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )

    # NR-MDP architecture: one protagonist actor, one adversary actor, ONE shared critic.
    prot_actor = Actor(envs).to(device)
    adv_actor = Actor(envs).to(device)
    critic = Critic(envs).to(device)

    # The shared critic is trained together with the protagonist; the adversary
    # only updates its own actor, on a faster timescale (two-timescale separation).
    optimizer_prot = optim.Adam(list(prot_actor.parameters()) + list(critic.parameters()), lr=args.learning_rate, eps=1e-5)
    optimizer_adv = optim.Adam(adv_actor.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage for training data (single obs/value/reward stream; one action+logprob per actor)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_prot = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    actions_adv = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_prot = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_adv = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # 1.0 where the adversary chose the executed action, 0.0 where the protagonist did
    controllers = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Init args for alpha updates
    lambda_threshold = args.lambda_threshold
    curr_alpha = args.start_alpha
    max_alpha = args.max_alpha
    lr_alpha = args.lr_alpha
    barrier_t = args.barrier_t
    alpha_step_limit = args.alpha_step_limit
    alpha_clip_count = 0
    v_ema_beta = args.v_ema_beta
    v_ema = None  # running estimate of raw episodic return V(alpha); set on first completed episode
    len_ema = None  # running effective horizon (steps-per-episode), used to scale grad_V to episodic units

    envs.set_attr('alpha', float(curr_alpha))

    # Start training loop
    print("\n--- Starting Training ---\n")

    for iteration in range(1, args.num_iterations + 1):

        batch_episodic_returns = []

        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow_adv = frac * args.learning_rate
            lrnow_prot = frac * args.learning_rate
            optimizer_adv.param_groups[0]["lr"] = lrnow_adv
            optimizer_prot.param_groups[0]["lr"] = lrnow_prot

        # Collect data for num_steps steps
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action_p, logprob_p, _ = prot_actor.get_action(next_obs)
                action_a, logprob_a, _ = adv_actor.get_action(next_obs)
                values[step] = critic.get_value(next_obs).flatten()  # single shared critic

            actions_prot[step] = action_p
            actions_adv[step] = action_a
            logprobs_prot[step] = logprob_p
            logprobs_adv[step] = logprob_a

            # Choose action of protagonist or adversary according to alpha
            adv_wins = np.random.random(args.num_envs) < curr_alpha
            controllers[step] = torch.as_tensor(adv_wins, dtype=torch.float32, device=device)
            if getattr(prot_actor, "is_continuous", False):
                adv_wins_expanded = adv_wins[:, None]
            else:
                adv_wins_expanded = adv_wins
            action = np.where(adv_wins_expanded, action_a.cpu().numpy(), action_p.cpu().numpy())


            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = np.logical_or(terminations, truncations)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if terminations.any() or truncations.any():
                if "final_info" in infos:
                    for i, info in enumerate(infos["final_info"]):
                        if info and "episode" in info:
                            ep_r = info["episode"]["r"]
                            ep_r = ep_r.item() if hasattr(ep_r, 'item') else ep_r
                            
                            print(f"global_step={global_step}, episodic_return={ep_r}")
                            
                            writer.add_scalar("charts/episodic_return", ep_r, global_step + i)
                            if args.track:
                                wandb.log({"Custom_Metrics/Episodic_Return": ep_r}, step=global_step + i)

                            batch_episodic_returns.append(ep_r)
                
                elif "episode" in infos and "_episode" in infos:
                    for i, done in enumerate(infos["_episode"]):
                        if done:
                            ep_r = infos["episode"]["r"][i] if isinstance(infos["episode"], dict) else infos["episode"][i]["r"]
                            ep_r = ep_r.item() if hasattr(ep_r, 'item') else ep_r
                            
                            print(f"global_step={global_step}, episodic_return={ep_r}")
                            
                            writer.add_scalar("charts/episodic_return", ep_r, global_step + i)
                            if args.track:
                                wandb.log({"Custom_Metrics/Episodic_Return": ep_r}, step=global_step + i)
                                
                            batch_episodic_returns.append(ep_r)

        update_adversary = (iteration % args.actor_adv_ratio == 0)
        b_returns = update_robust(prot_actor, adv_actor, critic, optimizer_prot, optimizer_adv,
                                  obs, actions_prot, logprobs_prot, actions_adv, logprobs_adv,
                                  rewards, values, dones, controllers, next_obs, next_done,
                                  args, writer, global_step, update_adversary=update_adversary)

        writer.add_scalar("charts/learning_rate", optimizer_prot.param_groups[0]["lr"], global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if len(batch_episodic_returns) > 0:
            batch_V = float(np.mean(batch_episodic_returns))
            v_ema = batch_V if v_ema is None else v_ema_beta * v_ema + (1.0 - v_ema_beta) * batch_V
            # Effective horizon N/M = steps-in-batch / episodes-completed: converts the
            # per-step mean of (score * return) into the per-episode SUM that dV/dalpha needs.
            batch_len = (args.num_steps * args.num_envs) / len(batch_episodic_returns)
            len_ema = batch_len if len_ema is None else v_ema_beta * len_ema + (1.0 - v_ema_beta) * batch_len

        alpha_step = 0.0
        constraint = float("nan")
        grad_V_robust_star = float("nan")

        if v_ema is not None:
            constraint = v_ema - lambda_threshold

            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_actions = actions_prot.reshape((-1,) + envs.single_action_space.shape)
            if not getattr(prot_actor, "is_continuous", False):
                b_actions = b_actions.long()
            with torch.no_grad():
                _, logp_p, _ = prot_actor.get_action(b_obs, b_actions)
                _, logp_a, _ = adv_actor.get_action(b_obs, b_actions)
                r = torch.exp(torch.clamp(logp_a - logp_p, -20.0, 20.0))
                score = (r - 1.0) / ((1.0 - curr_alpha) + curr_alpha * r + 1e-8)
                grad_V_robust_star = (b_returns * score).mean().item()
                # (1) De-normalize into raw reward units: NormalizeReward scales rewards by
                #     sqrt(return_rms.var), so b_returns -- and thus grad_V -- are normalized.
                try:
                    rew_vars = [float(np.asarray(e.get_wrapper_attr('return_rms').var)) for e in envs.envs]
                    grad_V_robust_star *= float(np.sqrt(np.mean(rew_vars)))
                except AttributeError:
                    pass
                # (2) Scale to the EPISODIC sum: the estimator above averages over steps,
                #     but dV/dalpha sums score*return over a whole episode (factor N/M).
                #if len_ema is not None:
                #    grad_V_robust_star *= len_ema
            
            if args.alpha_method == "barrier":
                L_derivative_alpha = 1.0 + barrier_t * grad_V_robust_star / max(constraint, 1e-6)
                alpha_step = lr_alpha * L_derivative_alpha
            else:
                alpha_step = lr_alpha * constraint / np.abs(grad_V_robust_star) if grad_V_robust_star != 0.0 else 0.0

            if abs(alpha_step) > alpha_step_limit:
                alpha_clip_count += 1
            clipped_alpha_step = float(np.clip(alpha_step, -alpha_step_limit, alpha_step_limit))
            curr_alpha = float(np.clip(curr_alpha + clipped_alpha_step, 0.0, max_alpha))
            envs.set_attr('alpha', float(curr_alpha))

        # Add robustness metrics to WanDB and TensorBoard
        writer.add_scalar("Robustness/alpha", curr_alpha, global_step)
        writer.add_scalar("Robustness/V_ema", v_ema if v_ema is not None else float("nan"), global_step)
        writer.add_scalar("Robustness/constraint", constraint, global_step)
        writer.add_scalar("Robustness/alpha_step", alpha_step, global_step)
        writer.add_scalar("Robustness/grad_V_alpha", grad_V_robust_star, global_step)
        writer.add_scalar("Robustness/alpha_clip_count", alpha_clip_count, global_step)

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    model_path = f"runs/{run_name}/robust_protagonist.pt"
    checkpoint = {
        "prot_actor": prot_actor.state_dict(),
        "adv_actor": adv_actor.state_dict(),
        "critic": critic.state_dict(),
    }
    obs_stats = get_obs_norm_stats(envs)  # freeze the obs normalization with the weights
    if obs_stats is not None:
        checkpoint["obs_mean"], checkpoint["obs_var"] = obs_stats
    torch.save(checkpoint, model_path)

    print("\n--- Training Complete ---\n")

    envs.close()
    writer.close()
    if args.track:
        wandb.finish()