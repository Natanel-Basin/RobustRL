import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from helper import Args, make_env, Agent, update_agent

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

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

    # Init protagonist and adversary agents
    prot_agent = Agent(envs)
    prot_agent.reward_sign = 1.0
    optimizer_prot = optim.Adam(prot_agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    adv_agent = Agent(envs)
    adv_agent.reward_sign = -1.0
    optimizer_adv = optim.Adam(adv_agent.parameters(), lr=args.learning_rate * 9.0, eps=1e-5)

    # Storage for training data
    obs_prot = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    obs_adv = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_prot = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    actions_adv = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_prot = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_adv = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_prot = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_adv = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_prot = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_adv = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Init args for alpha updates
    lambda_threshold = args.lambda_threshold
    curr_alpha = args.start_alpha
    max_alpha = args.max_alpha
    nu_alpha = args.nu_alpha
    barrier_t = args.barrier_t
    alpha_step_limit = args.alpha_step_limit
    alpha_clip_count = 0

    envs.set_attr('alpha', float(curr_alpha))


    # Start training loop
    print("\n--- Starting Training ---\n")

    for iteration in range(1, args.num_iterations + 1):

        batch_episodic_returns = []

        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow_adv = frac * args.learning_rate
            lrnow_prot = frac * args.learning_rate # Match protagonist LR with adversary
            optimizer_adv.param_groups[0]["lr"] = lrnow_adv
            optimizer_prot.param_groups[0]["lr"] = lrnow_prot

        # Collect data for num_steps steps
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            
            obs_prot[step] = next_obs
            obs_adv[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action_p, logprob_p, _, value_p = prot_agent.get_action_and_value(next_obs)
                action_a, logprob_a, _, value_a = adv_agent.get_action_and_value(next_obs)
                
                values_prot[step] = value_p.flatten()
                values_adv[step] = value_a.flatten()
            
            actions_prot[step] = action_p
            actions_adv[step] = action_a
            logprobs_prot[step] = logprob_p
            logprobs_adv[step] = logprob_a

            # Choose action of protagonist or adversary according to alpha
            adv_wins = np.random.random(args.num_envs) < curr_alpha
            if getattr(prot_agent, "is_continuous", False):
                adv_wins_expanded = adv_wins[:, None]
            else:
                adv_wins_expanded = adv_wins
            action = np.where(adv_wins_expanded, action_a.cpu().numpy(), action_p.cpu().numpy())


            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = np.logical_or(terminations, truncations)

            shared_reward = torch.tensor(reward).to(device).view(-1)
            rewards_prot[step] = prot_agent.reward_sign * shared_reward
            rewards_adv[step] = adv_agent.reward_sign * shared_reward
            
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Log episodic returns for any finished episodes
            if terminations.any() or truncations.any():
                if "final_info" in infos:
                    for i, info in enumerate(infos["final_info"]):
                        if info and "episode" in info:
                            ep_r = info["episode"]["r"]
                            ep_r = ep_r.item() if hasattr(ep_r, 'item') else ep_r
                            
                            # print(f"global_step={global_step}, episodic_return={ep_r}")
                            
                            writer.add_scalar("charts/episodic_return", ep_r, global_step + i)
                            if args.track:
                                wandb.log({"Custom_Metrics/Episodic_Return": ep_r}, step=global_step + i)

                            batch_episodic_returns.append(ep_r)
                
                elif "episode" in infos and "_episode" in infos:
                    for i, done in enumerate(infos["_episode"]):
                        if done:
                            ep_r = infos["episode"]["r"][i] if isinstance(infos["episode"], dict) else infos["episode"][i]["r"]
                            ep_r = ep_r.item() if hasattr(ep_r, 'item') else ep_r
                            
                            # print(f"global_step={global_step}, episodic_return={ep_r}")
                            
                            writer.add_scalar("charts/episodic_return", ep_r, global_step + i)
                            if args.track:
                                wandb.log({"Custom_Metrics/Episodic_Return": ep_r}, step=global_step + i)
                                
                            batch_episodic_returns.append(ep_r)

        # Update both agents side-by-side using the same batch of data
        update_agent(adv_agent, optimizer_adv, obs_adv, actions_adv, logprobs_adv, rewards_adv, values_adv, dones, next_obs, next_done, args, writer, global_step, agent_name="adversary")
        b_returns = update_agent(prot_agent, optimizer_prot, obs_prot, actions_prot, logprobs_prot, rewards_prot, values_prot, dones, next_obs, next_done, args, writer, global_step, agent_name="protagonist")

        writer.add_scalar("charts/learning_rate", optimizer_prot.param_groups[0]["lr"], global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Compute V_robust_star
        if len(batch_episodic_returns) > 0:
            V_robust_star = np.mean(batch_episodic_returns)
        else:
            V_robust_star = b_returns.mean().item() 

        b_obs = obs_prot.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions_prot.reshape((-1,) + envs.single_action_space.shape)

        # Compute policy probabilities under both agents for the collected batch
        with torch.no_grad():
            _, logprob_p, _, _ = prot_agent.get_action_and_value(b_obs, b_actions)
            prob_p = logprob_p.exp()
            _, logprob_a, _, _ = adv_agent.get_action_and_value(b_obs, b_actions)
            prob_a = logprob_a.exp()

        # Find next alpha using log barrier method   
        mix_prob = (1 - curr_alpha) * prob_p + curr_alpha * prob_a
        
        centered_returns = b_returns - b_returns.mean()
        standard_returns = centered_returns / (b_returns.std() + 1e-8)
        
        grad_V_terms = standard_returns * (prob_a - prob_p) / (mix_prob + 1e-8)
        grad_V_robust_star = grad_V_terms.mean().item()
        
        denominator = V_robust_star - lambda_threshold

        L_derivative_alpha = 1.0 - barrier_t * grad_V_robust_star / max(denominator, 0.05)
        polyak_step = (V_robust_star - lambda_threshold) / (abs(grad_V_robust_star) + 1e-8) ** 2  
        alpha_step = polyak_step * L_derivative_alpha
        
        if abs(alpha_step) > alpha_step_limit:
            alpha_clip_count += 1
        
        clipped_alpha_step = np.clip(alpha_step, -alpha_step_limit, alpha_step_limit)
        curr_alpha = np.clip(curr_alpha + clipped_alpha_step, 0.0, max_alpha)

        envs.set_attr('alpha', float(curr_alpha))

        # Add robustness metrics to WanDB and TensorBoard
        writer.add_scalar("Robustness/alpha", curr_alpha, global_step)
        writer.add_scalar("Robustness/V_robust_star", V_robust_star, global_step)
        writer.add_scalar("Robustness/grad_V_alpha", grad_V_robust_star, global_step)
        writer.add_scalar("Robustness/alpha_clip_count", alpha_clip_count, global_step)

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    model_path = f"runs/{run_name}/robust_protagonist.pt"
    torch.save(prot_agent.state_dict(), model_path)

    print("\n--- Training Complete ---\n")

    envs.close()
    writer.close()
    if args.track:
        wandb.finish()