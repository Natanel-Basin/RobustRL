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
                   entity=args.wandb_entity,
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
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )

    # Init protagonist and adversary agents
    prot_agent = Agent(envs)
    prot_agent.reward_sign = 1.0
    optimizer_prot = optim.Adam(prot_agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    adv_agent = Agent(envs)
    adv_agent.reward_sign = -1.0
    optimizer_adv = optim.Adam(adv_agent.parameters(), lr=args.learning_rate / 10, eps=1e-5)

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
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Init args for learning alpha and eta parameters
    lambda_threshold = args.lambda_threshold
    curr_alpha = args.start_alpha
    max_alpha = args.max_alpha
    curr_eta = args.start_eta
    nu_alpha = args.nu_alpha
    nu_eta = args.nu_eta

    envs.set_attr('alpha', float(curr_alpha))
    
    num_outer_iterations = args.num_iterations // args.inner_updates
    iteration = 0

    print("\n--- Starting Training ---\n")

    # Outer loop
    for outer_iteration in range(1, num_outer_iterations + 1):

        # Inner loop
        for inner_iteration in range(args.inner_updates):
            iteration += 1
            batch_episodic_returns = []

            # Annealing the learning rate
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer_prot.param_groups[0]["lr"] = lrnow
                optimizer_adv.param_groups[0]["lr"] = lrnow

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

                adv_wins = np.random.random(args.num_envs) < curr_alpha
                if getattr(prot_agent, "is_continuous", False):
                    adv_wins_expanded = adv_wins[:, None]
                else:
                    adv_wins_expanded = adv_wins
                action = np.where(adv_wins_expanded, action_a.cpu().numpy(), action_p.cpu().numpy())
                            
                next_obs, reward, terminations, truncations, infos = envs.step(action)
                next_done = np.logical_or(terminations, truncations)

                # Reward shaping for MountainCar
                if args.env_id == "MountainCar-v0":
                    positions = next_obs[:, 0]
                    velocities = next_obs[:, 1]
                    potential_energy = (positions + 0.5) ** 2
                    kinetic_energy = (velocities * 10) ** 2
                    shaping_bonus = (potential_energy + kinetic_energy) * 10.0
                    win_bonus = np.where(positions >= 0.5, 500.0, 0.0)
                    reward = win_bonus + shaping_bonus

                shared_reward = torch.tensor(reward).to(device).view(-1)
                rewards_prot[step] = prot_agent.reward_sign * shared_reward
                rewards_adv[step] = adv_agent.reward_sign * shared_reward
                
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "episode" in infos and "_episode" in infos:
                    for i, done in enumerate(infos["_episode"]):
                        if done: 
                            if isinstance(infos["episode"], dict):
                                ep_r = infos["episode"]["r"][i]
                                ep_l = infos["episode"]["l"][i]
                            else:
                                ep_r = infos["episode"][i]["r"]
                                ep_l = infos["episode"][i]["l"]
                                
                            ep_r = ep_r.item() if hasattr(ep_r, 'item') else ep_r
                            ep_l = ep_l.item() if hasattr(ep_l, 'item') else ep_l
                            
                            # Original print format restored
                            print(f"global_step={global_step}, episodic_return={ep_r}")
                            
                            writer.add_scalar("charts/episodic_return", ep_r, global_step)
                            writer.add_scalar("charts/episodic_length", ep_l, global_step)

                            batch_episodic_returns.append(ep_r)

            # Update prot and adv
            b_returns = update_agent(prot_agent,
                                     optimizer_prot,
                                     obs_prot,
                                     actions_prot,
                                     logprobs_prot,
                                     rewards_prot, 
                                     values_prot,
                                     dones,
                                     next_obs,
                                     next_done,
                                     args,
                                     writer,
                                     global_step, 
                                     agent_name="protagonist")
            update_agent(adv_agent,
                             optimizer_adv,
                             obs_adv,
                             actions_adv,
                             logprobs_adv,
                             rewards_adv,
                             values_adv,
                             dones,
                             next_obs,
                             next_done,
                             args,
                             writer,
                             global_step,
                             agent_name="adversary")

            # Original SPS print restored
            writer.add_scalar("charts/learning_rate", optimizer_prot.param_groups[0]["lr"], global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Update alpha and eta
        if len(batch_episodic_returns) > 0:
            V_robust_star = np.mean(batch_episodic_returns)
        else:
            V_robust_star = -1000.0  
        
        b_obs = obs_prot.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions_prot.reshape((-1,) + envs.single_action_space.shape)

        with torch.no_grad():
            _, logprob_p, _, _ = prot_agent.get_action_and_value(b_obs, b_actions)
            prob_p = logprob_p.exp()
            _, logprob_a, _, _ = adv_agent.get_action_and_value(b_obs, b_actions)
            prob_a = logprob_a.exp()

        mix_prob = (1 - curr_alpha) * prob_p + curr_alpha * prob_a
        grad_V_terms = b_returns * (prob_a - prob_p) / (mix_prob + 1e-8)
        grad_V_robust_star = grad_V_terms.mean().item()
        '''
        curr_eta = max(0.0, curr_eta + nu_eta * (lambda_threshold - V_robust_star))
        
        grad_L_alpha = 1.0 + curr_eta * grad_V_robust_star
        curr_alpha = np.clip(curr_alpha + nu_alpha * grad_L_alpha, 0.0, max_alpha)
        '''
        envs.set_attr('alpha', float(curr_alpha))

        writer.add_scalar("Robustness/alpha", curr_alpha, global_step)
        writer.add_scalar("Robustness/eta", curr_eta, global_step)
        writer.add_scalar("Robustness/V_robust_star", V_robust_star, global_step)
        writer.add_scalar("Robustness/grad_V_alpha", grad_V_robust_star, global_step)

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    model_path = f"runs/{run_name}/robust_protagonist.pt"
    torch.save(prot_agent.state_dict(), model_path)

    envs.close()
    if args.track:
        writer.close()