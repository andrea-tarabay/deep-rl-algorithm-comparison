import argparse
import os
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device),
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        action = torch.tanh(self.l3(x))
        return self.max_action * action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q1 network
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 network
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400 + action_dim, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = torch.cat([q1, action], dim=1)
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = torch.cat([q2, action], dim=1)
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def q1(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = torch.cat([q1, action], dim=1)
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class TD3:
    def __init__(self, state_dim, action_dim, max_action, action_low, action_high, device):
        self.device = device
        self.max_action = max_action

        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            action = self.actor(state)

        return action.cpu().numpy().flatten()

    def train(
        self,
        replay_buffer,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size, self.device
        )

        with torch.no_grad():
            noise = torch.randn_like(action) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)

            next_action = self.actor_target(next_state) + noise
            next_action = torch.max(torch.min(next_action, self.action_high), self.action_low)

            # Clipped Double Q-learning:
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            target_q = reward + not_done * discount * target_q #bellman target 

        # Update both critics
        current_q1, current_q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed actor update
        if self.total_it % policy_freq == 0:
            actor_action = self.actor(state)
            actor_loss = -self.critic.q1(state, actor_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target critic
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Soft update target actor
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval_policy(policy, env_name, seed, eval_episodes=10):
    env = gym.make(env_name)

    returns = []

    for ep in range(eval_episodes):
        state, _ = env.reset(seed=seed + 100 + ep)

        done = False
        episode_return = 0.0

        while not done:
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            episode_return += reward

        returns.append(episode_return)

    env.close()

    return float(np.mean(returns))


def train_one_run(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = gym.make(args.env)
    env.action_space.seed(args.seed)

    state, _ = env.reset(seed=args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_low = env.action_space.low
    action_high = env.action_space.high
    max_action = float(env.action_space.high[0])

    policy = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        action_low=action_low,
        action_high=action_high,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        max_size=args.buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    os.makedirs(args.results_dir, exist_ok=True)

    rows = []

    print("Environment:", args.env)
    print("Seed:", args.seed)
    print("Device:", device)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)

    avg_return = eval_policy(policy, args.env, args.seed, args.eval_episodes)
    rows.append(
        {
            "env": args.env,
            "seed": args.seed,
            "timestep": 0,
            "avg_return": avg_return,
        }
    )
    print("t=0", "avg return=", round(avg_return, 2))

    for t in range(1, args.steps + 1):
        if t < args.start_steps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state))

            # Exploration noise N(0, 0.1)
            action = action + np.random.normal(0, args.expl_noise, size=action_dim)
            action = np.clip(action, action_low, action_high)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done_for_reset = terminated or truncated

        done_for_target = float(terminated)

        replay_buffer.add(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done_for_target,
        )

        state = next_state

        if replay_buffer.size >= args.batch_size:
            policy.train(
                replay_buffer=replay_buffer,
                batch_size=args.batch_size,
                discount=args.discount,
                tau=args.tau,
                policy_noise=args.policy_noise,
                noise_clip=args.noise_clip,
                policy_freq=args.policy_freq,
            )

        if done_for_reset:
            state, _ = env.reset()

        if t % args.eval_freq == 0:
            avg_return = eval_policy(policy, args.env, args.seed, args.eval_episodes)

            rows.append(
                {
                    "env": args.env,
                    "seed": args.seed,
                    "timestep": t,
                    "avg_return": avg_return,
                }
            )

            print("t=", t, "avg return=", round(avg_return, 2))

    env.close()

    out_path = os.path.join(args.results_dir, f"{args.env}_seed{args.seed}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("saved", out_path)

    model_path = os.path.join(args.results_dir, f"{args.env}_seed{args.seed}_actor.pth")
    torch.save(policy.actor.state_dict(), model_path)
    print("saved model", model_path)


def plot_env(env_name, results_dir="results", plots_dir="plots"):
    files = [
        f for f in os.listdir(results_dir)
        if f.startswith(env_name + "_seed") and f.endswith(".csv")
    ]

    all_data = []

    for f in files:
        path = os.path.join(results_dir, f)
        all_data.append(pd.read_csv(path))

    data = pd.concat(all_data, ignore_index=True)

    stats = data.groupby("timestep")["avg_return"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0.0)

    os.makedirs(plots_dir, exist_ok=True)

    std = stats["std"]

    plt.figure()
    plt.plot(stats["timestep"], stats["mean"], label="TD3 mean")
    plt.fill_between(
        stats["timestep"],
        stats["mean"] - std,
        stats["mean"] + std,
        label="1 std",
        alpha=0.2,
    )

    plt.xlabel("Environment steps")
    plt.ylabel("Average return")
    plt.title(env_name + " TD3")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(plots_dir, f"{env_name}.png")
    plt.savefig(out_path, dpi=200)

    print("saved", out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--steps", type=int, default=1_000_000)

    parser.add_argument("--start_steps", type=int, default=1000)

    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=10)

    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expl_noise", type=float, default=0.1)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)

    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--plots_dir", type=str, default="plots")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    if args.plot:
        plot_env(args.env, args.results_dir, args.plots_dir)
    else:
        train_one_run(args)


if __name__ == "__main__":
    main()