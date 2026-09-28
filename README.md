# Comparative Study of Deep Reinforcement Learning Algorithms

**EPFL — Deep Reinforcement Learning Project**

This project implements and compares four deep reinforcement learning algorithms **from scratch**:

- **DQN**
- **PPO**
- **SAC**
- **TD3**

They are evaluated on discrete and continuous Gym control tasks, including **CartPole, Acrobot, Pendulum, and MountainCarContinuous**. The comparison focuses on learning stability, sample efficiency, exploration, hyperparameter sensitivity, and suitability for different action spaces. :chatgpt-content-reference{index="0"}

## Algorithms

- **DQN** — value-based learning for discrete action spaces
- **PPO** — on-policy actor-critic with clipped policy updates
- **SAC** — off-policy stochastic actor-critic with entropy-based exploration
- **TD3** — deterministic actor-critic with twin critics and delayed policy updates

## Results

- PPO showed strong and stable performance on the discrete-control tasks.
- SAC achieved the strongest continuous-control performance on Pendulum.
- TD3 performed well on dense continuous-control tasks.
- MountainCarContinuous highlighted the importance of exploration and reward design for sparse-reward problems.

  
## Repository Structure

```text
deep-rl-algorithm-comparison/
├── DQN_and_PPO/          # DQN implementation and DQN/PPO experiments
├── PPO/                  # PPO implementation and evaluation
├── SAC/                  # SAC implementation and experiments
├── td3_rl/               # TD3 implementation, plots, models, and results
├── Deep_RL_Algorithms_Poster.pdf
├── Report.pdf
└── README.md
```

## Team

- Andrea Tarabay
- Joelle Alachkar
- Omar Shibli

## Report & Poster

📄 [Full Project Report](Report.pdf)  
📊 [Project Poster](Deep_RL_Algorithms_Poster.pdf)
