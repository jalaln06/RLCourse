import random

import torch
from torch import nn
import numpy as np
import gym
import matplotlib.pyplot as plt

class CEMContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 10),
            nn.ReLU(),
            nn.Linear(10, self.action_dim)
        )
        self.loss = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state, noise_std=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state).squeeze().detach().numpy()
        #print(action)
        noise =  noise_std * (2*np.random.sample() - 1)*5
        action += noise
        #print(action)
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
        return action


    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for elite_trajectory in elite_trajectories:
            for state, action in zip(elite_trajectory['states'], elite_trajectory['actions']):
                elite_states.append(state)
                elite_actions.append(action)

        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.FloatTensor(np.array(elite_actions))
        predict_actions = self.forward(elite_states)
        loss = self.loss(predict_actions.squeeze(), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

def get_trajectory(env, agent, trajectory_len, noise_std=0.1, visualize=False):
    trajectory = {'states': [], 'actions': [], 'total_reward': 0}

    state = env.reset()
    trajectory['states'].append(state)

    for _ in range(trajectory_len):
        action = agent.get_action(state, noise_std)
        trajectory['actions'].append(action)

        state, reward, done, _ = env.step([action])
        trajectory['total_reward'] += reward
        if done:
            break

        if visualize:
            env.render()

        trajectory['states'].append(state)

    return trajectory

elite_container = []
def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories if trajectory['total_reward'] > 0]
    if not total_rewards:
        return []

    print(total_rewards)

    quantile = np.quantile(total_rewards, q=q_param)
    elite_trajectories_current = [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]

    elite_container.extend(elite_trajectories_current)

    num_samples = min(5, len(elite_container))
    elite_samples = random.sample(elite_container, num_samples)
    elite_trajectories_current.extend(elite_samples)

    return elite_trajectories_current
env = gym.make('MountainCarContinuous-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = CEMContinuous(state_dim, action_dim)
start_sigma = 1.5
end_sigma = 0.1
episode_n = 200
total_episodes = episode_n
trajectory_n = 100
trajectory_len = 999
q_param = 0.9
start_q = 0.1
end_q = 0.8
scores = []
for episode in range(episode_n):
    sigma = start_sigma - episode * (start_sigma - end_sigma) / total_episodes
    trajectories = [get_trajectory(env, agent, trajectory_len, sigma) for _ in range(trajectory_n)]

    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')
    scores.append(mean_total_reward)
    q_param = start_q - episode * (start_q - end_q) / total_episodes
    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)


#very low noise
for episode in range(episode_n):
    sigma = 0.05
    trajectories = [get_trajectory(env, agent, trajectory_len, sigma) for _ in range(trajectory_n)]

    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')
    scores.append(mean_total_reward)
    q_param = start_q - episode * (start_q - end_q) / total_episodes
    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    print(len(elite_trajectories))
    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()