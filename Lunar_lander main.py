import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))   

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_dims=8, fc1_dims=100, fc2_dims=100,
                 n_actions=4):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

def select_action(state, EPS):
    global steps_done
    sample = random.random()
    eps_threshold = EPS
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('LunarLander-v2')

# Training the agent 

BATCH_SIZE = 100
GAMMA = 0.999

EPS = 0.9
EPS_END = 0.03
EPS_DECAY = 0.0001

TARGET_UPDATE = 5
n_actions = 4

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

num_episodes = 500
scores = []
EPSs = []
for i_episode in range(num_episodes):

    state = env.reset()
    state = torch.tensor([state], device = device)
    score = 0 
    for t in count():

        action = select_action(state, EPS)
        next_state, reward, done, info = env.step(action.item())
        score += reward
        next_state = torch.tensor([next_state], device=device)
        reward = torch.tensor([reward], device=device)

        memory.push(state, action, next_state, reward)

        state = next_state
        
        if EPS > EPS_END:
            EPS -= EPS_DECAY

        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    print(f'Episode:{i_episode} Score:{score} EPS:{EPS}')
    scores.append(score)
    EPSs.append(EPS)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

scores_np = np.array(scores)
saved_scores = scores_np.copy()


fig, ax = plt.subplots()
ax.plot(scores_np)
ax.set_xlabel('Episode')
ax.set_ylabel('Rewards')
ax.set_title('Rewards before training')
plt.savefig('Rewards before traing.png')



# after training (removed the optimization)
num_episodes = 100
scores = []
for i_episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor([state], device = device)
    score = 0 
    for t in count():
        action = select_action(state, EPS)
        next_state, reward, done, info = env.step(action.item())
        score += reward
        next_state = torch.tensor([next_state], device=device)
        reward = torch.tensor([reward], device=device)

        memory.push(state, action, next_state, reward)

        state = next_state
    
        if done:
            episode_durations.append(t + 1)
            break
    print(f'Episode:{i_episode} Score:{score} EPS:{EPS}')
    scores.append(score)
    EPSs.append(EPS)
    
scores_np = np.array(scores)
fig, ax = plt.subplots()
ax.plot(scores_np)
ax.set_xlabel('Episode')
ax.set_ylabel('Rewards')
ax.set_title('Rewards after training')
plt.savefig('Rewards after training.png')


# 1. tunning the parameter 
# traget_update layer = 2

BATCH_SIZE = 100
GAMMA = 0.999

EPS = 0.9
EPS_END = 0.03
EPS_DECAY = 0.0001

TARGET_UPDATE = 2
n_actions = 4

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

num_episodes = 150
scores = []
EPSs = []
for i_episode in range(num_episodes):

    state = env.reset()
    state = torch.tensor([state], device = device)
    score = 0 
    for t in count():

        action = select_action(state, EPS)
        next_state, reward, done, info = env.step(action.item())
        score += reward
        next_state = torch.tensor([next_state], device=device)
        reward = torch.tensor([reward], device=device)

        memory.push(state, action, next_state, reward)

        state = next_state
        
        if EPS > EPS_END:
            EPS -= EPS_DECAY

        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    print(f'Episode:{i_episode} Score:{score} EPS:{EPS}')
    scores.append(score)
    EPSs.append(EPS)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

scores_target_update_2 = np.array(scores)
scores_target_update_10 = saved_scores[:150]


fig, ax = plt.subplots()
ax.plot(scores_target_update_2, label = 'target update: 2' )
ax.plot(scores_target_update_10, label = 'target update: 10' )
ax.set_xlabel('Episode')
ax.set_ylabel('Rewards')
ax.set_title('target update 2 vs 10')
fig.legend()
plt.savefig('target update 2 vs 10.png')

# 2. tunning the parameter 
# gamma = 0.5

BATCH_SIZE = 100
GAMMA = 0.5

EPS = 0.9
EPS_END = 0.03
EPS_DECAY = 0.0001

TARGET_UPDATE = 10
n_actions = 4

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

num_episodes = 150
scores = []
EPSs = []
for i_episode in range(num_episodes):

    state = env.reset()
    state = torch.tensor([state], device = device)
    score = 0 
    for t in count():

        action = select_action(state, EPS)
        next_state, reward, done, info = env.step(action.item())
        score += reward
        next_state = torch.tensor([next_state], device=device)
        reward = torch.tensor([reward], device=device)

        memory.push(state, action, next_state, reward)

        state = next_state
        
        if EPS > EPS_END:
            EPS -= EPS_DECAY

        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    print(f'Episode:{i_episode} Score:{score} EPS:{EPS}')
    scores.append(score)
    EPSs.append(EPS)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

scores_gamma_0_5 = np.array(scores)
scores_gamma_0_999 = saved_scores[:150]

fig, ax = plt.subplots()
ax.plot(scores_gamma_0_5, label = 'gamma: 0.5' )
ax.plot(scores_gamma_0_999, label = 'gamma: 0.999' )
ax.set_xlabel('Episode')
ax.set_ylabel('Rewards')
ax.set_title('gamma 0.5 vs 0.999')
fig.legend()
plt.savefig('gamma.png')


# 3. tunning the parameter 
# batch size = 1

BATCH_SIZE = 1
GAMMA = 0.999

EPS = 0.9
EPS_END = 0.03
EPS_DECAY = 0.0001

TARGET_UPDATE = 10
n_actions = 4

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

num_episodes = 150
scores = []
EPSs = []
for i_episode in range(num_episodes):

    state = env.reset()
    state = torch.tensor([state], device = device)
    score = 0 
    for t in count():

        action = select_action(state, EPS)
        next_state, reward, done, info = env.step(action.item())
        score += reward
        next_state = torch.tensor([next_state], device=device)
        reward = torch.tensor([reward], device=device)

        memory.push(state, action, next_state, reward)

        state = next_state
        
        if EPS > EPS_END:
            EPS -= EPS_DECAY

        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    print(f'Episode:{i_episode} Score:{score} EPS:{EPS}')
    scores.append(score)
    EPSs.append(EPS)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

scores_batch_1 = np.array(scores)
scores_batch_20 = saved_scores[:150]

fig, ax = plt.subplots()
ax.plot(scores_batch_1, label = 'batch size: 1' )
ax.plot(scores_batch_20, label = 'batch size: 20' )
ax.set_xlabel('Episode')
ax.set_ylabel('Rewards')
ax.set_title('batch size 1 vs 20')
fig.legend()
plt.savefig('batch.png')



