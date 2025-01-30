import numpy as np
import random
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).to(device).unsqueeze(1),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).to(device).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

class ReplayMemoryGPU:
    def __init__(self, cfg, device):
        self.capacity = cfg.replay_size
        self.device = device
        self.dim_state = cfg.dim_state  # 128
        self.dim_action = 3  # [accident_score, x, y]
        self.dim_mem = self.dim_state + self.dim_action + 1 + self.dim_state + 1  # state + action + reward + next_state + done
        self.buffer = torch.zeros((self.capacity, self.dim_mem), device=device)
        self.position = 0
        self.length = 0

    def push(self, state, action, reward, next_state, done):
        transition = torch.cat([state, action, reward, next_state, done], dim=-1)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.length, (batch_size,), device=self.device)
        batch = self.buffer[indices]
        state = batch[:, :self.dim_state]
        action = batch[:, self.dim_state:self.dim_state+self.dim_action]
        reward = batch[:, self.dim_state+self.dim_action:self.dim_state+self.dim_action+1]
        next_state = batch[:, self.dim_state+self.dim_action+1:self.dim_state*2+self.dim_action+1]
        done = batch[:, -1:]
        return state, action, reward, next_state, done

    def __len__(self):
        return self.length