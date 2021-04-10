import torch
import numpy as np

import collections

from typing import Tuple

"""
    My implementation of a basic replay buffer.
    `obs_shape` has type Tuple[int, int] as `frame_stack` dictates the number of channels.
"""
class ExperienceBuffer:
    def __init__(self, 
            capacity: int, 
            obs_shape: Tuple[int, int], frame_stack: int = 4,
            unroll_steps: int = 1, gamma: float = 0.99,
            prioritized: bool = False, alpha: float = 0.6,
            beta_start: float = 0.6, beta_end: float = 1.0, beta_steps: int = 100_000,
        ):
        self.capacity = capacity

        self.obs_shape = obs_shape
        self.frame_stack = frame_stack

        self.unroll_steps = unroll_steps
        self.gamma = gamma

        self.prioritized = prioritized
        self.alpha = alpha

        self.next_idx = 0
        self.nb_samples = 0

        self.observations   = torch.empty(self.capacity, *(obs_shape), dtype=torch.uint8) # uint8
        self.actions        = torch.empty(self.capacity, dtype=torch.int64)
        self.rewards        = torch.empty(self.capacity, dtype=torch.float)
        self.dones          = torch.empty(self.capacity, dtype=torch.bool)
        if self.prioritized:
            self.priorities = torch.empty(self.capacity, dtype=torch.float)
            self.alpha = alpha

            self.beta_start, self.beta_steps, self.beta_end = beta_start, beta_steps, beta_end
            self.beta = self.beta_start

        self.pending_effect = False

    def update_beta(self, t):
        assert self.prioritized, "prioritized replay is not enabled!"
        if t < 0: return self.beta
        self.beta = min(self.beta_end, self.beta_start + t * (1.0 - self.beta_start) / self.beta_steps)
        return self.beta

    def __len__(self):
        return self.nb_samples

    def get_stacked_obs(self, idx: int):
        idx %= self.capacity
        start_idx = idx - self.frame_stack

        stacked_obs = torch.zeros(self.frame_stack, *self.obs_shape)
        stacked_obs[-1] = self.observations[idx]

        for s, i in enumerate(range(idx-1, start_idx, -1)):
            if self.dones[i] or (i % self.capacity) >= self.nb_samples:
                break
            stacked_obs[-(2+s)] = self.observations[i]

        return stacked_obs

    def store_obs(self, obs) -> int:
        assert not self.pending_effect, "Missing effect to previous observation!"
        idx = self.next_idx
        self.observations[idx] = torch.from_numpy(obs)

        self.next_idx = (self.next_idx + 1) % self.capacity
        self.nb_samples = min(self.capacity, self.nb_samples + 1)

        self.pending_effect = True
        return idx

    def store_effect(self, idx: int, action: int, reward: float, done: bool):
        assert self.pending_effect, "No observation to assign effect to!"
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        if self.prioritized:
            self.priorities[idx] = self.priorities[:self.nb_samples-1].max() if self.nb_samples - 1 > 0 else 1.0

        self.pending_effect = False

    def get_unrolled_reward(self, idx: int):
        idx %= self.capacity
        rewards = []
        for t in range(self.unroll_steps):
            t = (idx + t) % self.capacity
            rewards.append(self.rewards[t])
            if self.dones[t] or (t+1) % self.capacity >= self.nb_samples:
                break
        total_reward = torch.tensor(0.0)
        for r in rewards:
            total_reward *= self.gamma
            total_reward += r
        return total_reward
            
        # idx %= self.capacity
        # start_idx = idx - self.unroll_steps
        # total_reward = torch.tensor(0.0)
        # for i in range(idx, start_idx, -1):
            # total_reward *= self.gamma
            # total_reward += self.rewards[i]
            # if self.dones[i] or (i % self.capacity) >= self.nb_samples:
                # break
        # return total_reward

    def sample(self, batch_size: int):
        if self.prioritized:
            p = self.priorities[:self.nb_samples]
            p = p ** self.alpha
            p /= p.sum()
            idx = np.random.choice(self.nb_samples, batch_size, p=p.numpy())
            w = (self.nb_samples * p[idx]) ** (-self.beta)
            w /= w.max()
        else:
            idx = np.random.choice(self.nb_samples, batch_size, replace=False)

        obs_batch = torch.cat([self.get_stacked_obs(i).unsqueeze(0) for i in idx], dim=0)
        next_obs_batch = torch.cat([self.get_stacked_obs(i+self.unroll_steps).unsqueeze(0) for i in idx], dim=0)
        action_batch = self.actions[idx]
        reward_batch = torch.cat([self.get_unrolled_reward(i).unsqueeze(0) for i in idx], dim=0)
        
        done_batch = self.dones[idx]

        if self.prioritized:
            return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, idx, w
        else:
            return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, idx

    def update_priorities(self, idx, prio):
        assert self.prioritized, "prioritized replay is not enabled!"
        self.priorities[idx] = (prio + 1e-5).detach().cpu()

    def can_sample(self, batch_size: int):
        return self.nb_samples >= batch_size

    def clear(self):
        self.__init__(self.capacity, self.obs_shape, self.frame_stack)

"""
    Simpler example from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py
    I think I can make it faster, but I'll leave it here for reference.
"""
class ExperienceBufferSlow:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)

if __name__ == '__main__':
    import sys
    import tqdm
    import random
    memory = ExperienceBuffer(100_000, (4, 4), 4, unroll_steps=4)

    for _ in range(100):
        idx = memory.store_obs(np.random.randn(4, 4))
        memory.store_effect(idx, 0, 1.0, done=random.sample([True, False], 1)[0])

    batch = memory.sample(1)
    print(batch[2], batch[-2])
