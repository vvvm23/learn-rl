import torch
import numpy as np

import collections

from typing import Tuple

"""
    My implementation of a basic replay buffer.
    `obs_shape` has type Tuple[int, int] as `frame_stack` dictates the number of channels.
"""
class ExperienceBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int], frame_stack: int = 4):
        self.obs_shape = obs_shape
        self.frame_stack = frame_stack
        self.capacity = capacity

        self.next_idx = 0
        self.nb_samples = 0

        self.observations   = torch.empty(self.capacity, *(obs_shape), dtype=torch.uint8)
        self.actions        = torch.empty(self.capacity, dtype=torch.uint8)
        self.rewards        = torch.empty(self.capacity, dtype=torch.float)
        self.dones          = torch.empty(self.capacity, dtype=torch.bool)

        self.pending_effect = False

    def __len__(self):
        return self.nb_samples

    def get_stacked_obs(self, idx: int):
        idx %= self.capacity
        start_idx = (idx - self.frame_stack)

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

        self.pending_effect = False

    def sample(self, batch_size: int):
        idx = np.random.choice(self.nb_samples, batch_size, replace=False)

        obs_batch = torch.cat([self.get_stacked_obs(i).unsqueeze(0) for i in idx], dim=0)
        next_obs_batch = torch.cat([self.get_stacked_obs(i+1).unsqueeze(0) for i in idx], dim=0)
        action_batch = self.actions[idx]
        reward_batch = self.rewards[idx]
        done_batch = self.dones[idx]

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, idx

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
    memory = ExperienceBuffer(100_000, (4, 4), 4)

    # for i in tqdm.tqdm(range(200_000)):
        # idx = memory.store_obs(np.random.randn(4, 4))
        # memory.store_effect(idx, 0, 0., random.choice([True, False]))

        # if memory.can_sample(64):
            # memory.sample(64)
    for i in range(10):
        idx = memory.store_obs(np.random.randn(4, 4))
        memory.store_effect(idx, 0, 0., random.choice([True, False]))

    print(memory.get_stacked_obs(0))
    print(memory.dones[:10])
