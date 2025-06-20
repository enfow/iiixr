import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.logprobs, self.rewards, self.dones = (
            [],
            [],
            [],
            [],
            [],
        )

    def __len__(self):
        return len(self.states)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment_per_sampling = (1.0 - beta_start) / beta_frames
        self.tree = SumTree(capacity)
        self.epsilon = 1e-5

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = 1.0
        data = (state, action, reward, next_state, done)
        self.tree.add(max_priority, data)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self.tree.size * sampling_probabilities) ** -self.beta
        weights /= weights.max()

        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones, idxs, weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)
