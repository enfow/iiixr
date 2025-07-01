"""
Replay Buffers

Reference
---------
- [Prioritized Experience Replay](<https://arxiv.org/pdf/1511.05952>)
"""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Simple Replay Buffer

    Note
    ----
    - if seq_len == 1, return a batch of transitions(default)
    - if seq_len > 1, return sequences that don't cross episode boundaries
    """

    def __init__(
        self,
        capacity,
        seq_len: int = 1,
    ):
        self.buffer = deque(maxlen=capacity)
        self.seq_len: int = seq_len
        self.episode_starts = []

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if self.seq_len == 1:
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done
        else:
            return self._sample_sequences(batch_size)

    def _sample_sequences(self, batch_size):
        if len(self.buffer) < self.seq_len:
            return None

        valid_starts = []
        for i in range(len(self.buffer) - self.seq_len + 1):
            sequence_indices = set(range(i + 1, i + self.seq_len))
            if not sequence_indices.intersection(self.episode_starts):
                valid_starts.append(i)

        if len(valid_starts) < batch_size:
            selected_starts = random.choices(valid_starts, k=batch_size)
        else:
            selected_starts = random.sample(valid_starts, batch_size)

        sequences = []
        for start in selected_starts:
            sequence = list(self.buffer)[start : start + self.seq_len]
            sequences.append(sequence)

        return self._format_sequences(sequences)

    def _format_sequences(self, sequences):
        if not sequences:
            return tuple(np.array([]) for _ in range(5))

        num_sequences = len(sequences)
        seq_length = len(sequences[0])

        first_transition = sequences[0][0]
        state_shape = np.array(first_transition[0]).shape
        action_shape = np.array(first_transition[1]).shape

        # pre-allocation for better performance
        state_seqs = np.zeros((num_sequences, seq_length, *state_shape))
        action_seqs = np.zeros((num_sequences, seq_length, *action_shape))
        reward_seqs = np.zeros((num_sequences, seq_length))
        next_state_seqs = np.zeros((num_sequences, seq_length, *state_shape))
        done_seqs = np.zeros((num_sequences, seq_length), dtype=bool)

        for i, seq in enumerate(sequences):
            for j, transition in enumerate(seq):
                state_seqs[i, j] = transition[0]
                action_seqs[i, j] = transition[1]
                reward_seqs[i, j] = transition[2]
                next_state_seqs[i, j] = transition[3]
                done_seqs[i, j] = transition[4]

        return state_seqs, action_seqs, reward_seqs, next_state_seqs, done_seqs

    def __len__(self):
        return len(self.buffer)


# Deprecated
class PPOMemory:
    """
    On-Policy Memory
    """

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
    """
    SumTree is a binary tree where each leaf node stores a priority,

    Note
    ----
    - each parent node stores the sum of the priorities of its children
    - the root node stores the sum of all priorities
    - the leaf nodes store the priorities of the experiences
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float, data: object):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s: float) -> tuple[int, float, object]:
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def total(self) -> float:
        return self.tree[0]

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Replay Buffer (PER) with N-step returns.

    Note
    ----
    - the buffer stores experiences and samples them based on their TD-error
    - the buffer computes N-step returns to provide a more stable learning target
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        n_steps: int = 3,
        gamma: float = 0.99,
    ):
        # PER parameters
        # alpha: controls how much prioritization is used (0=uniform, 1=full)
        # beta_start: initial value of beta for Importance-Sampling (IS) correction
        # beta_frames: number of frames to anneal beta to 1.0
        # n_steps: the number of steps for N-step returns
        # gamma: the discount factor
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment_per_sampling = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-5

        # N-step learning parameters
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=self.n_steps)

    def _get_priority(self, error: float) -> float:
        return (np.abs(error) + self.epsilon) ** self.alpha

    def _get_n_step_info(self) -> tuple[float, np.ndarray, bool]:
        n_step_reward = 0.0
        _, _, _, n_step_next_state, n_step_done = self.n_step_buffer[-1]

        for i in range(len(self.n_step_buffer)):
            _, _, r, next_s, is_done = self.n_step_buffer[i]
            n_step_reward += (self.gamma**i) * r
            if is_done:
                n_step_next_state = next_s
                n_step_done = is_done
                break

        return n_step_reward, n_step_next_state, n_step_done

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_steps:
            return

        n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
        start_state, start_action, _, _, _ = self.n_step_buffer[0]

        data = (
            start_state,
            start_action,
            n_step_reward,
            n_step_next_state,
            n_step_done,
        )

        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, data)

    def sample(self, batch_size: int) -> tuple:
        batch = []
        idxs = []
        priorities = []

        if self.tree.size < batch_size:
            return None

        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.tree.get(s)

            retry_count = 0
            while priority == 0:
                if retry_count > 10:
                    s = random.uniform(0, self.tree.total())
                    idx, priority, data = self.tree.get(s)
                    if priority != 0:
                        break
                    else:
                        print(
                            "Warning: Failed to sample a non-zero priority item after fallback."
                        )
                        break  # Exit the while loop

                s = random.uniform(i * segment, (i + 1) * segment)
                idx, priority, data = self.tree.get(s)
                retry_count += 1

            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()

        weights = (self.tree.size * sampling_probabilities) ** -self.beta
        weights /= weights.max()

        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            idxs,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, idxs: list[int], errors: list[float]):
        """
        Updates the priorities of sampled experiences after a learning step.
        """
        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        """Returns the number of N-step transitions stored in the buffer."""
        return len(self.tree)
