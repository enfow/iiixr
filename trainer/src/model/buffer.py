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
    """
    A SumTree data structure used for prioritized experience replay.
    This implementation is a binary tree where each leaf node stores a priority,
    and each parent node stores the sum of the priorities of its children.
    This allows for efficient sampling of experiences based on their priorities.
    """

    def __init__(self, capacity: int):
        """
        Initializes the SumTree.

        Args:
            capacity (int): The maximum number of items that can be stored in the tree.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """
        Propagates a change in priority up the tree from the given index.

        Args:
            idx (int): The index of the node where the change started.
            change (float): The change in priority to be propagated.
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float):
        """
        Updates the priority of an item at a given index in the tree.

        Args:
            idx (int): The tree index of the item to update.
            priority (float): The new priority value.
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float, data: object):
        """
        Adds a new item with a given priority to the tree.

        Args:
            priority (float): The priority of the new item.
            data (object): The data (experience) to store.
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, idx: int, s: float) -> int:
        """
        Recursively retrieves the index of an item for a given sample value.

        Args:
            idx (int): The current node index in the tree.
            s (float): The sample value (a random number between 0 and total priority).

        Returns:
            int: The tree index of the sampled item.
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s: float) -> tuple[int, float, object]:
        """
        Gets an item from the tree based on a sample value.

        Args:
            s (float): The sample value.

        Returns:
            A tuple containing the tree index, priority, and data of the sampled item.
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def total(self) -> float:
        """Returns the total priority of all items in the tree."""
        return self.tree[0]

    def __len__(self) -> int:
        """Returns the current number of items in the tree."""
        return self.size


class PrioritizedReplayBuffer:
    """
    A Prioritized Replay Buffer (PER) with N-step returns.
    This buffer stores experiences and samples them based on their TD-error (priority),
    which makes learning more efficient. It also computes N-step returns to provide a
    more stable learning target.
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
        """
        Initializes the PrioritizedReplayBuffer.

        Args:
            capacity (int): Maximum number of experiences to store.
            alpha (float): Controls how much prioritization is used (0=uniform, 1=full).
            beta_start (float): Initial value of beta for Importance-Sampling (IS) correction.
            beta_frames (int): Number of frames to anneal beta to 1.0.
            n_steps (int): The number of steps for N-step returns.
            gamma (float): The discount factor.
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment_per_sampling = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-5  # Small constant to ensure non-zero priority

        # N-step learning parameters
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=self.n_steps)

    def _get_priority(self, error: float) -> float:
        """Calculates priority from a TD error."""
        return (np.abs(error) + self.epsilon) ** self.alpha

    def _get_n_step_info(self) -> tuple[float, np.ndarray, bool]:
        """
        Calculates the N-step return, the final next_state, and the done flag
        from the current contents of the n-step buffer.
        """
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
        """
        Adds a new experience to the buffer and processes the n-step returns.
        """
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
        """
        Samples a batch of experiences from the buffer.

        Returns:
            A tuple containing states, actions, rewards, next_states, dones,
            tree indices, and importance-sampling weights.
        """
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.tree.get(s)

            if priority == 0:
                s = random.uniform(i * segment, (i + 1) * segment)
                idx, priority, data = self.tree.get(s)

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
