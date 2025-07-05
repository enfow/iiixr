"""
Replay Buffers

Reference
---------
- [Prioritized Experience Replay](<https://arxiv.org/pdf/1511.05952>)
"""

import random
from collections import deque
from typing import List, Set, Tuple

import numpy as np

from schema.config import BaseConfig, BufferType


class AbstractBuffer:
    def push(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def sample(self, batch_size: int):
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError


class ReplayBuffer(AbstractBuffer):
    is_per = False
    is_sequential = False

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SeqReplayBuffer(AbstractBuffer):
    is_per = False
    is_sequential = True

    def __init__(self, capacity: int, seq_len: int, seq_stride: int = 1):
        self.buffer = deque(maxlen=capacity)
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.capacity = capacity

        self.episode_starts: List[int] = []
        self.min_seq_span = (self.seq_len - 1) * self.seq_stride + 1

    def push(self, state, action, reward, next_state, done: bool):
        # If the buffer is full, the oldest element is about to be dropped.
        # We must decrement all episode start indices to reflect this shift.
        if len(self.buffer) == self.capacity:
            if self.episode_starts and self.episode_starts[0] == 0:
                self.episode_starts.pop(0)
            self.episode_starts = [i - 1 for i in self.episode_starts]

        current_index = len(self.buffer)
        self.buffer.append((state, action, reward, next_state, done))

        if done:
            # The next transition will be the start of a new episode.
            self.episode_starts.append(current_index + 1)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if len(self.buffer) < self.min_seq_span:
            return self._return_for_invalids()

        valid_starts = self._get_valid_starts()

        if not valid_starts:
            return self._return_for_invalids()

        # Sample with replacement if not enough valid starts, otherwise sample without.
        if len(valid_starts) < batch_size:
            selected_starts = random.choices(valid_starts, k=batch_size)
        else:
            selected_starts = random.sample(valid_starts, batch_size)

        sequences = []
        for start_idx in selected_starts:
            end_idx = start_idx + self.min_seq_span
            sequence = [
                self.buffer[i] for i in range(start_idx, end_idx, self.seq_stride)
            ]
            sequences.append(sequence)
        batch = [list(zip(*seq)) for seq in sequences]

        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack into final numpy arrays.
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool),
        )

    def _get_valid_starts(self) -> List[int]:
        all_possible_starts = set(range(len(self.buffer) - self.min_seq_span + 1))

        invalid_starts: Set[int] = set()
        for episode_start_idx in self.episode_starts:
            start_of_invalid_range = episode_start_idx - (self.min_seq_span - 1)
            end_of_invalid_range = episode_start_idx

            for i in range(start_of_invalid_range, end_of_invalid_range):
                if i >= 0:
                    invalid_starts.add(i)

        valid_starts = list(all_possible_starts - invalid_starts)
        return valid_starts

    def _return_for_invalids(self) -> Tuple[np.ndarray, ...]:
        if self.buffer:
            s_shape = (0, self.seq_len, *np.array(self.buffer[0][0]).shape)
            a_shape = (0, self.seq_len, *np.array(self.buffer[0][1]).shape)
        else:  # Fallback if buffer is also empty.
            s_shape, a_shape = (0, self.seq_len, 0), (0, self.seq_len, 0)

        return (
            np.empty(s_shape, dtype=np.float32),
            np.empty(a_shape, dtype=np.float32),
            np.empty((0, self.seq_len), dtype=np.float32),
            np.empty(s_shape, dtype=np.float32),
            np.empty((0, self.seq_len), dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def size(self) -> int:
        return len(self.buffer)


# Deprecated
class PPOMemory(AbstractBuffer):
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


class SumTree(AbstractBuffer):
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
        self._size = 0

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
        self._size = min(self._size + 1, self.capacity)

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
        return self._size


class PrioritizedReplayBuffer(AbstractBuffer):
    """
    Prioritized Replay Buffer (PER) with N-step returns.

    Note
    ----
    - the buffer stores experiences and samples them based on their TD-error
    - the buffer computes N-step returns to provide a more stable learning target

    Parameters
    ----------
    capacity: int
        The maximum number of transitions to store in the buffer.
    alpha: float
        The priority exponent.
    beta_start: float
        The initial value of beta for Importance-Sampling (IS) correction.
    beta_frames: int
        The number of frames to anneal beta to 1.0.
    n_steps: int
        The number of steps for N-step returns.
    gamma: float
        The discount factor.
    """

    is_per = True
    is_sequential = False

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        n_steps: int = 3,
        gamma: float = 0.99,
    ):
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

        if self.tree._size < batch_size:
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

        weights = (self.tree._size * sampling_probabilities) ** -self.beta
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

    @property
    def size(self) -> int:
        return self.tree._size


class SequentialPrioritizedReplayBuffer(AbstractBuffer):
    is_per = True
    is_sequential = True

    def __init__(
        self,
        capacity: int,
        seq_len: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        seq_stride: int = 1,
    ):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.min_seq_span = (self.seq_len - 1) * self.seq_stride + 1
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment_per_sampling = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-5
        self.episode_starts: List[int] = [0]
        self._size = 0

    def push(self, state, action, reward, next_state, done: bool) -> None:
        if self._size == self.capacity:
            # Adjust episode start indices due to the oldest element being dropped.
            if self.episode_starts and self.episode_starts[0] == 0:
                self.episode_starts.pop(0)
            self.episode_starts = [i - 1 for i in self.episode_starts]

        current_index = self._size if self._size < self.capacity else self.capacity - 1
        self.buffer.append((state, action, reward, next_state, done))
        if self._size < self.capacity:
            self._size += 1

        if done:
            self.episode_starts.append(current_index + 1)

        # New transitions are given max priority to encourage exploration.
        max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, None)  # Data is not stored in the tree

    def sample(
        self, batch_size: int
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, np.ndarray
    ]:
        """
        Samples a batch of sequences from the buffer using prioritized sampling.

        Returns:
            A tuple containing stacked sequences of states, actions, rewards,
            next_states, dones, the indices of the sampled transitions in the SumTree,
            and the importance sampling weights.
        """
        if self._size < self.min_seq_span:
            return self._return_for_invalids()

        valid_starts = self._get_valid_starts()
        if not valid_starts:
            return self._return_for_invalids()

        temp_tree_map = {
            start: self.tree.tree[start + self.tree.capacity - 1]
            for start in valid_starts
        }
        valid_priorities = np.array(list(temp_tree_map.values()))
        valid_total_priority = np.sum(valid_priorities)

        if valid_total_priority == 0:
            return self._return_for_invalids()

        segment = valid_total_priority / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        sequences = []
        idxs = []
        priorities = []
        selected_starts = []

        for i in range(batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            (idx, priority, start_idx) = self._get_from_valid(
                s, valid_starts, temp_tree_map
            )

            priorities.append(priority)
            idxs.append(idx)
            selected_starts.append(start_idx)

        for start_idx in selected_starts:
            end_idx = start_idx + self.min_seq_span
            sequence = [
                self.buffer[i] for i in range(start_idx, end_idx, self.seq_stride)
            ]
            sequences.append(sequence)

        batch = [list(zip(*seq)) for seq in sequences]
        states, actions, rewards, next_states, dones = zip(*batch)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self._size * sampling_probabilities) ** -self.beta
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=bool),
            idxs,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, idxs: list[int], errors: np.ndarray) -> None:
        """
        Updates the priorities of the sampled sequences.

        The priority of a sequence is determined by the TD error of its first transition.
        """
        priorities = (np.abs(errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)

    def _get_valid_starts(self) -> List[int]:
        """
        Computes the set of valid starting indices for sequences.

        A start index is valid if the entire sequence it initiates falls within the
        bounds of the replay buffer and does not cross an episode boundary.
        """
        all_possible_starts = set(range(self._size - self.min_seq_span + 1))
        invalid_starts: Set[int] = set()
        for episode_start_idx in self.episode_starts:
            start_of_invalid_range = episode_start_idx - self.min_seq_span + 1
            end_of_invalid_range = episode_start_idx
            for i in range(start_of_invalid_range, end_of_invalid_range):
                if i >= 0:
                    invalid_starts.add(i)
        return list(all_possible_starts - invalid_starts)

    def _get_from_valid(
        self, s: float, valid_starts: List[int], temp_tree_map: dict
    ) -> tuple[int, float, int]:
        """
        Retrieves a sample from the set of valid start indices.
        """
        cumulative_priority = 0.0
        for start_idx in valid_starts:
            priority = temp_tree_map[start_idx]
            cumulative_priority += priority
            if cumulative_priority > s:
                tree_idx = start_idx + self.tree.capacity - 1
                return tree_idx, priority, start_idx
        last_start_idx = valid_starts[-1]
        tree_idx = last_start_idx + self.tree.capacity - 1
        priority = temp_tree_map[last_start_idx]
        return tree_idx, priority, last_start_idx

    def _return_for_invalids(self) -> Tuple[np.ndarray, ...]:
        """
        Returns empty numpy arrays with the correct shapes when sampling is not possible.
        """
        s_shape, a_shape = (0, self.seq_len, 0), (0, self.seq_len, 0)
        if self.buffer:
            s_shape = (0, self.seq_len, *np.array(self.buffer[0][0]).shape)
            a_shape = (0, self.seq_len, *np.array(self.buffer[0][1]).shape)

        return (
            np.empty(s_shape, dtype=np.float32),
            np.empty(a_shape, dtype=np.int64),
            np.empty((0, self.seq_len), dtype=np.float32),
            np.empty(s_shape, dtype=np.float32),
            np.empty((0, self.seq_len), dtype=bool),
            [],
            np.empty((0,), dtype=np.float32),
        )

    def __len__(self) -> int:
        """Returns the current number of transitions in the buffer."""
        return self._size

    @property
    def size(self) -> int:
        return self._size


class ReployBufferFactory:
    def __new__(cls, config: BaseConfig):
        if config.model.model in ["ppo_seq", "ppo"]:
            # ppo use their own memory
            return None

        # validation check
        if (
            config.buffer.buffer_type
            in [BufferType.SEQUENTIAL, BufferType.PER_SEQUENTIAL]
            and config.model.seq_len == 1
        ):
            raise ValueError(
                f"seq_len must be greater than 1 for sequential buffer, but got {config.model.seq_len}"
            )
        if (
            config.buffer.buffer_type in [BufferType.DEFAULT, BufferType.PER]
            and config.model.seq_len > 1
        ):
            raise ValueError(
                f"seq_len must be 1 for non-sequential buffer, but got {config.model.seq_len}"
            )

        if config.buffer.buffer_type == BufferType.DEFAULT:
            print(f"Using ReplayBuffer | size: {config.buffer.buffer_size}")
            return ReplayBuffer(config.buffer.buffer_size)

        elif config.buffer.buffer_type == BufferType.SEQUENTIAL:
            print(
                f"Using SeqReplayBuffer | size: {config.buffer.buffer_size} ",
                f"| seq_len: {config.model.seq_len} ",
                f"| seq_stride: {config.model.seq_stride}",
            )
            return SeqReplayBuffer(
                config.buffer.buffer_size,
                config.model.seq_len,
                config.model.seq_stride,
            )

        elif config.buffer.buffer_type == BufferType.PER:
            print(
                f"Using PrioritizedReplayBuffer | size: {config.buffer.buffer_size} ",
                f"| n_steps: {config.buffer.per_n_steps}",
            )
            return PrioritizedReplayBuffer(
                capacity=config.buffer.buffer_size,
                alpha=config.buffer.alpha,
                beta_start=config.buffer.beta_start,
                beta_frames=config.buffer.beta_frames,
                n_steps=config.buffer.per_n_steps,
                gamma=config.gamma,
            )
        elif config.buffer.buffer_type == BufferType.PER_SEQUENTIAL:
            print(
                f"Using SequentialPrioritizedReplayBuffer | size: {config.buffer.buffer_size} ",
                f"| seq_len: {config.model.seq_len} ",
                f"| seq_stride: {config.model.seq_stride}",
            )
            return SequentialPrioritizedReplayBuffer(
                capacity=config.buffer.buffer_size,
                seq_len=config.model.seq_len,
                seq_stride=config.model.seq_stride,
                alpha=config.buffer.alpha,
                beta_start=config.buffer.beta_start,
                beta_frames=config.buffer.beta_frames,
            )
        else:
            raise ValueError(f"Invalid buffer type: {config.buffer.buffer_type}")
