import numpy as np
import random
import pickle
import os
import sys


# class TrajectoryBuffer(object):
#     def __init__(self, size, save_dir="./data/trajectories/", load_dir="./data/trajectories/", file_name="unknown"):
#         """Create Prioritized Replay buffer.
#
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         """
#
#         self._storage = []
#         self._maxsize = int(size)
#         self._next_idx = 0
#
#         self.set_file(save_dir, load_dir, file_name)
#
#     def __len__(self):
#         return len(self._storage)
#
#     def set_file(self, save_dir="./data/trajectories/", load_dir="./data/trajectories/", file_name="unknown"):
#         self.save_file = os.path.join(save_dir, file_name + ".pkl")
#         self.load_file = os.path.join(load_dir, file_name + ".pkl")
#
#     def clear(self):
#         self._storage = []
#         self._next_idx = 0
#
#     def add(self, ob, action):
#         data = (ob, action)
#
#         if self._next_idx >= len(self._storage):
#             self._storage.append(data)
#         else:
#             self._storage[self._next_idx] = data
#         self._next_idx = (self._next_idx + 1) % self._maxsize
#
#     def _encode_sample(self, idxes):
#         obs, actions = [], []
#         for i in idxes:
#             data = self._storage[i]
#             ob, action = data
#             obs.append(np.array(ob, copy=False))
#             actions.append(np.array(action, copy=False))
#         return np.array(obs), np.array(actions)
#
#     def make_index(self, batch_size):
#         return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
#
#     def make_latest_index(self, batch_size):
#         idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
#         np.random.shuffle(idx)
#         return idx
#
#     def sample_index(self, idxes):
#         return self._encode_sample(idxes)
#
#     def sample(self, batch_size):
#         # Sample a batch of trajectories.
#         if batch_size > 0:
#             idxes = self.make_index(batch_size)
#         else:
#             idxes = range(0, len(self._storage))
#         return self._encode_sample(idxes)
#
#     def collect(self):
#         return self.sample(-1)
#
#     def save_to_file(self):
#         data = [self._storage, self._next_idx, self._maxsize]
#         # if os.path.exists(self.save_file):
#         #     os.rmdir(self.save_file)
#         output = open(self.save_file, 'wb')
#         pickle.dump(data, output, -1)
#
#
#
#     def load_from_file(self):
#         pkl_file = open(self.load_file, 'rb')
#         data = pickle.load(pkl_file)
#         self._storage, self._next_idx, self._maxsize = data



class TrajectoryBuffer(object):
    def __init__(self, size, save_dir="./data/trajectories/", load_dir="./data/trajectories/", file_name="unknown"):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """

        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

        self.set_file(save_dir, load_dir, file_name)

    def __len__(self):
        return len(self._storage)

    def set_file(self, save_dir="./data/trajectories/", load_dir="./data/trajectories/", file_name="unknown"):
        self.save_file = os.path.join(save_dir, file_name + ".pkl")
        self.load_file = os.path.join(load_dir, file_name + ".pkl")

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, ob, action, reward):
        data = (ob, action, reward)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obs, actions, rewards = [], [], []
        for i in idxes:
            data = self._storage[i]
            ob, action, reward = data
            obs.append(np.array(ob, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
        return np.array(obs), np.array(actions), np.array(rewards)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        # Sample a batch of trajectories.
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)

    def save_to_file(self):
        data = [self._storage, self._next_idx, self._maxsize]
        # if os.path.exists(self.save_file):
        #     os.rmdir(self.save_file)
        output = open(self.save_file, 'wb')
        pickle.dump(data, output, -1)

    def load_from_file(self):
        pkl_file = open(self.load_file, 'rb')
        data = pickle.load(pkl_file)
        self._storage, self._next_idx, self._maxsize = data