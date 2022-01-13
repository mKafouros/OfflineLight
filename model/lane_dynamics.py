import random
import numpy as np
from collections import deque
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import os
from sklearn.model_selection import train_test_split
import pickle

RANDOM_SEED=42

def get_inside_roads_list(n, m):
    target_road_list = []
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if not i == 1:
                target_road_list.append("road_{}_{}_{}".format(i, j, 2))
            if not i == n:
                target_road_list.append("road_{}_{}_{}".format(i, j, 0))
            if not j == 1:
                target_road_list.append("road_{}_{}_{}".format(i, j, 3))
            if not j == m:
                target_road_list.append("road_{}_{}_{}".format(i, j, 1))
    return target_road_list
    # return ["road_1_1_0"]

def name_to_id(name):
    if len(name.split("_")) > 6: # is a lane link
        return list(map(int, name.split("_")[1:5]))


    return list(map(int, name.split("_")[1:]))


def get_raodnet_size(agents):
    id1 = []
    id2 = []
    for agent in agents:
        id1.append(name_to_id(agent.I.id)[0])
        id2.append(name_to_id(agent.I.id)[1])
    return max(id1), max(id2)

def get_prev_roads(lane_name):
    iid = name_to_id(lane_name)[0:2]
    direc = name_to_id(lane_name)[2]
    prev_roads = []
    if direc != 0:
        prev_roads.append("road_{}_{}_{}".format(iid[0] + 1, iid[1], 2))
    if direc != 1:
        prev_roads.append("road_{}_{}_{}".format(iid[0], iid[1] + 1, 3))
    if direc != 2:
        prev_roads.append("road_{}_{}_{}".format(iid[0] - 1, iid[1], 0))
    if direc != 3:
        prev_roads.append("road_{}_{}_{}".format(iid[0], iid[1] - 1, 1))

    return prev_roads

def id_to_road(road_id):
    return "road_{}_{}_{}".format(road_id[0], road_id[1], road_id[2])

def id_to_lane(lane_id):
    return "road_{}_{}_{}_{}".format(lane_id[0], lane_id[1], lane_id[2], lane_id[3])

def lane_to_road(lane_name):
    road_id = name_to_id(lane_name)[0:3]
    return "road_{}_{}_{}".format(road_id[0], road_id[1], road_id[2])

def get_prev_lanes(lane_name):
    lane_id = name_to_id(lane_name)
    current_road_name = id_to_road(lane_id[0:3])
    prev_roads = get_prev_roads(lane_name)
    prev_lanes = []
    for road in prev_roads:
        direc = get_lane_direc(road, current_road_name)
        prev_lanes.append("{}_{}".format(road, direc))
    return prev_lanes


def if_road_connected(road1, road2):
    iid = name_to_id(road1)[0:2]
    iid2 = name_to_id(road2)[0:2]
    direc = name_to_id(road1)[2]
    if direc == 0:
        iid[0] += 1
    elif direc == 1:
        iid[1] += 1
    elif direc == 2:
        iid[0] -= 1
    elif direc == 3:
        iid[1] -= 1
    else:
        raise Exception

    return iid == iid2


def get_lane_direc(road1, road2):
    assert if_road_connected(road1, road2)
    direc1 = name_to_id(road1)[2]
    direc2 = name_to_id(road2)[2]
    if direc1 == direc2:
        return 1 # straight
    elif direc2 - direc1 == -1 or direc2 - direc1 == 3:
        return 2 # turn right
    elif direc2 - direc1 == 1 or direc2 - direc1 == -3:
        return 0 # turn left
    else:
        raise Exception


def get_target_lane_idx(current_dirc, target_dirc):
    # print(current_dirc, target_dirc)
    if current_dirc == target_dirc:
        target_lane_idx = 1  # go straight
    elif current_dirc - target_dirc == 1 or current_dirc - target_dirc == -3:
        target_lane_idx = 2  # turn right
    elif current_dirc - target_dirc == -1 or current_dirc - target_dirc == 3:
        target_lane_idx = 0  # turn left
    else:
        raise ValueError
    return target_lane_idx

def flat(arr):
    result = []
    for i in arr:
        for j in i:
            result.append(j)
    return result

class OracleModel(object):
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.eng = self.env.eng
        self.data_center = ModelDataCenter()

        self.alert_lable = False

    def get_obs(self, archive):
        backup = self.env.take_snapshot()
        self.env.load_snapshot(archive)
        obs = self.env.get_current_obs()
        self.env.load_snapshot(backup)
        return obs

    def start_rollout(self, archive):
        self.backup = self.env.take_snapshot()
        self.env.load_snapshot(archive)
        obs = self.env.get_current_obs()
        return obs

    def next_rollout(self, actions, action_interval=20, get_travel_time=False):
        obs, rewards = None, None
        for i in range(action_interval):
            obs, rewards, _, _ = self.env.step(actions)

        if get_travel_time:
            travel_time = self.eng.get_average_travel_time() * (-1.)
            rewards = [travel_time for i in range(len(self.agents))]

        return obs, rewards

    def finish_rollout(self):
        self.env.load_snapshot(self.backup)
        del self.backup[0]

    def predict(self, archive, actions, action_interval=20, get_travel_time=False):
        backup = self.env.take_snapshot()
        self.env.load_snapshot(archive)

        obs, rewards = None, None
        if not self.alert_lable:
            self.alert_lable = True
            print("Info: action interval in oracle model is {}".format(action_interval))

        for i in range(action_interval):
            obs, rewards, _, _ = self.env.step(actions)

        snapshot = self.env.take_snapshot()

        if get_travel_time:
            travel_time = self.eng.get_average_travel_time() * (-1.)
            rewards = [travel_time for i in range(len(self.agents))]

        self.env.load_snapshot(backup)

        return snapshot, rewards

class DynamicModel(object):
    def __init__(self, env):
        self.env = env
        self.ob_model = PassedVehicleModel(env)
        self.rw_model = WaitingVehicleModel(env)

    def get_obs(self, obs):
        return obs


    def predict(self, obs, actions):
        _next_ob = self.ob_model.predict(obs, actions)
        _next_rw = self.rw_model.predict(obs, actions)

        # print(len(_next_ob), len(_next_rw))

        return self.get_ob(_next_ob, _next_rw), self.get_rw(_next_rw)

    def train(self, verbose=False, epochs=200):
        print("training ob model...")
        self.ob_model.train(verbose=verbose, epochs=epochs)
        print("training rw model...")
        self.rw_model.train(verbose=verbose, epochs=epochs)

    def update_target_lanes(self):
        self.rw_model.update_target_lanes()
        self.ob_model.update_target_lanes()

    def collect(self, obs, actions, next_obs):
        self.ob_model.collect(obs, actions, next_obs, use_lane_target_stats=True)
        self.rw_model.collect(obs, actions, next_obs, use_lane_target_stats=True)

    def start_rollout(self, obs):
        self.current_obs = obs
        return obs

    def next_rollout(self, actions, action_interval=None, get_travel_time=False):
        obs, rewards = self.predict(self.current_obs, actions)
        self.current_obs = obs
        return obs, rewards

    def finish_rollout(self):
        pass


    def get_ob(self, lane_vehicles, lane_waiting_vehicles, including_waiting=True):
        if not including_waiting:
            return lane_vehicles
        obs = []
        # print(lane_vehicles.shape)
        # print(lane_vehicles)
        # print(lane_waiting_vehicles.shape)
        # print(lane_waiting_vehicles)
        for i in range(len(lane_vehicles)):
            obs.append(np.asarray(list(lane_vehicles[i]) + list(lane_waiting_vehicles[i])))
        # print(len(obs), len(obs[0]))
        return obs

    def get_rw(self, waiting_vehicles, reward_type="queue_length"):
        if reward_type == "queue_length":
            rewards = []
            # print(waiting_vehicles)
            for i in waiting_vehicles:
                rewards.append(sum(i) / (len(waiting_vehicles) + 8))
            return rewards
        else:
            #TODO pressure
            raise NotImplementedError

    def save(self, name):
        self.rw_model.save(name=name)
        self.ob_model.save(name=name)

    def load(self, name):
        self.rw_model.load(name=name)
        self.ob_model.load(name=name)


class WaitingVehicleModel(object):
    def __init__(self, env):
        self.env = env
        self.agents = env.agents
        self.data_center = ModelDataCenter()

        self.nn = NN(15)

        self.lane_target_counter = LaneVehicleTargetCounter(env.world.eng, env.agents)
        m, n = get_raodnet_size(env.agents)
        self.target_roads = get_inside_roads_list(m, n)

    def lane_phase_available(self, lane, phase):
        lane_id = name_to_id(lane)
        if lane_id[-1] == 2:
            return 1
        if phase == 0:
            return 0
        agent = self.agents[self.lane_target_agent(lane)]
        available_startlanes = agent.I.phase_available_startlanes[phase]
        if lane in available_startlanes:
            return 1
        else:
            return 0

    def lane_target_agent(self, lane):
        iid = name_to_id(lane)[0:2]
        direc = name_to_id(lane)[2]
        if direc == 0:
            iid[0] += 1
        elif direc == 1:
            iid[1] += 1
        elif direc == 2:
            iid[0] -= 1
        elif direc == 3:
            iid[1] -= 1
        else:
            raise Exception
        intersection_name = "intersection_{}_{}".format(iid[0], iid[1])
        for agent_id, agent in enumerate(self.agents):
            if intersection_name == agent.I.id:
                return agent_id
        raise Exception


    def get_lane_vehicles(self, lane_name, obs):
        agent_id = self.lane_target_agent(lane_name)
        lane_id = flat(self.agents[agent_id].ob_generator.lanes).index(lane_name)
        return obs[agent_id][lane_id]

    def get_passed_vehicles(self, lane_name, obs):
        agent_id = self.lane_target_agent(lane_name)
        lane_id = flat(self.agents[agent_id].ob_generator.lanes).index(lane_name)
        return obs[agent_id][lane_id + self.length_per_info]

    def get_waiting_vehicles(self, lane_name, lane_waiting_vehicles):
        agent_id = self.lane_target_agent(lane_name)
        lane_id = flat(self.agents[agent_id].ob_generator.lanes).index(lane_name)
        return lane_waiting_vehicles[agent_id][lane_id + self.length_per_info * 2]

    def update_target_lanes(self):
        self.lane_target_counter.update_target_lanes()

    def train(self, epochs=200, verbose=0):
        X_train, X_test, y_train, y_test = self.data_center.train_test_split()
        self.nn.train(X_train, X_test, y_train, y_test, epochs=epochs, verbose=verbose)

    def collect(self, last_obs, last_actions, obs, use_lane_target_stats=True):
        self.length_per_info = int(len(last_obs[0]) / 3)
        for agent_id in range(len(obs)):
            lanes = flat(self.agents[agent_id].ob_generator.lanes)
            for i, lane in enumerate(lanes):
                if lane_to_road(lane) not in self.target_roads:
                    continue
                input = []
                output = obs[agent_id][i + self.length_per_info * 2]

                input.append(last_obs[agent_id][i])
                input.append(last_obs[agent_id][i + self.length_per_info * 2])
                input.append(self.lane_phase_available(lane, last_actions[agent_id]))

                source_lanes = get_prev_lanes(lane)
                for source_lane in source_lanes:
                    w = 0.33
                    if use_lane_target_stats:
                        w = self.lane_target_counter.get_rate(source_lane, lane)
                    input.append(self.get_lane_vehicles(source_lane, last_obs) * w)
                    input.append(self.get_waiting_vehicles(source_lane, last_obs) * w)
                    input.append(self.get_passed_vehicles(source_lane, last_obs) * w)
                    input.append(self.lane_phase_available(source_lane, last_actions[self.lane_target_agent(source_lane)]))
                # print(input, output)
                self.data_center.add(input, output)

    def predict(self, last_obs, last_actions, use_lane_target_stats=True, get_sum=False):
        self.length_per_info = int(len(last_obs[0]) / 3)
        result = []
        for agent_id in range(len(last_obs)):
            lanes = flat(self.agents[agent_id].ob_generator.lanes)
            lane_waiting_vehicles = []
            for i, lane in enumerate(lanes):
                if lane_to_road(lane) not in self.target_roads:
                    if self.lane_phase_available(lane, last_actions[agent_id]):
                        lane_waiting_vehicles.append(last_obs[agent_id][i] * 0.8)
                    else:
                        lane_waiting_vehicles.append(last_obs[agent_id][i] * 1.5)
                    continue
                input = []

                input.append(last_obs[agent_id][i])
                input.append(last_obs[agent_id][i + self.length_per_info * 2])
                input.append(self.lane_phase_available(lane, last_actions[agent_id]))

                source_lanes = get_prev_lanes(lane)
                for source_lane in source_lanes:
                    w = 0.33
                    if use_lane_target_stats:
                        w = self.lane_target_counter.get_rate(source_lane, lane)
                    input.append(self.get_lane_vehicles(source_lane, last_obs) * w)
                    input.append(self.get_waiting_vehicles(source_lane, last_obs) * w)
                    input.append(self.get_passed_vehicles(source_lane, last_obs) * w)
                    input.append(self.lane_phase_available(source_lane, last_actions[self.lane_target_agent(source_lane)]))
                # print(input, output)
                input = np.asarray(input)

                output = self.nn.predict(np.asarray([input]))
                output = abs(round(output[0][0]))

                lane_waiting_vehicles.append(output)
            if get_sum:
                result.append(-1 * sum(lane_waiting_vehicles))
            else:
                result.append(np.asarray(lane_waiting_vehicles))
        return result

    def save(self, dir="saves/rw_model/", name="test"):
        self.nn.save_model(dir=dir, name=name)

    def load(self, dir="saves/rw_model/", name="test"):
        self.nn.load_model(dir=dir, name=name)

class PassedVehicleModel():
    def __init__(self, env):
        self.env = env
        self.agents = env.agents
        self.data_center = ModelDataCenter()
        self.nn = NN(11, 2)
        self.lane_target_counter = LaneVehicleTargetCounter(env.world.eng, env.agents)
        m, n = get_raodnet_size(env.agents)
        self.target_roads = get_inside_roads_list(m, n)

    def lane_phase_available(self, lane, phase):
        lane_id = name_to_id(lane)
        if lane_id[-1] == 2:
            return 1
        if phase == 0:
            return 0
        agent = self.agents[self.lane_target_agent(lane)]
        available_startlanes = agent.I.phase_available_startlanes[phase]
        if lane in available_startlanes:
            return 1
        else:
            return 0

    def lane_target_agent(self, lane):
        iid = name_to_id(lane)[0:2]
        direc = name_to_id(lane)[2]
        if direc == 0:
            iid[0] += 1
        elif direc == 1:
            iid[1] += 1
        elif direc == 2:
            iid[0] -= 1
        elif direc == 3:
            iid[1] -= 1
        else:
            raise Exception
        intersection_name = "intersection_{}_{}".format(iid[0], iid[1])
        for agent_id, agent in enumerate(self.agents):
            if intersection_name == agent.I.id:
                return agent_id
        raise Exception

    def get_lane_vehicles(self, lane_name, obs):
        agent_id = self.lane_target_agent(lane_name)
        lane_id = flat(self.agents[agent_id].ob_generator.lanes).index(lane_name)
        return obs[agent_id][lane_id]

    def get_passed_vehicles(self, lane_name, obs):
        agent_id = self.lane_target_agent(lane_name)
        lane_id = flat(self.agents[agent_id].ob_generator.lanes).index(lane_name)
        return obs[agent_id][lane_id + self.length_per_info]


    def update_target_lanes(self):
        self.lane_target_counter.update_target_lanes()

    def train(self, epochs=200, verbose=0):
        X_train, X_test, y_train, y_test = self.data_center.train_test_split()
        print("train mean: {}, test mean: {}".format(np.mean(y_train), np.mean(y_test)))
        self.nn.train(X_train, X_test, y_train, y_test, epochs=epochs, verbose=verbose)
        print("train mean: {}, test mean: {}".format(np.mean(y_train), np.mean(y_test)))

    def collect(self, last_obs, last_actions, obs, use_lane_target_stats=True):
        self.length_per_info = int(len(last_obs[0]) / 3)
        for agent_id in range(len(last_obs)):
            lanes = flat(self.agents[agent_id].ob_generator.lanes)
            for i, lane in enumerate(lanes):
                if lane_to_road(lane) not in self.target_roads:
                    continue
                input = []
                output = [obs[agent_id][i], obs[agent_id][i + self.length_per_info]]
                # output = lane_vehicles[agent_id][i]
                # output = passed_vehicles[lane]

                input.append(last_obs[agent_id][i])
                input.append(self.lane_phase_available(lane, last_actions[agent_id]))

                source_lanes = get_prev_lanes(lane)
                for source_lane in source_lanes:
                    w = 0.33
                    if use_lane_target_stats:
                        w = self.lane_target_counter.get_rate(source_lane, lane)
                    input.append(self.get_lane_vehicles(source_lane, last_obs) * w)
                    input.append(self.get_passed_vehicles(source_lane, last_obs) * w)
                    input.append(self.lane_phase_available(source_lane, last_actions[self.lane_target_agent(source_lane)]))
                # print(input, output)
                self.data_center.add(input, output)

    def predict(self, last_obs, last_actions, use_lane_target_stats=True):
        self.length_per_info = int(len(last_obs[0]) / 3)
        result = []
        for agent_id in range(len(last_obs)):
            lanes = flat(self.agents[agent_id].ob_generator.lanes)
            lane_vehicles = []
            passed_vehicles = []
            for i, lane in enumerate(lanes):
                if lane_to_road(lane) not in self.target_roads:
                    if self.lane_phase_available(lane, last_actions[agent_id]):
                        lane_vehicles.append(last_obs[agent_id][i] * 0.8)
                        passed_vehicles.append(last_obs[agent_id][i] * 0.8)
                    else:
                        lane_vehicles.append(last_obs[agent_id][i] * 1.5)
                        passed_vehicles.append(last_obs[agent_id][i] * 1.5)
                    continue


                input = []

                input.append(last_obs[agent_id][i])
                # input.append(last_obs[agent_id][i + self.length_per_info])
                input.append(self.lane_phase_available(lane, last_actions[agent_id]))

                source_lanes = get_prev_lanes(lane)
                for source_lane in source_lanes:
                    w = 0.33
                    if use_lane_target_stats:
                        w = self.lane_target_counter.get_rate(source_lane, lane)
                    input.append(self.get_lane_vehicles(source_lane, last_obs) * w)
                    input.append(self.get_passed_vehicles(source_lane, last_obs) * w)
                    input.append(self.lane_phase_available(source_lane, last_actions[self.lane_target_agent(source_lane)]))
                # print(input, output)
                input = np.asarray(input)

                output = self.nn.predict(np.asarray([input]))
                output1 = abs(round(output[0][0]))
                output2 = abs(round(output[0][1]))

                lane_vehicles.append(output1)
                passed_vehicles.append(output2)
                # print(lane_vehicles)
                # print(passed_vehicles)
            result.append(np.asarray(lane_vehicles + passed_vehicles))
        return result

    def save(self, dir="saves/ob_model/", name="test"):
        self.nn.save_model(dir=dir, name=name)

    def load(self, dir="saves/ob_model/", name="test"):
        self.nn.load_model(dir=dir, name=name)


class CentralizedTrajectoryBuffer(object):
    def __init__(self, size=10000, save_dir="./data/offline_data/", load_dir="./data/offline_data/", file_name="unknown"):
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

    def set_file(self, save_dir="./data/offline_data/", load_dir="./data/offline_data/", file_name="unknown"):
        self.save_file = os.path.join(save_dir, file_name + ".pkl")
        self.load_file = os.path.join(load_dir, file_name + ".pkl")

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def get_all_states(self):
        obs = []
        for data in self._storage:
            ob, action, reward, next_ob = data
            obs.append(np.array(ob, copy=False))
        return np.array(obs)

    def add(self, obs, actions, rewards, next_obs):
        data = (obs, actions, rewards, next_obs)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obs, actions, rewards, next_obs = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            ob, action, reward, next_ob = data
            obs.append(np.array(ob, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            next_obs.append(np.array(next_ob, copy=False))
        return np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs)

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

class ModelDataCenter(object):
    def __init__(self, size=100000, save_dir="./data/offline_data/", load_dir="./data/offline_data/", file_name="unknown"):
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

        self.set_file(save_dir, load_dir, file_name)

    def __len__(self):
        return len(self._storage)

    def set_file(self, save_dir="./data/offline_data/", load_dir="./data/offline_data/", file_name="unknown"):
        self.save_file = os.path.join(save_dir, file_name + ".pkl")
        self.load_file = os.path.join(load_dir, file_name + ".pkl")

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def train_test_split(self):
        X, y = self.get_all_data()
        print(np.mean(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
        return X_train, X_test, y_train, y_test

    def get_all_data(self):
        return self.sample(len(self._storage))

    def add(self, last_info, lane_vehicles):
        data = (last_info, lane_vehicles)
        # print(data)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        inputs, outputs = [], []
        for i in idxes:
            data = self._storage[i]
            input, output = data
            inputs.append(np.array(input, copy=False))
            outputs.append(np.array(output, copy=False))
        return np.array(inputs), np.array(outputs)

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


class LaneVehicleTargetCounter(object):
    def __init__(self, eng, agents, max_size=10, max_lanes=3):
        self.time = 0
        self.time_limit = 8
        self.agents = agents
        self.eng = eng
        self.data = np.zeros((max_size, max_size, 4, max_lanes, 3))
        m, n = get_raodnet_size(agents)
        self.target_roads = get_inside_roads_list(m, n)
        self.buffer = []

    # def get_target_roads(self, agents):
    #     m, n = get_raodnet_size(agents)
    #     return get_inside_roads_list(m, n)

    def update_target_lanes(self):
        self.time += 1
        for vehicle in self.eng.get_vehicles(include_waiting=False):
            if not self.eng.get_vehicle_info(vehicle)["running"]:
                continue
            lane = id_to_lane(name_to_id(self.eng.get_vehicle_info(vehicle)["drivable"]))
            route = self.eng.get_vehicle_info(vehicle)["route"].strip().split(" ")
            # print(route)
            road = lane_to_road(lane)
            if road in self.target_roads and len(route) > 2:
                # print(lane, route)
                self.add(lane, route[1], route[2])


        for item in self.buffer:
            lane_id, target_id, time = item
            if self.time - time > self.time_limit:
                self.data[lane_id[0]][lane_id[1]][lane_id[2]][lane_id[3]][target_id] -= 1

        while True:
            if len(self.buffer) == 0:
                break
            if self.time - self.buffer[0][2] <= self.time_limit:
                break
            else:
                del self.buffer[0]

    def _add(self, lane_id, target_id):
        self.data[lane_id[0]][lane_id[1]][lane_id[2]][lane_id[3]][target_id] += 1
        self.buffer.append([lane_id, target_id, self.time])

    def add(self, lane_name, next_road_name, next_next_road_name):
        lane_id = name_to_id(lane_name)
        current_dirc = name_to_id(next_road_name)[2]
        target_dirc = name_to_id(next_next_road_name)[2]

        target_lane_idx = get_target_lane_idx(current_dirc, target_dirc)
        self._add(lane_id, target_lane_idx)

    def get_lane_stat(self, lane_name):
        lane_id = name_to_id(lane_name)
        return self.data[lane_id[0]][lane_id[1]][lane_id[2]][lane_id[3]]

    def get_rate(self, source_lane, target_lane):
        lane_id = name_to_id(source_lane)
        current_dirc = name_to_id(source_lane)[2]
        target_dirc = name_to_id(target_lane)[2]
        target_lane_idx = get_target_lane_idx(current_dirc, target_dirc)
        lane_data = self.data[lane_id[0]][lane_id[1]][lane_id[2]][lane_id[3]]

        if sum(lane_data) == 0:
            return 1./3.

        return float(lane_data[target_lane_idx]) / sum(lane_data)

class NN(object):
    def __init__(self, input_length=8, output_length=1):
        self.input_length = input_length
        self.output_length = output_length
        self.model = self._build_model()
        self.batch_size = 32
        self.epochs = 50

        self.logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_dim=self.input_length, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.output_length, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam()
        )
        return model

    def predict(self, X):
        return self.model.predict(X)

    def train(self, X_train, X_test, y_train, y_test, batch_size=None, epochs=None, verbose=0):
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)
        if batch_size is None:
            batch_size = self.batch_size
        if epochs is None:
            epochs = self.epochs
        training_history = self.model.fit(
                            X_train, # input
                            y_train, # output
                            batch_size=batch_size,
                            verbose=verbose, # Suppress chatty output; use Tensorboard instead
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=[tensorboard_callback],
                        )
        # print("keys: ", training_history.history.keys())
        print("Average train loss: ", np.average(training_history.history['loss']))
        print("Average test loss: ", np.average(training_history.history['val_loss']))
        # print(training_history.history['loss'])
        # print(training_history.history['val_loss'])

    def load_model(self, dir="model/dynamics/", name=None):
        if name is None:
            name = "test.h5"
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dynamics/", name=None):
        if name is None:
            name = "test.h5"
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)