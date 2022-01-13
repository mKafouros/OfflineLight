import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import MaxPressureAgent, FixedTimeAgent
from metric import TravelTimeMetric
import argparse
import numpy as np
from model import OracleModel, DynamicModel, WaitingVehicleModel, PassedVehicleModel, ModelDataCenter, LaneVehicleTargetCounter
# from mute_tf_warnings import tf_mute_warning
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

RANDOM_SEED=42
import time

# tf_mute_warning()

def parse_args():
    # parse args
    parser = argparse.ArgumentParser(description='Run Example')
    parser.add_argument('--policy', type=str, default="maxpressure", help='data collection policy')
    parser.add_argument('--thread', type=int, default=1, help='number of threads')
    parser.add_argument('--steps', type=int, default=3600, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=10, help='how often agent make decisions')
    # parser.add_argument("--target_lane", action="store_true", default=False)
    # parser.add_argument("--signal", action="store_true", default=False)
    args = parser.parse_args()
    return args


def create_env(args, config_file):
    # create world
    world = World(config_file, thread_num=args.thread, max_steps=args.steps)

    # create agents
    agents = []
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(MaxPressureAgent(
            action_space, i, world,
            LaneVehicleGenerator(world, i, ["lane_count", "lane_passed_count", "lane_waiting_count"], in_only=True),
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average="all", negative=True)
        ))

    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, agents, metric)

    return env, agents

def flat(arr):
    result = []
    for i in arr:
        for j in i:
            result.append(j)
    return result

# def update_target_lanes(eng, agents, traffic_counter):
#     lane_vehicles = eng.get_lane_vehicles()
#     inside_roads = traffic_counter.get_inside_roads()
#     for agent_id, agent in enumerate(agents):
#         lanes = flat(agent.ob_generator.lanes)


# def update_avg_lane_vehicles(avg, current_num, obs):
#     count = 0
#     vehicles = 0
#     for ob in obs:
#         for lane_vehicles in ob:
#             count += 1
#             vehicles += lane_vehicles
#     avg = float(avg + vehicles / current_num) / float(1 + count / current_num)
#     current_num += vehicles
#     return avg, current_num

def collect_data(args, env, agents, neighbor_model):

    last_obs = env.reset()
    obs = env.reset()
    i = 0
    while i < args.steps:
        if i % args.action_interval == 0:

            neighbor_model.update_target_lanes()
            actions = []
            for agent_id, agent in enumerate(agents):
                # print(flat(agent.ob_generator.lanes))
                actions.append(agent.get_action(last_obs[agent_id]))
            for _ in range(args.action_interval):
                obs, rewards, dones, info = env.step(actions)
                i += 1
            neighbor_model.collect(last_obs, actions, obs)

            last_obs = obs

args = parse_args()
env, agents = create_env(args, config_file="config/config33offline_test.json")
#
# # # neighbor_model = PassedVehicleModel(env)
neighbor_model = DynamicModel(env)
#
# # collect_data(args, env, agents, neighbor_model)
# args = parse_args()
# neighbor_model = DynamicModel(env)
# file_name = "offline_data_{}_train_exp".format(args.policy)
# neighbor_model.ob_model.data_center.set_file(file_name=file_name)
# neighbor_model.ob_model.data_center.load_from_file()
# neighbor_model.rw_model.data_center.set_file(file_name=file_name)
# neighbor_model.rw_model.data_center.load_from_file()
#
#
# for i in range(12):
#     config_file = "config/config33offline_{}.json".format(i+1)
#     env, agents = create_env(args, config_file)
# # neighbor_model = PassedVehicleModel(env)
#
#     collect_data(args, env, agents, neighbor_model)

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

    def _build_model_CNN(self):
        model = Sequential()
        model.add(keras.layers.Conv2D(32, 1, 3, input_shape=(1, 4, 6)))
        model.add(keras.layers.MaxPooling2D((1, 3)))
        model.add(keras.layers.Conv2D(6, 1, 1, activation='relu'))
        model.add(keras.layers.MaxPooling2D((1, 1)))
        model.add(keras.layers.Conv2D(6, 1, 1, activation='relu'))
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
                            nb_epoch=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=[tensorboard_callback],
                        )
        # print("keys: ", training_history.history.keys())
        print("Average train loss: ", np.average(training_history.history['loss']))
        print("Average test loss: ", np.average(training_history.history['val_loss']))
        return training_history.history['val_loss']
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


file_name = "offline_training_data_fixedtime_exp7"
neighbor_model.ob_model.data_center.set_file(file_name=file_name)
neighbor_model.ob_model.data_center.load_from_file()

print(len(neighbor_model.ob_model.data_center._storage))
print(neighbor_model.ob_model.data_center._storage[0])


X = []
y = []
for i in neighbor_model.ob_model.data_center._storage:
    X.append(i[0])
    y.append(i[0])
X = np.asarray(X)
y = np.asarray(y)
X = X.reshape((-1, 24))
y = y.reshape((-1, 24))
# X = X.reshape((-1, 1, 4, 6))
# y = y.reshape((-1, 1, 4, 6))
nn = NN(24, 24)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

loss = nn.train(X_train, X_test, y_train, y_test, epochs=300, verbose=True)

loss = np.sqrt(loss)
x = np.linspace(0, len(loss), 300)
np.save("result.npy", loss)

plt.plot(x,loss,color='red',label='shit')
plt.savefig("plot.png")

# neighbor_model.train(verbose=True)
# name = "offline_0.h5"

# neighbor_model.save(name=name)

