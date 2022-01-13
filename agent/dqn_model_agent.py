from . import RLAgent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import os
import time

class DQNModelAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, iid, oracle=False, reconstruct_ob=True):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid
        self.I = ob_generator.I

        self.ob_length = ob_generator.ob_length

        # print(self.ob_length)

        self.reconstruct_ob = reconstruct_ob
        self.reconstruct_ob_length = int(self.ob_length / 3) if self.reconstruct_ob else self.ob_length

        # self.dynamic_model = dynamic_model
        self.oracle = oracle

        self.memory = deque(maxlen=300)
        self.learning_start = 200
        self.update_model_freq = 1
        self.update_target_model_freq = 5

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005 #TODO use lr
        self.batch_size = 8

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()


    def get_action(self, ob, test=False):
        if not test and np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        ob = self._reshape_ob(ob)
        act_values = self.model.predict(ob)
        return np.argmax(act_values[0])

    def get_actions(self, obs, test=False):
        actions = []
        if self.reconstruct_ob:
            obs = self.get_vehicle_count(obs)
        for ob in obs:
            actions.append(self.get_action(ob, test=test))
        return actions

    def sample(self):
        return self.action_space.sample()

    def sample_actions(self, obs):
        actions = []
        for i in range(len(obs)):
            actions.append(self.sample())
        return actions

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # print(self.reconstruct_ob_length)
        model.add(Dense(20, input_dim=self.reconstruct_ob_length, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=RMSprop()
        )
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, obs, actions, rewards, next_obs):
        self.memory.append((obs, actions, rewards, next_obs))

    def get_vehicle_count(self, obs):
        obs = np.asarray(obs)
        vehicle_count = []
        for ob in obs:
            vehicle_count.append(ob[int(len(ob) / 3 * 2):])
        return vehicle_count


    def get_single_batch_data(self, single_batch, dynamic_model, model_steps, use_travel_time=False):
        t0 = time.time()
        obs, actions, rewards, next_obs = single_batch
        # print(obs)

        predicted_rewards_list = []

        # predicted_obs = next_obs
        # for i in range(model_steps):
        #     t5 = time.time()
        #     predicted_actions = self.get_actions(dynamic_model.get_obs(predicted_obs))
        #     t6 = time.time()
        #     predicted_obs, predicted_rewards = dynamic_model.predict(predicted_obs, predicted_actions)
        #     t7 = time.time()
        #     # print(predicted_obs, predicted_rewards)
        #     # predicted_obs_list.append(predicted_obs)
        #     predicted_rewards_list.append(predicted_rewards)

        predicted_obs = dynamic_model.start_rollout(next_obs)
        for i in range(model_steps):
            t5 = time.time()
            predicted_actions = self.get_actions(predicted_obs, test=True)
            t6 = time.time()
            predicted_obs, predicted_rewards = dynamic_model.next_rollout(predicted_actions, get_travel_time=use_travel_time)
            # print(predicted_rewards)
            t7 = time.time()
            predicted_rewards_list.append(predicted_rewards)
        dynamic_model.finish_rollout()


        t2 = time.time()
        obs = np.asarray(dynamic_model.get_obs(obs))
        next_obs = np.asarray(dynamic_model.get_obs(next_obs))
        if self.reconstruct_ob:
            vehicle_count = self.get_vehicle_count(obs)
            target_f = self.model.predict(np.array(vehicle_count))
        else:
            target_f = self.model.predict(np.array(obs))
        t3 = time.time()

        # print(predicted_obs.shape)

        if self.reconstruct_ob:
            predicted_obs = self.get_vehicle_count(predicted_obs)
        # print(rewards)

        for i in range(len(actions)):
            # print(i, len(actions), len(dynamic_model.get_obs(obs)), len(rewards), len(next_obs))
            # print(dynamic_model.get_obs(obs))
            ob, action, reward, next_ob = obs[i], actions[i], rewards[i], next_obs[i]

            if not use_travel_time:
                target = reward
                # print(target)
                step = 0
                while step < model_steps:
                    target += predicted_rewards_list[step][i] * self.gamma**(step+1)
                    # print(target)
                    step += 1
                target += self.gamma**(step+1) * np.amax(self.target_model.predict(np.asarray([predicted_obs[i]])), axis=1)[0]
                # print(target)
            else:
                target = predicted_rewards_list[-1][i] + self.gamma**(model_steps + 1) * np.amax(self.target_model.predict(np.asarray([predicted_obs[i]])), axis=1)[0]
            # print(target_f)
            # print(target)
            target_f[i][action] = target

        t1 = time.time()
        # print("sample time: {:.2f}, rollout time: {:.2f}, model time: {:.2f}, Q time: {:.2f}".format(t1-t0, t2-t0, t3-t2, t1-t3))
        # print("get action time: {:.2f}, model time: {:.2f}".format(t6-t5, t7-t6))
        return np.asarray(obs), np.asarray(target_f)

    def replay(self, dynamic_model, model_steps=1, use_travel_time=False):
        t0 = time.time()
        obs_batch, target_batch = [], []
        minibatch = random.sample(self.memory, self.batch_size)
        t2 = time.time()
        for batch in minibatch:
            obs, target = self.get_single_batch_data(batch, dynamic_model, model_steps, use_travel_time=use_travel_time)
            # print(obs.shape)
            # print(target.shape)
            obs_batch.extend(obs)
            target_batch.extend(target)
        t3 = time.time()
        obs_batch = np.asarray(obs_batch)
        target_batch = np.asarray(target_batch)
        # print(obs_batch.shape)
        # print(target_batch.shape)

        permutation = np.random.permutation(target_batch.shape[0])
        obs_batch = obs_batch[permutation, :]
        target_batch = target_batch[permutation, :]
        if self.reconstruct_ob:
            obs_batch = np.asarray(self.get_vehicle_count(obs_batch))
        history = self.model.fit(obs_batch, target_batch, epochs=1, verbose=0)
        t4 = time.time()
        # print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        t1 = time.time()
        # print("replay time: {:.2f}, sample time: {:.2f}, fit time: {:.2f}".format(t1-t0, t3-t2, t4-t3))

    def load_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)