from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent.dqn_agent import DQNAgent
from metric import TravelTimeMetric
from model.lane_dynamics import CentralizedTrajectoryBuffer
import gym
import argparse
import os
import numpy as np
import logging
from datetime import datetime
from mute_tf_warnings import tf_mute_warning
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
from tqdm import tqdm
import time


def init(args, test=False):
    tf_mute_warning()
    args.save_dir = save_dir + args.config_file[7:-5]
    if test:
        args.save_dir = save_dir

    # config_name = args.config_file.split('/')[1].split('.')[0]
    # args.agent_save_dir = args.save_dir + "/" + config_name
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S') + ".log"))
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # create world
    world = World(args.config_file, thread_num=args.thread, silent=True)

    # create agents
    agents = []
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(DQNAgent(
            action_space,
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None),
            LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
            i.id
        ))
        if args.load_model:
            agents[-1].load_model(args.save_dir)
    if args.share_weights:
        model = agents[0].model
        for agent in agents:
            agent.model = model

    # create metric
    metric = TravelTimeMetric(world)

    # create env
    env = TSCEnv(world, agents, metric)

    return env


# train dqn_agent
def train(env, offline_data_buffer, args):

    avg_rewards = []
    travel_times = []

    last_obs, actions, rewards, obs = offline_data_buffer.sample(2000)
    length = len(last_obs)
    agent = env.agents[0]
    for i in tqdm(range(length)):
        last_ob, action, reward, ob = last_obs[i][:, 0:12], actions[i], rewards[i], obs[i][:, 0:12]
        for agent_id in range(len(env.agents)):
            agent.remember(last_ob[agent_id], action[agent_id], reward[agent_id], ob[agent_id])

    print("start learning")
    for e in range(args.episodes):
        t1 = time.time()

        for i in range(180):
            agent.replay()
            if i % 20 == 19:
                agent.update_target_network()

        travel_time, avg_reward = evaluate(env)
        avg_rewards.append(avg_reward)
        travel_times.append(travel_time)
        t2 = time.time()
        print("episode: {}, mean reward: {}, travel time: {}, time: {}, eta: {}".format(
            e, np.mean(avg_reward), travel_time, t2 - t1, (args.episodes - e - 1) * (t2 - t1)))

        env.agents[0].save_model(model_id=e)

    file = open("reward-dqn-offline-mb.txt", "w")
    for value in avg_rewards:
        file.write(str(value))
        file.write("\n")
    file.close()
    file = open("traveltime-dqn-offline-mb.txt", "w")
    for value in travel_times:
        file.write(str(value))
        file.write("\n")
    file.close()


def evaluate(env):
    obs_n = env.reset()
    step = 0
    rewards = []
    while step < args.steps:
        if step % args.action_interval == 0:
            # get action
            action_n = [agent.get_action(obs, test=True) for agent, obs in
                        zip(env.agents, obs_n)] if not args.share_weights else [env.agents[0].get_action(obs, test=True)
                                                                                for obs in obs_n]
            for _ in range(args.action_interval):
                obs_n, rew_n, done_n, info_n = env.step(action_n)
                step += 1
            rewards.append(np.mean(rew_n))

    return env.eng.get_average_travel_time(), np.mean(rewards)


def test(env, args, model_id=None):
    obs = env.reset()
    if not args.share_weights:
        for agent in env.agents:
            agent.load_model(args.save_dir, model_id=model_id)
    else:
        env.agents[0].load_model(args.save_dir, model_id=model_id)

    i = 0
    while i < args.steps:
        if i % args.action_interval == 0:
            actions = []
            for agent_id, agent in enumerate(env.agents):
                if args.share_weights:
                    agent = env.agents[0]
                actions.append(agent.get_action(obs[agent_id], test=True))
            for _ in range(args.action_interval):
                obs, rewards, dones, info = env.step(actions)
                i += 1
        # print(rewards)

        if all(dones):
            break
    return env.eng.get_average_travel_time()


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Run Example')
    parser.add_argument('postfix', type=str, help='postfix of target dataset')
    parser.add_argument('--policy', type=str, default="maxpressure", help='data collection policy')
    parser.add_argument('--thread', type=int, default=1, help='number of threads')
    parser.add_argument('--steps', type=int, default=3600, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
    parser.add_argument('--episodes', type=int, default=40, help='training episodes')
    parser.add_argument('--share_weights', action="store_true", default=False)
    parser.add_argument('--save_model', action="store_true", default=False)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument("--save_rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument('--save_dir', type=str, default="saves/model/dqn/",
                        help='directory in which model should be saved')
    parser.add_argument('--log_dir', type=str, default="log/dqn", help='directory in which logs should be saved')
    args = parser.parse_args()
    save_dir = args.save_dir

    training_data_name = "offline_training_data_{}_{}".format(args.policy, args.postfix)
    args.config_file = "./config/offline/offline_3X3_{}/config_test.json".format(args.postfix)

    offline_data_buffer = CentralizedTrajectoryBuffer(file_name=training_data_name)
    offline_data_buffer.load_from_file()
    print(len(offline_data_buffer))
    env = init(args, test=True)
    train(env, offline_data_buffer, args)
