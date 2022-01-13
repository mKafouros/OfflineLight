import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import MaxPressureAgent, FixedTimeAgent
from metric import TravelTimeMetric
import argparse
import numpy as np
from model import OracleModel, DynamicModel, WaitingVehicleModel, PassedVehicleModel, ModelDataCenter, LaneVehicleTargetCounter
from mute_tf_warnings import tf_mute_warning
import time

tf_mute_warning()

def parse_args():
    # parse args
    parser = argparse.ArgumentParser(description='Run Example')
    parser.add_argument('config_file', type=str, help='path of config file')
    parser.add_argument('--thread', type=int, default=1, help='number of threads')
    parser.add_argument('--steps', type=int, default=3600, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=10, help='how often agent make decisions')
    # parser.add_argument("--target_lane", action="store_true", default=False)
    # parser.add_argument("--signal", action="store_true", default=False)
    args = parser.parse_args()
    return args

def create_env(args):
    # create world
    world = World(args.config_file, thread_num=args.thread, max_steps=args.steps)

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
env, agents = create_env(args)
# neighbor_model = PassedVehicleModel(env)
neighbor_model = DynamicModel(env)

collect_data(args, env, agents, neighbor_model)


#
# test = np.array([1,2,3,4,5,6,7,8])
# test = test.reshape((8,))


# file_name = "rw"
# neighbor_model.data_center.set_file(file_name=file_name)
# neighbor_model.data_center.load_from_file()

# print(len(neighbor_model.data_center))

# print(time.time())

neighbor_model.train(verbose=True)
name = "{}.h5".format(args.config_file[7:-5])

neighbor_model.save(name=name)




# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())
#
# def test_reward_model(env, agents, neighbor_model):
#
#     last_obs = env.reset()
#     obs = env.reset()
#     actions = []
#     i = 0
#
#     real_rewards = []
#     predicted_rewards = []
#
#     while i < args.steps:
#         if i % args.action_interval == 0:
#             actions = []
#             for agent_id, agent in enumerate(agents):
#                 actions.append(agent.get_action(last_obs[agent_id]))
#             predict = neighbor_model.predict(last_obs, actions, get_sum=True)
#             for _ in range(args.action_interval):
#                 obs, rewards, dones, info = env.step(actions)
#                 i += 1
#             last_obs = obs
#
#
#             real_rewards = real_rewards + rewards
#             predicted_rewards = predicted_rewards + predict
#
#             # print(rewards)
#             # print(predict)
#
#     real_rewards = np.asarray(real_rewards)
#     predicted_rewards = np.asarray(predicted_rewards)
#     predicted_rewards = predicted_rewards / 20.
#
#     print(real_rewards.mean())
#     print(rmse(predicted_rewards, real_rewards))
#     print(rmse(predicted_rewards, real_rewards) / real_rewards.mean())
#
#     print("mean: {}, rmse: {}, %: {}".format(real_rewards.mean(), rmse(predicted_rewards, real_rewards), rmse(predicted_rewards, real_rewards) / real_rewards.mean()))
#
# for i in range(200):
#     neighbor_model.nn.train(X_train, X_test, y_train, y_test, epochs=1, verbose=0)
#     if i % 20 == 10:
#         test_reward_model(env, agents, neighbor_model)

# obs = env.reset()
# actions =  [1,1,2,3,4,5,6,7,0]
# print(neighbor_model.predict(obs, actions, get_sum=True))
# # for _ in range(args.action_interval):
#     obs, rewards, dones, info = env.step(actions)
# print(obs)


# print(neighbor_model.data_center.sample(2)[0])
# print(neighbor_model.lane_model.model.predict(neighbor_model.data_center.sample(2)[0]))
# print(neighbor_model.data_center.sample(2)[0].shape)






# neighbor_model.data_center.save_to_file()
# print(len(data_center))
# print(env.eng.get_average_travel_time())


# print("Final Travel Time is %.4f" % env.metric.update(done=True))