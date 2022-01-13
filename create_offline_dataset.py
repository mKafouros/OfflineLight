import gym
import argparse
import numpy as np
import time
# from mute_tf_warnings import tf_mute_warning

from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import MaxPressureAgent, FixedTimeAgent
from metric import TravelTimeMetric
from model.lane_dynamics import CentralizedTrajectoryBuffer

# tf_mute_warning()

def create_env(args, config_file):
    # create world
    world = World(config_file, thread_num=args.thread, max_steps=args.steps)

    # create agents
    agents = []
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i.phases))
        if args.policy == "maxpressure":
            agents.append(MaxPressureAgent(
                action_space, i, world,
                LaneVehicleGenerator(world, i, ["lane_count", "lane_passed_count", "lane_waiting_count"], in_only=True),
                LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average="all", negative=True)
            ))
        elif args.policy == "fixedtime":
            agents.append(FixedTimeAgent(
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


def collect_data(args, env, agents, data_center):
    last_obs = env.reset()
    obs = env.reset()
    i = 0
    while i < args.steps:
        if i % args.action_interval == 0:

            # neighbor_model.update_target_lanes()
            actions = []
            rewards = []
            for agent_id, agent in enumerate(agents):
                # print(flat(agent.ob_generator.lanes))
                actions.append(agent.get_action(last_obs[agent_id]))
            for _ in range(args.action_interval):
                obs, rewards, dones, info = env.step(actions)
                i += 1
            data_center.add(last_obs, actions, rewards, obs)

            last_obs = obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Example')
    parser.add_argument('config_file', type=str, help='path of config file')
    parser.add_argument('postfix', type=str, help='postfix of target dataset')
    parser.add_argument('--thread', type=int, default=1, help='number of threads')
    parser.add_argument('--policy', type=str, default="fixedtime", help='data collection policy')
    parser.add_argument('--steps', type=int, default=3600, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
    parser.add_argument('--num_flows', type=int, default=12, help='number of training flow files')
    args = parser.parse_args()

    trajectory_buffer = CentralizedTrajectoryBuffer(
        file_name="offline_training_data_{}_{}".format(args.policy, args.postfix))

    for i in range(args.num_flows):
        config_file = args.config_file
        env, agents = create_env(args, config_file)

        collect_data(args, env, agents, trajectory_buffer)
        print(len(trajectory_buffer))

    print(len(trajectory_buffer))
    trajectory_buffer.save_to_file()