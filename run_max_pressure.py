import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import MaxPressureAgent
from metric import TravelTimeMetric
import argparse
import time
import numpy as np

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread,  max_steps=args.steps)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(MaxPressureAgent(
        action_space, i, world, 
        LaneVehicleGenerator(world, i, ["lane_count", "lane_waiting_count"], in_only=True),
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True)
    ))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)
t0 = time.time()
env.eng.set_save_replay(True)
# simulate
obs = env.reset()
actions = []
avg_rewards = []
i = 0
while i < args.steps:
    if i % args.action_interval == 0:
        actions = []
        for agent_id, agent in enumerate(agents):
            actions.append(agent.get_action(obs[agent_id]))
        for _ in range(args.action_interval):
            obs, rewards, dones, info = env.step(actions)
            i += 1
        avg_rewards.append(np.mean(rewards))
        # print(sorted(world._get_passed_vehicles().values()))
        # print(obs)
t1 = time.time()
print(env.eng.get_average_travel_time())
print(np.mean(avg_rewards))
print("running time: {}".format(t1-t0))


# print("Final Travel Time is %.4f" % env.metric.update(done=True))