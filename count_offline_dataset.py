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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Example')
    parser.add_argument('postfix', type=str, help='postfix of target dataset')
    args = parser.parse_args()

    trajectory_buffer = CentralizedTrajectoryBuffer(
        file_name="offline_training_data_{}_{}".format("fixedtime", args.postfix))

    trajectory_buffer.load_from_file()

    print(len(trajectory_buffer))
    print(np.mean(np.mean(trajectory_buffer.get_all_states(), axis=0)))