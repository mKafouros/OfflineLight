import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="run experiment")

    ''' Simulation Arguments '''
    parser.add_argument('config_file', type=str, help='path of config file')
    parser.add_argument('--thread', type=int, default=1, help='number of threads')
    parser.add_argument('--steps', type=int, default=3600, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=20, help='how often agent make decisions')

    ''' Experiment Arguments'''
    parser.add_argument("--prefix", type=str, default="exp1", help="")
    parser.add_argument("--parameter_sharing", type=int, default=0, help="")
    parser.add_argument("--test_model_freq", type=int, default=10, help="test model after how many training epochs")

    ''' Path Arguments '''
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--save_model", action="store_true", default=False)

    parser.add_argument("--model_dir", type=str, default="model/iteration/gail", help="directory in which model should be saved")
    parser.add_argument("--best_model_dir", type=str, default="model/iteration/best/gail", help="directory in which model should be saved")

    parser.add_argument("--log_dir", type=str, default="log/iteration", help="directory in which logs should be saved")


    args = parser.parse_args()
    return args
