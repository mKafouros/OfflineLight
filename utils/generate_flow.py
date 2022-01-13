import json
import argparse
import time
import random
import numpy as np
import os
from tqdm import tqdm

vehicle_template = {
    "vehicle":
         {
             "length": 5.0,
              "width": 2.0,
              "maxPosAcc": 2.0,
              "maxNegAcc": 4.5,
              "usualPosAcc": 2.0,
              "usualNegAcc": 4.5,
              "minGap": 2.5,
              "maxSpeed": 11.111,
              "headwayTime": 2
         },
     "route": ["road_0_1_0", "road_1_1_0", "road_2_1_3"],
     "interval": 1.0,
     "startTime": 0,
     "endTime": 0
}

config_template = {
    "interval": 1.0,
    "seed": 0,
    "dir": "./",
    "saveReplay": False,
    "roadnetFile": "dataset/3X3_offline_train_3/roadnet.json",
    "flowFile": "dataset/3X3_offline_train_3/flow_12.json",
    "rlTrafficLight": True,
    "laneChange": False,
    "roadnetLogFile": "replay/offline_roadnet_3x3.json",
    "replayLogFile": "replay/offline_replay_1x1.txt"
}


def safe_mkdir(target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)


def write_json(data_dict, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data_dict, file, ensure_ascii=False, indent=4)


def create_offline_data_config(date_time, length):
    for i in range(1, length+1):
        train_config = config_template.copy()
        train_config["roadnetFile"] = "dataset/offline/offline_3X3_{}/roadnet.json".format(date_time)
        train_config["flowFile"] = "dataset/offline/offline_3X3_{}/flow_train_{}.json".format(date_time, i)
        write_json(train_config, "config/offline/offline_3X3_{}/config_train_{}.json".format(date_time, i))
    test_config = config_template.copy()
    test_config["roadnetFile"] = "dataset/offline/offline_3X3_{}/roadnet.json".format(date_time)
    test_config["flowFile"] = "dataset/offline/offline_3X3_{}/flow_test.json".format(date_time)
    write_json(test_config, "config/offline/offline_3X3_{}/config_test.json".format(date_time))


def target_intersection_id(road):
    ids = list(map(int, road.split("_")[1:]))
    direction = ids[2]
    if direction == 0:
        ids[0] += 1
    elif direction == 1:
        ids[1] += 1
    elif direction == 2:
        ids[0] -= 1
    else:
        ids[1] -= 1

    return [ids[0], ids[1]]


def turn_to(road, turning_ratio):
    directions = [1, 0, 3, 2]  # clockwise, start from go-north
    intersection_id = target_intersection_id(road)
    current_direction = list(map(int, road.split("_")[1:]))[-1]
    current_direction_idx = directions.index(current_direction)
    rand = random.random()
    if rand < turning_ratio[0]:  # turn left
        direction = directions[(current_direction_idx - 1) % 4]
    elif rand > turning_ratio[0] + turning_ratio[1]:  # turn right
        direction = directions[(current_direction_idx - 1) % 4]
    else:
        direction = current_direction

    if intersection_id[0] >= max_size[0] or intersection_id[1] >= max_size[1] or intersection_id[0] <= 0 or intersection_id[1] <= 0:
        return None
    else:
        return "road_{}_{}_{}".format(intersection_id[0], intersection_id[1], direction)


def get_initial_roads(directions=None):
    if directions is None:
        directions = [0, 1, 2, 3]
    roads = []
    for j in range(1, max_size[1]):
        if 0 in directions:
            roads.append("road_{}_{}_{}".format(0, j, 0))
        if 2 in directions:
            roads.append("road_{}_{}_{}".format(max_size[0], j, 2))

    for i in range(1, max_size[0]):
        if 1 in directions:
            roads.append("road_{}_{}_{}".format(i, 0, 1))
        if 3 in directions:
            roads.append("road_{}_{}_{}".format(i, max_size[1], 3))

    return roads


def create_vehicle(time, route):
    vehicle = vehicle_template.copy()
    vehicle["startTime"] = time
    vehicle["endTime"] = time
    vehicle["route"] = route
    return vehicle


def create_flow_pipeline(roadnet_path, out_flow_path, turn_rate, vehicle_in_freq, directions=[1,2,3,4]):
    roadnet = json.load(open(roadnet_path))
    global max_size
    intersections = roadnet["intersections"]
    ids = []
    for intersection in intersections:
        i_id = list(map(int, intersection["id"].split("_")[1:]))
        ids.append(np.asarray(i_id))
    ids = np.asarray(ids)
    # Assume a NxN road net.
    max_size = ids.max(axis=0)

    init_roads = get_initial_roads(directions=directions)
    vehicles = []
    time = 0
    while time < 3600:
        for start_road in init_roads:
            route = [start_road]
            next_road = turn_to(start_road, turn_rate)
            while next_road is not None:
                route.append(next_road)
                next_road = turn_to(next_road, turn_rate)
            # print(route)
            vehicle = create_vehicle(time, route)
            vehicles.append(vehicle)
        time += vehicle_in_freq
    # print(len(vehicles))
    with open(out_flow_path, 'w') as f:
        json.dump(vehicles, f, indent=1)


if __name__ == '__main__':
    # turning_ratios = [[0.1, 0.8, 0.1], [0.6, 0.2, 0.2],
    #                   [0.2, 0.2, 0.6], [0.3, 0.4, 0.3],
    #                   [0.2, 0.6, 0.2]]  # left, straight, right
    #
    # vehicle_in_freqs = [5, 10, 15]
    now_time = time.strftime("%m%d%H%M%S", time.localtime(int(time.time())))
    roadnet_path = "./dataset/6_6/roadnet_6_6.json"

    # 允许车辆进来的方向: 最多是[0,1,2,3]
    target_directions = [0, 1]
    # 车辆的轨迹中转弯的概率[左转，直行，右转]
    turning_ratios = [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.3, 0.4, 0.3]]
    turning_ratio_test = [0.3, 0.6, 0.1]

    # 车辆进入路网的频率：1代表每个方向每秒进路网一辆车(共3600秒)。（经测试3X3的路网设6会开始比较拥堵，不要设得比5更小了，5-12这个范围就可以）
    vehicle_in_freqs = [6, 8, 10, 12]
    vehicle_in_freq_test = 5

    target_len = len(turning_ratios) * len(vehicle_in_freqs)
    target_dir = "./dataset/offline/offline_3X3_{}".format(now_time)

    safe_mkdir("./dataset/offline/")
    safe_mkdir("./config/offline/")
    safe_mkdir("./config/offline/offline_3X3_{}".format(now_time))
    safe_mkdir(target_dir)

    # create testing data
    out_flow_path = os.path.join(target_dir, "flow_test.json")
    create_flow_pipeline(roadnet_path, out_flow_path, turning_ratio_test, vehicle_in_freq_test,
                         directions=target_directions)

    # create training data
    for i in tqdm(range(target_len)):
        out_flow_path = os.path.join(target_dir, "flow_train_{}.json".format(i + 1))
        turning_ratio = turning_ratios[i % len(turning_ratios)]
        vehicle_in_freq = vehicle_in_freqs[i // len(vehicle_in_freqs)]
        create_flow_pipeline(roadnet_path, out_flow_path, turning_ratio, vehicle_in_freq, directions=target_directions)

    # create corresponding config files for CityFlow
    create_offline_data_config(now_time, target_len)
    write_json(json.load(open(roadnet_path)), "dataset/offline/offline_3X3_{}/roadnet.json".format(now_time))

    print("dataset postfix: {}".format(now_time))
