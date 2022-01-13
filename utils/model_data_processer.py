import numpy as np
import random
import pickle
import os
import sys



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

def name_to_id(name):
    return list(map(int, name.split("_")[1:]))

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

