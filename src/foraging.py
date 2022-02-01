#!/usr/bin/env python3
from __future__ import print_function

import cv2
import gym
from gym import spaces
import numpy as np
import os
import time
import math
import robobo
from action_selection_c import ActionSelection
import re

# TODO: fix this?
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ForagingEnv(gym.Env):
    """
    Custom gym Environment.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, config):

        super(ForagingEnv, self).__init__()

        # params

        self.config = config

        self.max_food = 7
        self.food_reward = 100

        # init
        self.done = False
        self.total_success = 0
        self.total_hurt = 0
        self.current_step = 0
        self.exp_manager = None
        self.episode_length = 0
        self.food_setup = dict()
        self.last_touched_food = None
        self.last_distance = None
        self.total_distance = 0
        self.last_robobo_position = None

        # Define action and sensors space
        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(2,), dtype=np.float32)
        # why high and low?
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(16,), dtype=np.float32)

        self.action_selection = ActionSelection(self.config)

        self.robot = False
        while not self.robot:
            if self.config.sim_hard == 'sim':
                self.robot = robobo.SimulationRobobo(config=self.config).connect(address=self.config.robot_ip, port=self.config.robot_port)
            else:
                self.robot = robobo.HardwareRobobo(camera=True).connect(address=self.config.robot_ip_hard)

            time.sleep(1)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """

        self.done = False
        self.total_success = 0
        self.total_hurt = 0
        self.current_step = 0
        self.last_distance = None
        self.total_distance = 0
        self.last_robobo_position = None

        self.exp_manager.register_episode()

        if self.config.sim_hard == 'sim':
            self.robot.stop_world()
            while self.robot.is_simulation_running():
                pass

            self.robot.set_position()

            self.robot.play_simulation()
            while self.robot.is_simulation_stopped():
                pass

        if self.config.sim_hard == 'sim':
            # degrees to radians
            self.robot.set_phone_tilt(30*math.pi/180)#55
        else:
            self.robot.set_phone_tilt(109)

        sensors = self.get_infrared()
        robobo_position = self.robot.position()
        prop_green_points, color_y, color_x, prop_gray_points, color_y_gray, color_x_gray = self.detect_color()
        sensors = np.append(sensors, [color_y, color_x, prop_green_points, color_y_gray, color_x_gray, prop_gray_points,  robobo_position[0], robobo_position[1]])
        sensors = np.array(sensors).astype(np.float32)

        # self.initialize_food_setup()

        return sensors

    def initialize_food_setup(self):
        handles, positions = self.robot.get_food_setup()
        
        handles_pattern = re.compile(r'[0-9][0-9]')
        positions_pattern = re.compile(r'-?[0-9].[0-9]{2,20}')

        handles = handles_pattern.findall(handles)
        positions = positions_pattern.findall(positions)

        # group in 3-tuples
        positions = list(zip(*[iter(positions)]*3))

        # assign handle-position in dictionary
        for idx, val in enumerate(handles):
            if idx == 0:
                self.food_setup[handles[idx]] = [positions[idx], True]
            else:
                self.food_setup[handles[idx]] = [positions[idx], False]

    def print_food_setup(self):
        for item in self.food_setup.items():
            print(item)

    def get_active_food(self):
        for item in self.food_setup.items():
            if (item[1][1]): 
                return item

    def get_active_food_position(self):
        for item in self.food_setup.items():
            if (item[1][1]): 
                return item[1][0]
    
    def get_active_food_handle(self):
        for item in self.food_setup.items():
            if (item[1][1]): 
                return item[0]

    def set_next_active_food(self, key):
        active_handle = self.get_active_food_handle()
        active_position = self.get_active_food_position()
        next_handle = self.which_food_next(active_handle)
        if next_handle and key == active_handle:
            next_position = self.food_setup[next_handle][0]
            self.food_setup.update({active_handle: [active_position, False]})
            self.food_setup.update({next_handle: [next_position, True]})
        

    def which_food_next(self, key):
        if key == '18':
            return '56'
        elif key == '56':
            return '57'
        elif key == '57':
            return '58'
        elif key == '58':
            return '59'
        elif key == '59':
            return None

    def normal(self, var):
        if self.config.sim_hard == 'sim':
            return var * (self.config.max_speed - self.config.min_speed) + self.config.min_speed
        else:
            return var * (self.config.max_speed_hard - self.config.min_speed_hard) + self.config.min_speed_hard

    def step(self, actions):
        info = {}
        # fetches and transforms actions
        left, right, human_actions = self.action_selection.select(actions)

        self.robot.move(left, right, 400)

        # gets states
        sensors = self.get_infrared()
        prop_green_points, color_y, color_x, prop_gray_points, color_y_gray, color_x_gray = self.detect_color(human_actions)

        if self.config.sim_hard == 'sim':
            collected_food, robobo_hit_wall_position = self.robot.collected_food()
            food_info, robobo_hit_wall_position = self.robot.collected_food()
            collected_food = food_info[0]
            last_touched_food = food_info[1]
        else:
            collected_food = 0

        if self.exp_manager.config.train_or_test == 'train':
            # train
            if self.exp_manager.mode_train_validation == 'train':
                self.episode_length = self.config.episode_train_steps
            # validation
            else:
                self.episode_length = self.config.episode_validation_steps
        else:
            # final test
            self.episode_length = self.config.episode_test_steps

        # calculates rewards
        touched_finish = self.robot.touched_finish()[0]

        robobo_position = self.robot.position()
        
        distance, distance_reward = self.distance_from_goal(robobo_position)

        hit_wall_penalty = 0
        # if x and y are different than 0 which is the default value
        if robobo_hit_wall_position[0] and robobo_hit_wall_position[1]:
            self.total_hurt += 1
            hit_wall_penalty = -5    
        # green sight
        if prop_green_points > 0:
            sight = prop_green_points + distance_reward
        else:
            sight = -10
        
        sensors = np.append(sensors, [color_y, color_x, prop_green_points, color_y_gray, color_x_gray, prop_gray_points, robobo_position[0], robobo_position[1]])
        reward = hit_wall_penalty + sight + touched_finish * 10000
        
        if self.last_distance and distance < self.last_distance:
            self.total_success += 1

        self.last_distance = distance

        if self.last_robobo_position:
            robobo_distance, _ = self.distance_from_goal(robobo_position, self.last_robobo_position[0], self.last_robobo_position[1])
            self.total_distance += robobo_distance

        self.last_robobo_position = robobo_position

        # if episode is over
        # TODO: move this print after counter
        if self.current_step == self.episode_length-1 or touched_finish:
            self.done = True
            self.exp_manager.food_print()

        self.current_step += 1

        self.exp_manager.register_step(reward)

        self.last_touched_food = last_touched_food

        sensors = sensors.astype(np.float32)

        return sensors, reward, self.done, {}

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def distance_from_goal(self, current_position, target_x=1.35, target_y=0.8):
        x1 = current_position[0]
        y1 = current_position[1]
        distance = (((float(target_x) - x1 )**2) + ((float(target_y)-y1)**2))
        # print("distance", distance, "reward", 1 - (distance/26)**0.4)
        
        return distance, 1 - (distance/26)

    def get_infrared(self):

        irs = np.asarray(self.robot.read_irs()).astype(np.float32)

        if self.config.sim_hard == 'hard':
            for idx, val in np.ndenumerate(irs):
                # 100 is the noise of ghost signals
                if irs[idx] >= 100:
                    irs[idx] = 1 / math.log(irs[idx], 2)
                else:
                    irs[idx] = 0

        return irs

    def detect_color(self, human_actions=[]):
        image = self.robot.get_image_front()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if self.config.human_interference == 1 and self.config.sim_hard == 'sim':
            if len(human_actions)>0:
                image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0,0,255])
            cv2.imshow('robot view', image)
            cv2.waitKey(1)

        # mask of green
        mask = cv2.inRange(hsv, (45, 70, 70), (85, 255, 255))
        # mask of gray
        if self.config.sim_hard == 'hard':
            # for hardware, uses a red mask instead of gray
            mask_gray1 = cv2.inRange(hsv, (159, 50, 70), (180, 255, 255))
            mask_gray2 = cv2.inRange(hsv, (0, 50, 70), (9, 255, 255))
            mask_gray = mask_gray1 + mask_gray2
        else:
            mask_gray = cv2.inRange(hsv, (0, 0, 0), (255, 10, 255))

        # cv2.imwrite("imgs/" + str(self.current_step) + "mask.png", mask_gray)
        # cv2.imwrite("imgs/" + str(self.current_step) + "img.png", image)

        size_y = len(image)
        size_x = len(image[0])

        total_points = size_y * size_x
        number_green_points = cv2.countNonZero(mask)
        prop_green_points = number_green_points / total_points
        number_gray_points = cv2.countNonZero(mask_gray)
        prop_gray_points = number_gray_points / total_points

        if cv2.countNonZero(mask) > 0:
            y = np.where(mask == 255)[0]
            x = np.where(mask == 255)[1]

            # average positions normalized by image size
            avg_y = sum(y) / len(y) / (size_y - 1)
            avg_x = sum(x) / len(x) / (size_x - 1)
        else:
            avg_y = 0
            avg_x = 0

        if cv2.countNonZero(mask_gray) > 0:
            y_gray = np.where(mask_gray == 255)[0]
            x_gray = np.where(mask_gray == 255)[1]

            # average positions normalized by image size
            avg_y_gray = sum(y_gray) / len(y_gray) / (size_y - 1)
            avg_x_gray = sum(x_gray) / len(x_gray) / (size_x - 1)
        else:
            avg_y_gray = 0
            avg_x_gray = 0
            
            
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_pink = cv2.inRange(rgb, (175, 0, 175), (255, 100, 255))
        
        number_pink_points = cv2.countNonZero(mask_pink)
        prop_pink_points = number_pink_points / total_points

        if cv2.countNonZero(mask_pink) > 0:
            y_pink = np.where(mask_pink == 255)[0]
            x_pink = np.where(mask_pink == 255)[1]

            # average positions normalized by image size
            avg_y_pink = sum(y_pink) / len(y_pink) / (size_y - 1)
            avg_x_pink = sum(x_pink) / len(x_pink) / (size_x - 1)
        else:
            avg_y_pink = 0
            avg_x_pink = 0

        return prop_green_points, avg_y, avg_x, prop_gray_points, avg_y_gray, avg_x_gray
