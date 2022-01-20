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

        self.max_food = 4
        self.food_reward = 100

        # init
        self.done = False
        self.total_success = 0
        self.total_hurt = 0
        self.current_step = 0
        self.touched_trigger = None
        self.exp_manager = None
        self.episode_length = 0

        # Define action and sensors space
        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(2,), dtype=np.float32)
        # why high and low?
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(17,), dtype=np.float32)

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
        self.current_step = 0
        self.touched_trigger = None

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
            self.robot.set_phone_tilt(1*math.pi/180)#55
        else:
            self.robot.set_phone_tilt(109)

        sensors = self.get_infrared()
        prop_green_points, color_y, color_x, prop_gray_points, color_y_gray, color_x_gray, prop_pink_points, color_y_pink, color_x_pink = self.detect_color()
        sensors = np.append(sensors, [color_y, color_x, prop_green_points, color_y_gray, color_x_gray, prop_gray_points, color_y_pink, color_x_pink, prop_pink_points])
        sensors = np.array(sensors).astype(np.float32)

        return sensors

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
        prop_green_points, color_y, color_x, prop_gray_points, color_y_gray, color_x_gray, prop_pink_points, color_y_pink, color_x_pink = self.detect_color(human_actions)

        if self.config.sim_hard == 'sim':
            collected_food, aux = self.robot.collected_food()
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
        touched_trigger = self.robot.touched_trigger()[0]
        
        if touched_trigger and self.touched_trigger is None:
            self.touched_trigger = True

        if collected_food - self.total_success > 0:
            food_reward = self.food_reward
        else:
            food_reward = 0

        self.total_success = collected_food
        
        # green sight
        if prop_green_points > 0:
            sight = prop_green_points
        else:
            sight = -0.1
        # pink sight
        if prop_pink_points > 0:
            pink_sight = prop_pink_points
        else:
            pink_sight = -0.1
        
        combined_sight = 0
        touched_trigger_reward = 0
        if self.touched_trigger:
            # green&pink sight
            if prop_green_points > 0 and prop_pink_points > 0:
                combined_sight = (prop_green_points + prop_pink_points) * 100
            else:
                combined_sight = -0.1
            touched_trigger_reward = 50
                        
        # distance = math.sqrt((self.robot.position()[0] - (-3)) ** 2 + (self.robot.position()[1] - 1.625) ** 2)
        # distance_reward = (4 - distance) * 10
            
        sensors = np.append(sensors, [color_y, color_x, prop_green_points, color_y_gray, color_x_gray, prop_gray_points, color_y_pink, color_x_pink, prop_pink_points])
        reward = food_reward + sight + pink_sight + combined_sight + touched_trigger_reward + touched_finish * 10000

        # if episode is over
        # TODO: move this print after counter
        if self.current_step == self.episode_length-1 or touched_finish: #or collected_food == self.max_food
            self.done = True
            self.exp_manager.food_print()

        self.current_step += 1

        self.exp_manager.register_step(reward)

        sensors = sensors.astype(np.float32)

        # info = human_actions
        # if len(human_actions) > 0:
        #     self.exp_manager.human_steps.append(self.exp_manager.current_episode)

        return sensors, reward, self.done, {}

    def render(self, mode='console'):
        pass

    def close(self):
        pass

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

        return prop_green_points, avg_y, avg_x, prop_gray_points, avg_y_gray, avg_x_gray, prop_pink_points, avg_y_pink, avg_x_pink
