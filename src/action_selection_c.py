import numpy as np
import copy
import pygame


class ActionSelection:

    def __init__(self, config):
        self.config = config
        pygame.init()
        pygame.joystick.init()

        if self.config.human_interference:
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()

    def get_range_speed(self):
        if self.config.sim_hard == 'sim':
            range = self.config.range_speed
        else:
            range = self.config.range_speed_hard

        return range

    def normalize_move(self, var):
        range = self.get_range_speed()
        max = range
        min = -range
        return var * (max - min) + min

    def normalize_move_negative(self, var):
        range = self.get_range_speed()
        max = range
        min = -range

        old_min = -1
        old_range = 1 - old_min
        new_range = max - min
        var = (var - old_min) / old_range * new_range + min
        return var

    def select(self, actions):

        left, right = self.robot_mode(actions)
        human_actions = []

        # if present, human actions overrule robot actions
        if self.is_human_active():
            left, right, human_actions = self.human_mode()

        if self.config.sim_hard == 'hard':
            left = int(left)
            right = int(right)

        return left, right, human_actions

    def is_human_active(self):
        if not self.config.human_interference or pygame.joystick.get_count() == 0:
            return False

        pygame.event.get()

        # if passes garbage threshold
        if abs(self.joystick.get_axis(1)) > 0.2 or abs(self.joystick.get_axis(4)) > 0.2:
            return True
        else:
            return False

    def human_mode(self):

        #  left-directional down: -1/1 = front/back
        left = self.joystick.get_axis(1) * -1

        # right-directional: -1/1 = left/right
        right = self.joystick.get_axis(4) * -1

        actions = [left, right]

        left = self.normalize_move_negative(left)
        right = self.normalize_move_negative(right)

        return left, right, actions

    def robot_mode(self, actions):
        if type(actions) is tuple:
            action_move = actions[0].flat[0]
            action_left = actions[0].flat[1]
        else:
            action_move = actions[0]
            action_left = actions[1]

        left = self.normalize_move_negative(action_move)
        right = self.normalize_move_negative(action_left)

        return left, right


