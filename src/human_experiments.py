# from config import Config
# from experiment_manager import ExperimentManager

# # from TD3_loop import TD3_loop
# from foraging import ForagingEnv

# from stable_baselines3 import TD3
# from stable_baselines3.td3.policies import MlpPolicy
# from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.common.monitor import Monitor

# import numpy as np

# config = Config()
# config = config.parser.parse_args()

# # if config.task == 'foraging':
    
# env = ForagingEnv(config=config)
# env = Monitor(env)

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions),
#                                  sigma=0.1 * np.ones(n_actions))


# # def load(dir1, dir2, env, config):
# #     td = TD3(env, config)
# #     td.load(dir1, dir2)
# #     return td

# def load(name, env):
#     return TD3.load(name, env)


# model = TD3(MlpPolicy, env, config, tensorboard_log="./TD3/", action_noise=action_noise, verbose=1)

# ExperimentManager(config=config,
#                   model=model,
#                   environment=env,
#                   load=load
#                   ).run()

import numpy as np
import os
from stable_baselines3.common.monitor import Monitor

os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from config import Config
from experiment_manager import ExperimentManager
from foraging import ForagingEnv

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise

# from stable_baselines3 import PPO
# from stable_baselines3.ppo.policies import MlpPolicy

config = Config()
config = config.parser.parse_args()


# Create log dir
log_dir = "./PPO/"
os.makedirs(log_dir, exist_ok=True)

env = ForagingEnv(config=config)
env = Monitor(env, log_dir)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=0.1 * np.ones(n_actions))

def load(name, env):
    return TD3.load(name, env)

model = TD3(MlpPolicy,
            env,
            tensorboard_log="./PPO/",
            action_noise=action_noise,
            # n_steps=1024,
            # learning_rate=0.001,
            # gamma=0.8,
            verbose=1) # action_noise=action_noise, 

ExperimentManager(config=config,
                  model=model,
                  environment=env,
                  load=load
                  ).run()
