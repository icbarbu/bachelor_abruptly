import glob
import pickle
import os
import time
import traceback
import vrep

from info_log import Log
from stable_baselines3.common.callbacks import BaseCallback



class CustomCallback(BaseCallback):
    def __init__(self, experiment_manager=None, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.experiment_manager = experiment_manager
        self.num_timesteps = 0

    def _on_step(self) -> bool:
        self.validate_policy()
        self.save_checkpoint()

        return True

    def validate_policy(self):
        if self.num_timesteps % self.experiment_manager.config.validation_freq == 0:
            print('> Validating...')

            # if self.experiment_manager.config.human_interference == 1:
            #     self.experiment_manager.env.robot.stop_world()
            #     while self.experiment_manager.env.robot.is_simulation_running():
            #         pass
            #     vrep.simxSetBooleanParameter(self.experiment_manager.env.robot._clientID, 25, False, vrep.simx_opmode_oneshot)

            for i in range(1, self.experiment_manager.config.number_validations+1):
                self.experiment_manager.mode_train_validation = 'validation'

                self.experiment_manager.config.pos = i-1
                print(' validation', i, 'pos', self.experiment_manager.config.pos)

                obs = self.experiment_manager.env.reset()
                done = False
                while not done:
                    action = self.experiment_manager.model.predict(obs)
                    obs, reward, done, _ = self.experiment_manager.env.step(action)

                self.experiment_manager.mode_train_validation = 'train'

            self.experiment_manager.config.pos = -1

            # if self.experiment_manager.config.human_interference == 1:
            #     if self.experiment_manager.config.real_time == 1:
            #         self.experiment_manager.env.robot.stop_world()
            #         while self.experiment_manager.env.robot.is_simulation_running():
            #             pass
            #         vrep.simxSetBooleanParameter(self.experiment_manager.env.robot._clientID, 25, True, vrep.simx_opmode_oneshot)

            print(' ')

    def save_checkpoint(self):
        if self.num_timesteps % self.experiment_manager.config.checkpoint_timesteps == 0:

            self.experiment_manager.current_checkpoint += 1
            self.experiment_manager.save_checkpoint()
            self.experiment_manager.log.write(f'SAVED checkpoint {self.experiment_manager.current_checkpoint}', True)


class ExperimentManager:

    def __init__(self,
                 config,
                 model,
                 environment,
                 load,
                 log_food_print=False):

        self.config = config
        self.model = model
        self.env = environment.env
        self.load = load
        self.log_food_print = log_food_print
        self.env.exp_manager = self
        self.log = Log(self.config.experiment_name)

        self.results_episodes = []
        self.results_episodes_validation = []
        self.current_checkpoint = 0
        self.current_episode = 0
        self.human_steps = []
        self.mode_train_validation = 'train'

    def register_episode(self):
        if self.mode_train_validation == 'train':
            self.current_episode += 1

    def register_step(self, rewards):

        if self.mode_train_validation == 'train':
            self.results_episodes.append([
                                        self.current_episode,
                                        self.env.current_step,
                                        self.env.total_success,
                                        self.env.total_hurt,
                                        rewards])
        else:
            self.results_episodes_validation.append([
                                        self.current_episode,
                                        self.env.current_step,
                                        self.env.total_success,
                                        self.env.total_hurt,
                                        rewards])


    def food_print(self):
        if self.mode_train_validation == 'train':
            print(f'food in episode  {self.current_episode}: {self.env.total_success}  hurt: {self.env.total_hurt} steps: {self.env.current_step}')
        else:
            print(f'   food: {self.env.total_success} hurt: {self.env.total_hurt} steps: {self.env.current_step}')

        if self.log_food_print:
            file = open(f'experiments/final_tests.txt', 'a')
            file.write(f'{self.config.experiment_name}\t{self.config.sim_hard}\t{self.config.pos}\t{self.env.total_success}\t{self.env.total_hurt}\t {self.env.current_step}\n')
            file.close()

    def run(self):

        # loads checkpoint if existent, or prepares new stage
        self.prepare_stage()

        callback = CustomCallback(experiment_manager=self)

        self.log.write(f'STARTING checkpoint {self.current_checkpoint+1}')

        try:

            if self.config.human_interference == 1:
                remaining_training_timesteps = self.config.human_timesteps
            else:
                remaining_training_timesteps = self.config.training_timesteps-len(self.results_episodes)

            self.model.learn(total_timesteps=remaining_training_timesteps,
                             log_interval=5, #in episodes
                             callback=callback)
        except Exception as error:
            self.log.write(f'ERROR: {traceback.format_exc()}')

            time.sleep(1)

        self.env.robot.stop_world()

    def save_checkpoint(self):

        dir = f'experiments/{self.config.experiment_name}'

        self.model.save(f'{dir}/model_checkpoint_{self.current_checkpoint}', f'{dir}/buffer_checkpoint_{self.current_checkpoint}')

        f = open(f'{dir}/status_checkpoint_{self.current_checkpoint}.pkl', 'wb')
        pickle.dump([
                     self.results_episodes,
                     self.results_episodes_validation,
                     self.current_checkpoint,
                     self.current_episode,
                     self.human_steps
                     ], f)

    def prepare_stage(self):

        dir = f'experiments/{self.config.experiment_name}'
        if not os.path.exists(dir):
            os.makedirs(dir)


        else:

            # recovers the latest non-corrupted checkpoint, if existent

            checkpoints = []
            for file in glob.glob(f'{dir}/status_checkpoint*'):
                checkpoints.append(int(file.split('/status_checkpoint_')[1].split('.')[0]))
                checkpoints.sort()

            attempts = len(checkpoints)-1

            while attempts >= 0:
                try:
                    f = open(f'{dir}/status_checkpoint_{checkpoints[attempts]}.pkl', 'rb')
                    env_data = pickle.load(f)

                    if len(env_data) > 4:
                        self.results_episodes, self.results_episodes_validation, self.current_checkpoint, self.current_episode, self.human_steps = env_data
                    else:
                        self.results_episodes, self.results_episodes_validation, self.current_checkpoint, self.current_episode  = env_data

                    env2 = self.env
                    self.model = self.load(f'{dir}/model_checkpoint_{checkpoints[attempts]}',
                                           f'{dir}/buffer_checkpoint_{self.current_checkpoint}',
                                           env=env2,
                                           config=self.config)

                    attempts = -1
                    self.log.write(f'RECOVERED checkpoint {checkpoints[attempts]}')

                except:
                    self.log.write(f'ERROR: Could not recover checkpoint {checkpoints[attempts]}  {traceback.format_exc()}')
                    self.results_episodes, self.results_episodes_validation, self.current_checkpoint, self.current_episode = [], [], 0, 0

                attempts -= 1

    def test_policy(self, experiment, run, checkpoint):

        self.load_stage(checkpoint)

        print('> Testing ', experiment, run)
        for i in range(1, self.config.number_tests+1):
            finished_test = False

            while not finished_test:
                try:
                    print('  Test', i)
                    obs = self.env.reset()
                    done = False
                    while not done:
                        action = self.model.predict(obs)
                        obs, reward, done, _ = self.env.step(action)
                    finished_test = True
                except Exception as error:
                    print('ERROR: {}'.format(traceback.format_exc()))

        if self.config.sim_hard == 'sim':
            self.env.robot.stop_world()

    #TODO: reuse this function in preparestage later
    def load_stage(self, checkpoint):
        dir = f'experiments/{self.config.experiment_name}'
        f = open(f'{dir}/status_checkpoint_{checkpoint}.pkl', 'rb')
        env_data = pickle.load(f)
        if len(env_data) > 4:
            self.results_episodes, self.results_episodes_validation, self.current_checkpoint, self.current_episode, self.human_steps = env_data
        else:
            self.results_episodes, self.results_episodes_validation, self.current_checkpoint, self.current_episode = env_data
        env2 = self.env
        self.model = self.load(f'{dir}/model_checkpoint_{checkpoint}',
                               '',
                               env=env2,
                               config=self.config)
