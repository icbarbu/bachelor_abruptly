import argparse


class Config:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument(
            '--sim-hard',
            default='sim', type=str,
            help="sim for simulation and hard for hardware"
        )

        self.parser.add_argument(
            '--robot-ip-hard',
            default='10.15.3.131', type=str,
            help="Ip of the robot in hardware."
        )

        self.parser.add_argument(
            '--real-time',
            default=0, type=int,
            help="Runs simulation in real timeX4. 1=yes and 0=no, meaning runs as fast as possible."
        )

        self.parser.add_argument(
            '--human-interference',
            default=0, type=int,
            help="If human interference is enabled, using joystick. 0=False and 1=True"
        )

        # simulation

        self.parser.add_argument(
            '--robot-ip',
            default='10.15.3.187', type=str,
            help="Ip of the robot in simulation."
        )

        self.parser.add_argument(
            '--robot-port',
            default=19997, type=int,
            help="Port of the robot."
        )

        # experimental setup

        self.parser.add_argument(
            '--pos',
            default=-1, type=int,
            help="Initial position of the robot. 0-6, and -1 means random."
        )

        self.parser.add_argument(
            '--episode-train-steps',
            default=200, type=int,
            help="Maximum number of steps of an episode while training."
        )

        self.parser.add_argument(
            '--episode-validation-steps',
            default=200, type=int,
            help="Maximum duration of an episode while validating (during training)."
        )

        self.parser.add_argument(
            '--episode-test-steps',
            default=200, type=int,
            help="Maximum duration of an episode while making final tests."
        )

        self.parser.add_argument(
            '--validation-freq',
            default=1000, type=int,
            help="After how many steps the policy should be validated."
        )

        self.parser.add_argument(
            '--number-validations',
            default=7, type=int,
            help="Number of episodes to repeat the validation of the policy during training."
        )

        self.parser.add_argument(
            '--train-or-test',
            default='train', type=str,
            help="If it is running as training or (final) testing: 'train' or 'test'."
        )

        self.parser.add_argument(
            '--number-tests',
            default=1, type=int,
            help="Number of episodes to repeat the test of the policy after training."
        )

        self.parser.add_argument(
            '--training-timesteps',
            default=35000, type=int,
            help="Number of total time steps in the training."
        )

        self.parser.add_argument(
            '--human-timesteps',
            default=1000, type=int,
            help="Number of total time steps used in the human training."
        )

        self.parser.add_argument(
            '--checkpoint-timesteps',
            default=1000, type=int,
            help="After how many time steps sabes checkpoint."
        )

        self.parser.add_argument(
            '--experiment-name',
            default='default_experiment', type=str,
            help="Name of the current experiment."
        )

        # these two are being used only in the human experiments
        self.parser.add_argument(
            '--algorithm',
            default='TD', type=str,
            help="Name of the algorithm; TD or SAC."
        )

        self.parser.add_argument(
            '--task',
            default='foraging', type=str,
            help="foraging or avoiding"
        )
        ##

        # states and actions

        self.parser.add_argument(
            '--min-millis',
            default=500, type=int,
            help="Minimum duration of actions."
        )

        self.parser.add_argument(
            '--max-millis',
            default=1000, type=int,
            help="Maximum duration of actions."
        )

        self.parser.add_argument(
            '--range-speed',
            default=3, type=int,
            help="Maximum/min accelaration of the wheels in simulation (forward)."
        )

        self.parser.add_argument(
            '--range-speed-hard',
            default=100, type=int,
            help="Maximum/min accelaration of the wheels in hardware (forward)."
        )