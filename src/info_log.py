import datetime

class Log:

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def write(self, message, line_break=False):

        print(message)

        with open(f'experiments/{self.experiment_name}/log.txt', "a") as logfile:

            time = datetime.datetime.now().strftime("%Y-%m-%d | %H:%M:%S")

            logfile.write(f'\n{time} ')
            if line_break:
                logfile.write(f'\n')
            logfile.write(f'{message}')
