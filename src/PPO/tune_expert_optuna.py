import optuna
#import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from video_callback import VideoCallback
import os
import pathlib
from torch.cuda import is_available
import pandas as pd
from datetime import datetime
from get_expert_trajectories import ExpertModel

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env, filename=None)
        return env
    set_random_seed(seed)
    return _init


n_trails = 20

env_name = 'forwarder-v0'
env_id = "heavy_pb:{}".format(env_name) 
num_cpu = 2  # Number of processes to use
total_timesteps = 5000
eval_freq = 12_000
n_eval_episodes = 10
gif_rec_freq = 10000
device='cpu'

# Uncomment if you want to train with CUDA 
#if(is_available()):
#    device = 'cuda'

dataset_name = 'forwarder_107097_steps.pkl'
current_file_dir = pathlib.Path(__file__).parent
base_dir = current_file_dir.parent.parent
dataset_path = pathlib.Path('{}/{}/{}'.format(str(base_dir), 'data', dataset_name))
log_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/logs'))
save_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/models'))



def objective(trial, train_data, test_data):
    
    scheduler_gamma = trial.suggest_float("scheduler_gamma", .2, .8, step=.2)
    n_epochs = trial.suggest_int("n_epochs", 3, 15)
    learning_rate = trial.suggest_float("learning_rate", .2, .8, step=.2)
    batch_size = trial.suggest_int("batch_size", 32, 248, 32)
    parameters = [round(scheduler_gamma, 2), n_epochs, round(learning_rate, 2), batch_size]
    log_dir = pathlib.Path('{}/{}/trial_{}'.format(base_dir, 'logs', [parameters]))  

    print('Study created. \nScheduler_gamma: {}, \nn_epochs:{},  \nlearning_rate:{}, \nbatch_size:{}'.format(round(scheduler_gamma, 2), n_epochs, round(learning_rate, 2), batch_size))

    student = PPO("CnnPolicy", env)
    
    expert_model = ExpertModel(student=student,
                           expert_dataset_path=dataset_path,
                           env=env,
                           epochs=n_epochs,
                           scheduler_gamma=scheduler_gamma,
                           learning_rate=learning_rate,
                           batch_size=batch_size,
                           tensorboard_log_dir=log_dir,
                           )

    if((train_data == None) and (test_data == None)):
        train_data, test_data = expert_model.get_train_test()

    student = expert_model.pretrain_agent(train=train_data, test=test_data)
    student_save_path = pathlib.Path('{}/expert_{}'.format(save_dir, parameters))
    student.save(student_save_path)

    video_folder = "logs/videos/{}/".format(parameters) 
    customCallback = VideoCallback(video_folder=video_folder, 
                                    env_id=env_id, 
                                    gif_name='{}'.format(env_name),
                                    rec_freq=gif_rec_freq
                                    )

    student.learn(total_timesteps=total_timesteps, 
                tb_log_name='ppo_{}_trial_{}'.format(env_name, parameters),
                callback=[ customCallback]
                )

    mean_reward, _ = evaluate_policy(student, 
                                     env, 
                                     n_eval_episodes, 
                                     False, 
                                     False, 
                                     None, 
                                     None, 
                                     False,
                                     False
                                     )

    return mean_reward

if __name__ == '__main__':

    train_data = None
    test_data = None

    func = lambda trial: objective(trial, train_data, test_data)
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=n_trails)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    filepath = pathlib.Path('{}/trials_{}.csv'.format(log_dir, dt_string))