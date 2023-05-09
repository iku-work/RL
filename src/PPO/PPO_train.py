import gym 
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from video_callback import VideoCallback

import time

current_file_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = 'models'
log_dir = 'logs'

if(os.name != 'posix'):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)  

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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

if __name__ == '__main__':

    env_name = 'forwarder-v0'
    env_id = "heavy_pb:{}".format(env_name) 
    num_cpu = 3  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    #env = gym.make(env_id)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    eval_callback = EvalCallback(env ,
                                best_model_save_path='models',
                                log_path=log_dir,
                                eval_freq=12000,
                                n_eval_episodes=10,
                                deterministic=True,
                                render=False,
                                callback_on_new_best=None)
    

    video_folder = "logs/videos/{}/".format(env_name) 
    customCallback = VideoCallback(video_folder=video_folder, 
                                    env_id=env_id, 
                                    gif_name='{}'.format(env_name),
                                    rec_freq=1e3
                                    )

    #env.env_method('set_frame_skip', fs)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device='cpu')
    model.learn(total_timesteps=50000, 
                tb_log_name='ppo_{}'.format(env_name),
                #callback=[eval_callback, customCallback]
                )
    model.save(save_dir + 'control_{}'.format(env_name))    
    
    '''
    st = time.process_time()
    env = gym.make(env_id)
    env.reset()

    for i in range(2000):
        action = env.action_space.sample()
        rew, obs, done, _ = env.step(action)
        env.render()

        print(i)

        if(done):
            env.reset()
    # get the end time
    et = time.process_time()

    # get execution time
    res = et - st
    print('CPU Execution time:', res, 'seconds')
    '''