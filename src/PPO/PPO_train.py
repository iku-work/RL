import gym 
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from video_callback import VideoCallback
import pathlib
import time
from torch.cuda import is_available

if(os.name != 'posix'):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

current_file_dir = pathlib.Path(__file__).parent
base_dir = current_file_dir.parent.parent
log_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/logs'))
save_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/models'))

env_name = 'forwarder-v0'
num_cpu = 3  # Number of processes to use
env_id = "heavy_pb:{}".format(env_name) 
total_timesteps = 50000
eval_freq = 12_000
n_eval_episodes = 10
gif_rec_freq = 10000
device = 'cpu'

# Check if cuda available
if(is_available()):
    device = 'cuda'

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

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    #env = gym.make(env_id, increment=True)
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    env = VecTransposeImage(env)
    eval_callback = EvalCallback(env ,
                                best_model_save_path=save_dir,
                                log_path=log_dir,
                                eval_freq=eval_freq,
                                n_eval_episodes=n_eval_episodes,
                                deterministic=True,
                                render=False,
                                callback_on_new_best=None)

    video_folder = "{}/videos/{}/".format(log_dir,env_name) 
    customCallback = VideoCallback(video_folder=video_folder, 
                                    env_id=env_id, 
                                    gif_name='{}'.format(env_name),
                                    rec_freq=gif_rec_freq
                                    )

    #env.env_method('set_frame_skip', fs)
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, device=device)
    model.learn(total_timesteps=total_timesteps, 
                tb_log_name='ppo_{}'.format(env_name),
                callback=[eval_callback, customCallback]
                )
    model.save(save_dir + 'control_{}'.format(env_name))
    
    #from stable_baselines3.common.env_checker import check_env
    #check_env(env)
    ''' 
    st = time.process_time()
    obs = env.reset()

    for i in range(2000):
        action = model.predict(obs)
        
        obs, rew, done, _ = env.step(action[0])
        #print(obs.shape)
        #env.render_obs(obs)
        env.env_method('render_obs', obs[0])

        if(done.all()):
            env.reset()
    # get the end time
    et = time.process_time()

    # get execution time
    res = et - st
    print('CPU Execution time:', res, 'seconds')
    '''