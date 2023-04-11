import gym 
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import os 

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import TensorBoardOutputFormat

save_dir = 'models/PPO'
log_dir = 'logs'

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



if __name__ == "__main__":
    env_id = "heavy_pb:forwarder-v0" #"CartPole-v1" 
    num_cpu = 3  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    #eval_env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    eval_callback = EvalCallback(env,
                             best_model_save_path='../../models',
                             log_path=log_dir,
                             eval_freq=12000,
                             n_eval_episodes=10,
                             deterministic=True,
                             render=False,
                             callback_on_new_best=None)
    
    
    #ent_coefs = [.01, .05, .1, .5]
    #frame_skips = [20, 40, 80, 120, 160]

    #for fs in frame_skips:
    #env.env_method('set_frame_skip', fs)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

        #model = PPO('MlpPolicy', env, learning_rate=param[0], clip_range=param[1], ent_coef=param[2], n_steps=param[3], n_epochs=param[4])
    model.learn(total_timesteps=500000, tb_log_name='ppo_control', callback=eval_callback) # + str(fs))# + str(ent_coef))#, callback=mean_reward_tracker )#, callback=clipper)
    model.save(save_dir + 'control') # + str(fs))
    

    #model.load('/Users/ilyakurinov/Documents/University/models/PPO')
    '''obs = env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()

        if dones.any():
            env.reset()'''

        
    
