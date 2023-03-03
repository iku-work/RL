'''import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('heavy_pb:driving-v0')

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
# Save the agent
model.save("bobcat")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load("bobcat", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
'''

'''


import os

import gym
#import imageio
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import (load_results,
                                                      plot_results, ts2xy)
from stable_baselines3 import TD3

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

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
        return env
    set_random_seed(seed)
    return _init

# Create log dir

log_dir = "C:/Data/Pybullet/Results"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
#env = gym.make('heavy_pb:driving-v0')



env_id = "CartPole-v1"
num_cpu = 4  # Number of processes to use
# Create the vectorized environment
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])



#env = DummyVecEnv([lambda:gym.make('heavy_gym:wheelloader-v0')])
#env = VecNormalize(env, norm_obs=True, norm_reward=True,
#                   clip_obs=10.)
#env = gym.make('Reacher-v2')

env = Monitor(env, log_dir)

#wheelloader_callback = SaveOnBestTrainingRewardCallback()
best_callback = SaveOnBestTrainingRewardCallback(check_freq=50000, log_dir=log_dir)

# 0.0001, 0.2, 0.1, 1280, 4]

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, clip_range=0.2, ent_coef=0.1, n_steps=1280, n_epochs=4, device="auto")

#TD3
#n_actions = env.action_space.shape[-1]
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
#model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)


# Train the agent
timesteps = 2e6

model.learn(total_timesteps=int(timesteps), callback=best_callback)
plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "bobcat")
plt.show()



model.load('C:/Data/Pybullet/Results/best_model.zip')
observation = env.reset()

step = 0

for _ in range(1000000):

  action = model.predict(observation) 
  observation, reward, done, info = env.step(action[0])

  print(action)
  step += 1
  #env.render()

  if done:
    observation = env.reset()
    print(step)
    step = 0
    #print("Sim qpos:", env.sim.data.qpos)

env.close()
'''

import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import plotly.express as px
import os 

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
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "heavy_pb:driving-v0" #"CartPole-v1" 
    num_cpu = 6  # Number of processes to use
    # Create the vectorized environment
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    
    env = gym.make(env_id)
    for i in range(5):
    #env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    #mean_reward, _ = evaluate_policy(model, env, 5, False, True, None, None, False,False)
    #print("Mean reward:", mean_reward)
    
    
    #mean_reward_tracker = MeanRewardTracker()
    
    #model = PPO('MlpPolicy', env, learning_rate=param[0], clip_range=param[1], ent_coef=param[2], n_steps=param[3], n_epochs=param[4])
        model.learn(total_timesteps=10000, tb_log_name='ppo')#, callback=mean_reward_tracker )#, callback=clipper)

    #mean_reward, _ = evaluate_policy(model, env, 5, False, False, None, None, False,False)
    #print("Mean reward:", mean_reward)
        model.save(save_dir + str(i))
    

    #model.load('/Users/ilyakurinov/Documents/University/models/PPO')
    '''
    obs = env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        print(action)
        env.render()

        if dones:
            env.reset()'''

        
    
