import gym 
import numpy as np

from stable_baselines3 import PPO, DDPG, TD3, A2C, SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from video_callback import VideoCallback
import pathlib
import time
from torch.cuda import is_available
import torch as th
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack

from custom_cnn_policy import CustomActorCriticPolicy
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

if(os.name != 'posix'):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            #nn.LSTM(32, features_dim, 3), 
            #nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    #features_extractor_class=CustomCNN,
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    features_extractor_kwargs=dict(features_dim=128),
)

current_file_dir = pathlib.Path(__file__).parent
base_dir = current_file_dir.parent.parent
log_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/logs'))
save_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/models'))

#env_name = 'forwarder-v0'
env_name = 'forwarder-curriculum-v0'
num_cpu = 3  # Number of processes to use
env_id = "heavy_pb:{}".format(env_name) 
#env_id = "BipedalWalker-v3"
total_timesteps = 1000000
eval_freq = 20000
n_eval_episodes = 2
gif_rec_freq = 20000
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

    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    env = DummyVecEnv([lambda: Monitor(gym.make(env_id), filename=None)])
    #env = VecFrameStack(env, n_stack=4)

    eval_callback = EvalCallback(env ,
                                best_model_save_path=save_dir,
                                log_path=log_dir,
                                eval_freq=eval_freq,
                                n_eval_episodes=n_eval_episodes,
                                deterministic=True,
                                render=False,
                                callback_on_new_best=None)

    video_folder = "{}/videos/{}/".format(log_dir,env_name) 
    customCallback = VideoCallback(env=env,
                                   video_folder=video_folder, 
                                    env_id=env_id, 
                                    gif_name='{}'.format(env_name),
                                    rec_freq=gif_rec_freq
                                    )

    #custom_actor_critic = CustomActorCriticPolicy(env.observation_space, action_space=env.action_space, lr_schedule=linear_schedule(.8))
    #model = PPO('CnnPolicy' , env, verbose=1, tensorboard_log=log_dir, device=device, use_sde=False, sde_sample_freq=8)#, policy_kwargs=policy_kwargs) #use_sde - with continious
    #model.load('/Users/ilyakurinov/Documents/University/RL/models/expert_[0.6, 4, 0.8, 64]')
    #model = DDPG("CnnPolicy", env, verbose=1)
    #model = TD3("CnnPolicy", env, verbose=1,)
    '''model = SAC('MlpPolicy', 
                env, 
                use_sde=True,
                sde_sample_freq=8,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(n_sampled_goal=4,
                                          goal_selection_strategy="future",
                                        )
                )'''
    
    model = SAC('CnnPolicy', 
                env, 
                use_sde=True, 
                sde_sample_freq=16, 
                verbose=1,
                buffer_size=100000)
    #model.load('/Users/ilyakurinov/Documents/University/RL/student.zip')
    
    model.learn(total_timesteps=total_timesteps, 
                tb_log_name='ppo_{}'.format(env_name),
                callback=[eval_callback, customCallback]
                )
    model.save('{}/control_{}_CustomFE'.format(str(save_dir),env_name))
    
    '''
    st = time.process_time()
    obs = env.reset()
    print(env.action_space)
    for i in range(2000):
        action = model.predict(obs, deterministic=False)
        print(action[0])
        
        obs, rew, done, _ = env.step(action[0])
        #print(obs.shape)
        #env.render_obs(obs)
        env.env_method('render_obs', obs[0].transpose())
        #env._env_method('render')
        #env.render()

        if(done.all()):
            env.reset()
    # get the end time
    et = time.process_time()

    # get execution time
    res = et - st
    print('CPU Execution time:', res, 'seconds')
    ''' 