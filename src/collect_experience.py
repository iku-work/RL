import gym
import pybullet as p
import pandas as pd
#import pygame
import os
import numpy as np
import pandas as pd
import pygame

from datetime import datetime

if(os.name != 'posix'):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env_name = 'forwarder-v0'
env_id = 'heavy_pb:{}'.format(env_name)
env = gym.make(env_id, mode='DIRECT', increment=True, wait=True)
obs = env.reset()

pygame.init()
# Initialize the joysticks.
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
print('JOYSTICS:', joysticks)
joystick = pygame.joystick.Joystick(0)

invert_control = np.array([-1,1,1,1,1,1])

df = pd.DataFrame()
trajectories = {'act':[], 'rew':[], 'obs':[], 'dones':[]}
i = 0
for i in range(20000):

   pygame.event.get()
   axes_vals = list()

   #for joystick in joysticks:
   axes = joystick.get_numaxes()
      #print(axes)
   for i in range(axes):
      val = joystick.get_axis(i)
      axes_vals.append(val)
   
   action = np.asarray(axes_vals, dtype=np.float32)

   # A,B,X,Y,
   # 0,1,2,3
   buttons = list()
   for ii in range(joystick.get_numbuttons()):
      button = joystick.get_button(ii)
      buttons.append(button)
   

   action = action * invert_control
   obs, rew, done, info = env.step(action) 
   trajectories['act'].append(axes_vals)
   trajectories['rew'].append(rew)
   trajectories['obs'].append(obs)
   trajectories['dones'].append(done)
   env.render()
   print("Reward: ",rew)
   #debug_ctrl = read_debug_params()
   # Y
   if(buttons[3] == True):   
      break
   # X
   if(buttons[2] == True):
      obs = env.reset()
   # B
   if (buttons[1]):
      exit()
      pass
df = pd.DataFrame(trajectories)

# datetime object containing current date and time
now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d_%m_%Y_%H_%M")

pd.to_pickle(df, 'data/expert_data_{}_n_steps_{}.pkl'.format(dt_string, i))

env.close()
   # 
'''
#action = 

env_name = 'forwarder-v0'
env_id = 'heavy_pb:{}'.format(env_name)
env = gym.make(env_id, mode='DIRECT', increment=False, wait=True)
obs = env.reset()

n_steps = 1000

if isinstance(env.action_space, gym.spaces.Box):
    expert_observations = np.empty((n_steps,) + env.observation_space.shape)
    expert_actions = np.empty((n_steps,) + (env.action_space.shape[0],))
    dones = np.empty((n_steps,))

else:
    expert_observations = np.empty((n_steps,) + env.observation_space.shape)
    expert_actions = np.empty((n_steps,) + env.action_space.shape)
    dones = np.empty((n_steps,))

obs = env.reset()
for i in range(n_steps):
   print(i)
   #action = env.action_space.sample()
   axes_vals = []
   axes = joystick.get_numaxes()
      #print(axes)
   for i in range(axes):
      val = joystick.get_axis(i)
      axes_vals.append(val)
   print(axes_vals)
   action = axes_vals
   expert_observations[i] = obs
   expert_actions[i] = action
   obs, reward, done, info = env.step(action)
   env.render()
   #done = terminated or truncated
   if done:
      obs = env.reset()


# record each time a file
# dicard if not okay
# save if okay
# merge later 

#df = pd.DataFrame({'action': expert_actions, 'observations': expert_observations})
#df.to_pickle('models/expert_data.pkl')

np.savez_compressed(
    "models/expert_data",
    expert_actions=expert_actions,
    expert_observations=expert_observations,
)
'''