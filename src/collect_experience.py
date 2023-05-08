import gym
import pybullet as p
import pandas as pd
#import pygame
import os
import numpy as np
import pandas as pd
'''
if(os.name != 'posix'):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


pygame.init()
# Initialize the joysticks.
pygame.joystick.init()
#joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
joystick = pygame.joystick.Joystick(0)


env_name = 'forwarder-v0'
env_id = 'heavy_pb:{}'.format(env_name)
env = gym.make(env_id, mode='DIRECT', increment=False, wait=True)
obs = env.reset()

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
   
   # A,B,X,Y,
   # 0,1,2,3
   buttons = list()
   for i in range(joystick.get_numbuttons()):
      button = joystick.get_button(i)
      buttons.append(button)

   # record actions at each timestep
   #action = env.action_space.sample()
   #action = get_action()
   # record reward, observations, dones in each timestep
   
   rew, obs, done, info = env.step(axes_vals) 
   trajectories['act'].append(axes_vals)
   trajectories['rew'].append(rew)
   trajectories['obs'].append(obs)
   trajectories['dones'].append(done)
   env.render()

   #debug_ctrl = read_debug_params()

   #if(debug_ctrl['exit'] == True):   
   #   break

df = pd.DataFrame(trajectories)
print(df.head)
#env.close()
   # 

#action = '''

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
    action = env.action_space.sample()
    expert_observations[i] = obs
    expert_actions[i] = action
    obs, reward, done, info = env.step(action)
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