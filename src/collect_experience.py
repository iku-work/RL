import gym
import pybullet as p
import pandas as pd
import pygame
import os

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

#action = 