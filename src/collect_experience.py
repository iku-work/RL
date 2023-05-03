
import os
import gym
import numpy as np
import pygame as pg
from time import sleep

pg.init()

# Create log dir
log_dir = "C:/Data/Pybullet/Results"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('heavy_pb:driving-v0')

# Train the agent
timesteps = 2e6

def get_action():

   action = np.zeros(2)
    
   for event in pg.event.get():
      if (event.type == 768 and event.dict['unicode'] == 'a'):
         print('a')
         action[0] = 1

      if (event.type == 768 and event.dict['unicode'] == 'd'):
         print('d')
         action[1] = 1

   return action

env.reset()
step = 0

while(True):
  
   action = get_action()
   step += 1
   observation, reward, done, info = env.step(action)

   
   #env.render()
   #if(step > 100):
   #   done = True

   if done :
      observation = env.reset()
      print(step)
      step = 0
      #print("Sim qpos:", env.sim.data.qpos)

env.close()
