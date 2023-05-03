import gym
import torch
from agent import TRPOAgent
import time



nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                            torch.nn.Linear(64, 2))
agent = TRPOAgent(policy=nn)

#agent.load_model("C:/Data/Pybullet/Results/trpo_agent2.pth")
agent.train('heavy_pb:driving-v0', seed=0, batch_size=40000, iterations=5,
            max_episode_length=200, verbose=True)
agent.save_model("C:/Data/Pybullet/Results/trpo_agent2.pth")

env = gym.make('heavy_pb:driving-v0')
ob = env.reset()
while True:
    action = agent(ob)
    ob, _, done, _ = env.step(action)
    #env.render()
    if done:
        ob = env.reset()
        time.sleep(1/0)


