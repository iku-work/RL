import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.evaluation import evaluate_policy

import gym
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset, random_split
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from video_callback import VideoCallback
import pathlib
import imageio
import os

from torch.utils.tensorboard import SummaryWriter

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

class ExpertModel:
    
    def __init__(self, 
                 student, 
                 expert_dataset_path, 
                 env,
                 epochs: int = 10,
                 scheduler_gamma: float =  0.7,
                 learning_rate: float = 1,
                 log_interval: int =  100,
                 no_cuda: bool =  True,
                 seed: int =  1,
                 batch_size: int =  64,
                 test_batch_size: int =  1000,
                 tensorboard_log_dir: str = None,
                 verbose: bool = False
                 ):

        self.student = student
        self.expert_dataset_path = expert_dataset_path
        self.env = env

        self.criterion = None
        self.epochs = epochs
        self.scheduler_gamma = scheduler_gamma
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.no_cuda = no_cuda
        self.seed = seed
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.verbose = verbose

        self.use_tb = False
        if(tensorboard_log_dir != None):
            self.use_tb = True
            self.tb_train_step = 0
            self.writer = SummaryWriter(tensorboard_log_dir)

    def get_train_test(self):
        
        try:
            data = pd.read_pickle(self.expert_dataset_path)
        except Exception as e:
            raise e
        
        num_interactions = len(data)

        if isinstance(self.env.action_space, gym.spaces.Box):
            if('CnnPolicy' in self.student.policy_class.__name__):
                expert_observations = np.empty((num_interactions,) + self.env.observation_space.shape, dtype=np.uint8) #(4, 128, 128)
            expert_actions = np.empty((num_interactions,) + (self.env.action_space.shape[0],))
        else:
            expert_observations = np.empty((num_interactions,) + self.env.observation_space.shape)
            expert_actions = np.empty((num_interactions,) + self.env.action_space.shape)

        for i in range(len(data)):
            # Change to channel-first 
            expert_observations[i] = data['obs'].values[i].transpose()
            expert_actions[i] = data['act'].values[i]

        expert_dataset = ExpertDataSet(expert_observations, expert_actions)
        train_size = int(0.8 * len(expert_dataset))
        test_size = len(expert_dataset) - train_size

        return random_split(expert_dataset, [train_size, test_size])
    
    def pretrain_agent(self, train: th.utils.data.dataset.Subset = None, test: th.utils.data.dataset.Subset = None):
        use_cuda = not self.no_cuda and th.cuda.is_available()
        th.manual_seed(self.seed)
        device = th.device("cuda" if use_cuda else "cpu")
        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Extract initial policy
        model = self.student.policy.to(device)

        if(train == None and test == None):
            train_expert_dataset, test_expert_dataset = self.get_train_test()
        else: 
            train_expert_dataset = train
            test_expert_dataset = test

        if(self.verbose):
            print('Train dataset length: {}, Test dataset length: {}'.format(len(train_expert_dataset), len(test_expert_dataset)))

        # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
        # and testing
        train_loader = th.utils.data.DataLoader(
            dataset=train_expert_dataset, batch_size=self.batch_size, shuffle=True, **kwargs
        )
        test_loader = th.utils.data.DataLoader(
            dataset=test_expert_dataset,
            batch_size=self.test_batch_size,
            shuffle=True,
            **kwargs,
        )

        # Define an Optimizer and a learning rate schedule.
        optimizer = optim.Adadelta(model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.scheduler_gamma)

        # Now we are finally ready to train the policy model.
        for epoch in range(1, self.epochs + 1):
            self.train(model, device, train_loader, optimizer, epoch)
            self.test(model, device, test_loader, epoch)
            scheduler.step()

        # Implant the trained policy network back into the RL student agent
        self.student.policy = model
        return self.student

    def train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            #print('Target:', type(target), target)
            #if(type(target) == list):
            #    target = th.as_tensor(target)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(self.env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only
                if isinstance(self.student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    # SAC/TD3:
                    action = model(data)
                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()

            loss = self.criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if(self.use_tb):
                self.writer.add_scalar('Loss/train', self.tb_train_step)
                self.tb_train_step += 1
            
            if ((batch_idx % self.log_interval == 0) and self.verbose):
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def test(self, model, device, test_loader, epoch):
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                if isinstance(self.env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(self.student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()

                test_loss = self.criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        
        if(self.use_tb):        
            self.writer.add_scalar('Loss/Test', self.tb_train_step)
        
        if(self.verbose):
            print(f"Test set: Average loss: {test_loss:.4f}")

'''
env_name = 'forwarder-v0'
env_id = "heavy_pb:{}".format(env_name) 
#env_id = "CartPole-v1"
#env = gym.make(env_id)
env = DummyVecEnv([lambda: gym.make(env_id)])

dataset_name = 'forwarder_107097_steps.pkl'

current_file_dir = pathlib.Path(__file__).parent
base_dir = current_file_dir.parent.parent
dataset_path = pathlib.Path('{}/{}/{}'.format(str(base_dir), 'data', dataset_name))
log_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/logs'))
video_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/logs/videos'))
save_dir = pathlib.Path('{}/{}'.format(str(base_dir),'/models'))


student = PPO("CnnPolicy", env, verbose=1, device='cpu')
#student = DDPG("CnnPolicy", env, verbose=1)
#num_interactions = len(data)

expert_model = ExpertModel(student=student,
                           expert_dataset_path=dataset_path,
                           env=env,
                           epochs=1,
                           no_cuda=True,
                           verbose=1
                           )

student = expert_model.pretrain_agent()

student.save("a2c_student")

callback = VideoCallback(env=env,
                         env_id=env_id, 
                         gif_name='expert', 
                         record_len=200, 
                         video_folder=str(video_dir), 
                         rec_freq=200
                         )

student.learn(600, callback=callback)
''' 