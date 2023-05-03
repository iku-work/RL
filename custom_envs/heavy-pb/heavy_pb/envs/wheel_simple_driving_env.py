import gym
import numpy as np
import math
import pybullet as p
from heavy_pb.resources.wheelloader.wheel import Bobcat
from heavy_pb.resources.bobcat.plane import Plane
from heavy_pb.resources.bobcat.goal import Goal
import matplotlib.pyplot as plt
import math 
import time


class WheelSimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 5, 5, 10, 10], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.GUI)#p.DIRECT)#
        # Reduce length of episodes for RL algorithms
        #p.setTimeStep(1/30, self.client)

        #p.setRealTimeSimulation(1)

        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.goal_obj = None
        self.min_dist = 20
        self.total_steps = 0

        self.reset()

    def step(self, action):
        # Feed action to the car and get observation of car's state

        events = p.getKeyboardEvents()
            #print(len(events))

        if(len(events) != 0):
            for event in events:
                if(event == 113):
                    action[0] = 1
                if(event == 97):
                    action[0] = -1

                if(event == 101):
                    action[1] = 1
                if(event == 100):
                    action[1] = -1


        self.total_steps += 1

        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        reward = max((self.prev_dist_to_goal - dist_to_goal), 0) - action.sum()/10 #max(self.prev_dist_to_goal - dist_to_goal, 0) #- action.sum()*0
        #reward = max((self.min_dist - dist_to_goal), 0)
        #reward = 1 - (dist_to_goal/self.min_dist)
        #print(1 - (dist_to_goal/self.min_dist), self.min_dist-dist_to_goal, dist_to_goal/self.min_dist)

        if(self.min_dist > dist_to_goal):
            self.min_dist = dist_to_goal

        
        #print(reward, np.exp(-dist_to_goal)*np.sign(self.prev_dist_to_goal - dist_to_goal), action.sum()*np.exp(-dist_to_goal))
        #reward = max(np.exp(-dist_to_goal) * np.sign(self.prev_dist_to_goal - dist_to_goal), 0)
        #reward = -dist_to_goal #- (action.sum()/10)
        self.prev_dist_to_goal = dist_to_goal
        
        #reward = math.exp(-dist_to_goal) - action.sum()
        
        # Done by running off boundaries
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or
                car_ob[1] >= 10 or car_ob[1] <= -10):
            reward = -10
            self.done = True
        # Done by reaching goal
        if dist_to_goal < 1:
            reward = 100
            print("Reached")
            self.done = True

        ob = np.array(car_ob + self.goal, dtype=np.float32)

        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self.client)
        self.car = Bobcat(self.client)

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
             self.np_random.uniform(-5, -9))
        y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
             self.np_random.uniform(-5, -9))
        self.goal = (x, y)
        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)


        # Get observation to return
        car_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))

        self.total_steps = 0

        return np.array(car_ob + self.goal, dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 1
        pos[0] = -1

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)
