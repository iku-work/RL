import gym
import numpy as np
import math
import pybullet as p
from heavy_pb.resources.bobcat import Bobcat
from heavy_pb.resources.plane import Plane
from heavy_pb.resources.goal import Goal
import matplotlib.pyplot as plt
import math 
import time


class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):


        self.action_space = gym.spaces.box.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32))

        self.observation_space = gym.spaces.box.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)#p.GUI)#
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/240, self.client)
        #p.setRealTimeSimulation(1)

        self.left = p.addUserDebugParameter('Left', 0, 1, 0.5)
        self.right = p.addUserDebugParameter('Left', 0, 1, 0.5)

        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.goal_obj = None
        self.min_dist = 20
        self.total_steps = 0

        self.last_action = np.zeros(self.action_space.shape)
        self.action_smoothing = 0.7

        self.alpha = 0.9
        self.prev_action = np.zeros(self.action_space.shape)

        self.cam_distance = 100000
        self.img_w, self.img_h = 120, 80

        # Define the window size for smoothing
        self.window_size = 100

        # Initialize a buffer to store previous actions
        self.action_buffer = np.zeros((self.window_size, self.action_space.shape[0]))

        self.target_freq = 10 #Hz
        self.interval = 1 / self.target_freq
        self.now_time = self.time_ms()
        self.last_action_time = self.time_ms()

        self.frame_skip = 80

    def smooth_actions(self, action):
        # Add the current action to the buffer
        self.action_buffer[:-1] = self.action_buffer[1:]
        self.action_buffer[-1] = action
        
        # Compute the moving average of actions over the window
        smoothed_action = np.mean(self.action_buffer, axis=0)
        
        return smoothed_action

        #self.reset()

    def step(self, action):
        # Feed action to the car and get observation of car's state

        #print(p.getPhysicsEngineParameters()['fixedTimeStep'])
        

        self.now_time = self.time_ms()
        self.total_steps += 1

        #smoothed_action = self.last_action * self.action_smoothing + action * (1.0 - self.action_smoothing)
        
        # Low-pass filtering 
        smoothed_action = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.last_action = smoothed_action

        #clipped_action = np.clip(action, -1, 1)

        #action[0] = round(action[0])
        #action[1] = round(action[1])
        events = p.getKeyboardEvents()
            #print(len(events))

        '''if(len(events) != 0):
            for event in events:
                if(event == 113):
                    action[0] = 1
                if(event == 97):
                    action[0] = -1

                if(event == 101):
                    action[1] = 1
                if(event == 100):
                    action[1] = -1'''

        #Control robot with sliders in the GUI
        #left_throttle = p.readUserDebugParameter(self.left)
        #right_throttle = p.readUserDebugParameter(self.right)
        #action = np.array([left_throttle, right_throttle])


        #action = self.smooth_actions(action)

        #self.render()
        #for i in range(50):
        
        #while(True):
            #self.last_action_time = self.time_ms()
        #for _ in range(10):
        for _ in range(self.frame_skip):
            self.car.apply_action(action)
            p.stepSimulation()

            #if((self.last_action_time - self.now_time) >= 10):
            #    break


        car_ob = self.car.get_observation()

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        #reward = max((self.prev_dist_to_goal - dist_to_goal), 0) #max(self.prev_dist_to_goal - dist_to_goal, 0) #- action.sum()*0
        #reward = max((self.min_dist - dist_to_goal), 0)
        #reward = 1 - (dist_to_goal/self.min_dist)

        if(self.min_dist > dist_to_goal):
            self.min_dist = dist_to_goal

        #print(reward, np.exp(-dist_to_goal)*np.sign(self.prev_dist_to_goal - dist_to_goal), action.sum()*np.exp(-dist_to_goal))
        #reward = max(np.exp(-dist_to_goal) * np.sign(self.prev_dist_to_goal - dist_to_goal), 0)
        #reward = -dist_to_goal #- (action.sum()/10)
        self.prev_dist_to_goal = dist_to_goal
        
        #reward = math.exp(-dist_to_goal) - action.sum()

        #reward = -dist_to_goal - action.sum()
        #reward = np.exp(-dist_to_goal) * int(1e6)
        reward = 0
        # Done by running off boundaries
        if (car_ob[0] >= 10 or car_ob[0] <= -10 or
                car_ob[1] >= 10 or car_ob[1] <= -10):
            reward = -.1
            self.done = True
        # Done by reaching goal
        if dist_to_goal < 1:
            reward = 1
            print("Reached")
            self.done = True

        ob = np.array(car_ob + self.goal, dtype=np.float32)

        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.now_time = int(time.time() * 1000)
        self.last_action_time = int(time.time() * 1000)
    
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
        '''if self.rendered_img is None:
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
        plt.pause(.00001)'''

        
        agent_pos, agent_orn =\
        p.getBasePositionAndOrientation(self.car.get_ids()[0])

        yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        xA, yA, zA = agent_pos
        xA = xA
        zA = zA + 0.5 # make the camera a little higher than the robot

        # compute focusing point of the camera
        xB = xA + math.cos(yaw) * self.cam_distance
        yB = yA + math.sin(yaw) * self.cam_distance
        zB = zA

        view_matrix = p.computeViewMatrix(
                            cameraEyePosition=[xA, yA, zA],
                            cameraTargetPosition=[xB, yB, zB],
                            cameraUpVector=[0, 0, 1.0]
                        )

        projection_matrix = p.computeProjectionMatrixFOV(
                                fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)

        imgs = p.getCameraImage(self.img_w, self.img_h,
                                view_matrix,
                                projection_matrix, shadow=True,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)





    def close(self):
        p.disconnect(self.client)

    def time_ms(self):
        return int(time.time()) * 1000
    
    def set_frame_skip(self, fs):
        self.frame_skip = fs