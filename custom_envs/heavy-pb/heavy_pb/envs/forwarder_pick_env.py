import gym
from heavy_pb.resources.forwarder.forwarder import Forwarder, WoodPile
import pybullet as p
import numpy as np
import pybullet_data
import matplotlib.pyplot as plt
from time import sleep

class ForwarderPick(gym.Env):

    def __init__(self):
        super().__init__()

        #self.action_space = gym.spaces.MultiDiscrete(low=np.array([-1,-1,-1]),
        #                                             high=np.array([1,1,1]))
        #print(self.action_space.sample())
        

        # With np.int type it is a discrete shape 
        self.action_space = gym.spaces.Box(
            low=np.full((5,), -1, dtype = np.float32),
            high=np.full((5,), 1, dtype = np.float32),
            dtype = np.float32
        )
        #print(self.action_space.sample())

        self.observation_space = gym.spaces.Box(
            low=np.full((3600,), -np.inf, dtype = np.float32),
            high=np.full((3600,), np.inf, dtype = np.float32),
        )

        # Start the simulation
        self.client = p.connect(p.DIRECT) # or p.DIRECT
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=70, cameraPitch=-22, cameraTargetPosition=[3,0,2])
        p.setPhysicsEngineParameter(enableFileCaching=True)
        p.setPhysicsEngineParameter(fixedTimeStep = 1/120.)
        #p.setRealTimeSimulation(1)

        self.plane = p.loadURDF("plane.urdf")
        self.forwarder = Forwarder(self.client)
        self.forwarderId = self.forwarder.forwarder
        #self.woodPile = WoodPile([3.5,1,0.5], [1.54 ,0, 1.54], 5, 3, .5)

        self.update_freq = 240
        self.frameskip = 120

        self.rendered_img = None
        self.img = None
        self.depth_img = None


        self.delta_high = 0

        p.enableJointForceTorqueSensor(self.forwarderId, 6)
        p.enableJointForceTorqueSensor(self.forwarderId, 7)

        # Set timestep
        #p.setTimeStep(1/self.update_freq, self.client)
        #self.

        self.aabb = p.getAABB(self.forwarderId)
        self.aabb_min = self.aabb[0]
        self.aabb_max = self.aabb[1]


    def step(self, action):
        
        reward = 0
        done = False
        info = {}

        forwarderId = self.forwarder.forwarder
        
        driven_joints = [0,1,2,5,7]
        self.forwarder.apply_action(action)
        avg_delta = 1

        '''
        # Frameskip with timeout?

        i = 0
        timeout = 250
        self.forwarder.apply_action(action)
        while(abs(avg_delta) > .25):
            
            jnt_target_pos = []
            np_jnt_target_pos =np.zeros(len(driven_joints))
            jnt_states  = p.getJointStates(forwarderId, driven_joints)

            
            for jnt in jnt_states:
                jnt_target_pos.append(jnt[0])
            
            np_jnt_target_pos = np.array(jnt_target_pos)
            delta = np_jnt_target_pos - action
            avg_delta = np.average(delta)
            p.stepSimulation()
            i += 1

            if self.check_collision_results():
                done = True
                reward = -1
                break
            
            if ((avg_delta <= .1) and (i > self.delta_high)):
                self.delta_high = i
            
            if (i>timeout):
                break 

            self.check_grasp()
        '''
        reward = self.getNumLogs()

        if (self.check_grasp()):
            reward += 0.1

        if (self.check_collision_results()):
            done = True
            reward = -1

        #for _ in  range(self.frameskip):
        p.stepSimulation()

        self.img = self.forwarder.camera.getCameraImage()
        #obs = self.forwarder.get_observation()
        obs = self.get_depth_img().flatten()

        return obs, reward, done, info
    
    def reset(self):

        p.resetSimulation(self.client)
        p.setGravity(0,0,-10)

        self.plane = p.loadURDF("plane.urdf")
        self.forwarder = Forwarder(self.client)
        self.woodPile = WoodPile([3.5,1,0.5], [1.54 ,0, 1.54], 1, 1, .5)
        
        #p.enableJointForceTorqueSensor(self.forwarderId, 6)
        #p.enableJointForceTorqueSensor(self.forwarderId, 7)

        self.img = self.forwarder.camera.getCameraImage()
        #obs = self.forwarder.get_observation()
        obs = self.get_depth_img().flatten()

        #obs = self.forwarder.get_observation()
        return obs
    
    def close(self):
        return super().close()
    
    def render(self, mode='human'):

        if(self.rendered_img == None):
            width = self.forwarder.camera.img_width
            height = self.forwarder.camera.img_height
            channels = 4
            self.rendered_img =  plt.imshow(np.zeros((width, height, channels)), cmap='gray', vmin=0, vmax=10)

        #depth_buffer = self.get_depth_img()
        #self.rendered_img.set_data(depth_buffer)
        self.rendered_img.set_data(self.img[2])
        plt.draw()
        plt.pause(.00001)


    def get_depth_img(self):
        
        depth_buffer = np.reshape(self.img[3], [self.forwarder.camera.img_width, self.forwarder.camera.img_height])
        
        far = self.forwarder.camera.far
        near = self.forwarder.camera.near
        depth_buffer = far * near / (far - (far - near) * depth_buffer)
        depth_img = depth_buffer

        return depth_img


    def check_grasp(self): 

        jnts = p.getJointStates(self.forwarderId, [6,7])
        jnts_Fz = []

        for ind,jnt in enumerate(jnts):
            jnts_Fz.append(jnt[2][2])

        avg_fz = sum(jnts_Fz) / len(jnts_Fz)

        if ((avg_fz // 1000) > 5):
            return True

        return False 

    def check_collision_results(self):

        contact_base_grappleBody = p.getContactPoints(
                bodyA= self.forwarderId,
                linkIndexA= -1,
                bodyB= self.forwarderId,
                linkIndexB= 5,
            )

        contact_base_grappleL = p.getContactPoints(
                bodyA= self.forwarderId,
                linkIndexA= -1,
                bodyB= self.forwarderId,
                linkIndexB= 6,
            )
        contact_base_grappleR = p.getContactPoints(
                bodyA= self.forwarderId,
                linkIndexA= -1,
                bodyB= self.forwarderId,
                linkIndexB= 7,
            )     
        
        if (len(contact_base_grappleBody) > 0 
            or len(contact_base_grappleL) > 0
            or len(contact_base_grappleR) > 0
            ):
            return True
        
        return False

    def getNumLogs(self):
        
        overlappingObjs=p.getOverlappingObjects(self.aabb_min, self.aabb_max)

        overlap = 0
        for _,obj in enumerate(overlappingObjs):
            if(obj[0] != self.forwarderId and obj[0] != self.plane):
                overlap += 1

        return overlap
    
'''from time import sleep

fwd = ForwarderPick()

fwd.forwarder.camera.img_height = 480
fwd.forwarder.camera.img_width = 320

action = fwd.action_space.sample()
fwd.step(action)

delta_high = 0

for i in range(100000):

    action = fwd.action_space.sample()
    obs, rew, done, _ = fwd.step(action)
    
    print(obs.shape)

    #fwd.render()

    if (i % 500) == 0 or done:
        print("Reset at step: ", i)
        
        if (delta_high < fwd.delta_high):
            delta_high = fwd.delta_high
            print("New high delta: ", delta_high)

        fwd.reset()'''