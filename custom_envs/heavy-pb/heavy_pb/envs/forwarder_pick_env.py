import gym
from heavy_pb.resources.forwarder.forwarder import Forwarder, WoodPile,WoodPile2, MassSensor
import pybullet as p
import numpy as np
import pybullet_data
import matplotlib.pyplot as plt
from time import sleep
import cv2


class ForwarderPick(gym.Env):

    def __init__(self, **kwargs):
        super().__init__()

        # Run in DIRECT mode by default, can be called with GUI
        self.mode = 'DIRECT'
        if('mode' in kwargs):
            self.set_mode(kwargs['mode'])
        else:
            self.set_mode(self.mode)
        
        self.increment = True
        if ('increment' in kwargs):
            self.increment = kwargs['increment']
        
        self.wait = True
        if('wait' in kwargs):
            self.wait = kwargs['wait']
    
        self.vis_obs_width = 60
        self.vis_obs_height = 60
        self.vis_obs_shape = (self.vis_obs_width, self.vis_obs_height)

        self.action_scale = np.array([.05, .05, .05, .05, .5, .5])
        self.action_low = -1
        self.action_max = 1
        self.action_low_arr = np.full((6,), self.action_low,  dtype = np.float32) #* self.action_scale
        self.action_high_arr = np.full((6,), self.action_max,  dtype = np.float32)  #* self.action_scale

        self.update_freq =  160
        self.frameskip = 90

        # Start the simulation
        #self.client = p.connect(p.DIRECT)# p.GUI)# 
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=70, cameraPitch=-22, cameraTargetPosition=[3,0,2])
        p.setPhysicsEngineParameter(enableFileCaching=True)
        
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep = 1/self.update_freq)
        
        self.plane = p.loadURDF("plane.urdf")
        self.forwarder = Forwarder(self.client)
        self.forwarderId = self.forwarder.forwarder

        self.initial_wood_pos = np.array([3.5,0,0.5])
        self.initial_wood_rot = [1.54 ,0, 1.54]
        self.layer_dim = 2
        self.n_layers = 1
        self.wood_offset = 2
        self.woodPile = WoodPile2(self.initial_wood_pos, 
                                  self.initial_wood_rot, 
                                  self.layer_dim , 
                                  self.n_layers, 
                                  self.wood_offset
                                  )

        self.massSensor = MassSensor(self.forwarderId, 
                                triggerVolDim=[4.5, 1.5, 1.5], 
                                excludedBodiesIds=[self.plane])

        self.rendered_img = None
        self.img = None
        self.depth_img = None

        # For wait function
        self.delta_high = 0

        p.enableJointForceTorqueSensor(self.forwarderId, 6)
        p.enableJointForceTorqueSensor(self.forwarderId, 7)

        self.dist_now = 50
        self.last_smallest_dist = 50

        self.action_space = gym.spaces.Box(
            low=self.action_low_arr,
            high= self.action_high_arr,
            dtype = np.float32
        )
        self.reset()
        self.dummy_obs = self.render('rgb_array')

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.dummy_obs.shape,
            dtype = self.dummy_obs.dtype
            #low=np.full((117,), -np.inf, dtype = np.float32),
            #high=np.full((117,), np.inf, dtype = np.float32),
        )

        # Set timestep
        #p.setTimeStep(1/self.update_freq, self.client)
        #self.

    def actWithWait(self, action):

        forwarderId = self.forwarder.forwarder
        #self.forwarder.apply_action(action)
        avg_delta = 1
        i = 0

        self.forwarder.apply_action(action)
        for _ in range(self.frameskip):
        #while(True):
            
            p.stepSimulation()
            
            #if(i % 10):
            jnt_pos_now = []
            #np_jnt_target_pos =np.zeros(len(self.forwarder.active_joints))
            jnt_states  = p.getJointStates(forwarderId, self.forwarder.active_joints)
            
            for ind, jnt in enumerate(jnt_states):
                if(ind != 7 or ind != 8):
                    jnt_pos_now.append(jnt[0])
            
            np_jnt_pos_now = np.array(jnt_pos_now)
            delta = np_jnt_pos_now - action 
            
            avg_delta = np.average(delta)
            #print("avg delta:", avg_delta)
            i += 1

            if (np.abs(avg_delta) < .01):            
                break

    def step(self, action):
        
        reward = 0
        done = False
        info = {}

        if(self.increment):
            action = self.forwarder.incrementJointPosByAction(action, self.action_scale)
        else:
            action = self.forwarder.scaleToJntsLimits(action)
            self.forwarder.apply_action(action)

        if(self.wait):
            self.actWithWait(action)
        else:
            self.forwarder.apply_action(action)
            p.stepSimulation()

        self.dist_now = self.get_dist_to_pile()
        if( self.dist_now < self.last_smallest_dist):
            reward +=  (self.last_smallest_dist - self.dist_now)/self.last_smallest_dist  #.001
            self.last_smallest_dist = self.dist_now

        reward += self.massSensor.getMass()

        if (self.check_grasp()):
            reward += .1

        if (self.check_collision_results()):
            reward = -.0001
            #done = True

        self.img = self.forwarder.camera.getCameraImage()
        #print('Image type: ', type(self.img[2]))
        #obs = self.forwarder.get_observation()
        obs = self.get_depth_img()
        #obs = self.get_segmentation_mask().flatten()
        return self.img[2], reward, done, info

    def reset(self):

        p.resetSimulation(self.client)
        p.setGravity(0,0,-10)

        self.plane = p.loadURDF("plane.urdf")
        self.forwarder = Forwarder(self.client)
        self.woodPile = WoodPile(self.initial_wood_pos, 
                                  self.initial_wood_rot, 
                                  self.layer_dim , 
                                  self.n_layers, 
                                  self.wood_offset
                                  )
        
        #p.enableJointForceTorqueSensor(self.forwarderId, 6)
        #p.enableJointForceTorqueSensor(self.forwarderId, 7)

        self.img = self.forwarder.camera.getCameraImage()
        #obs = self.get_depth_img()
        
        #obs = self.get_segmentation_mask().flatten()
        #obs = self.forwarder.get_observation()
        return self.img[2]
    
    def close(self):

        p.disconnect()
        plt.close()

        return super().close()
    
    def render(self, mode='human'):
        if (mode == 'rgb_array'):
            return self.img[2]
        else:
            if(self.rendered_img == None):
                width = self.forwarder.camera.img_width
                height = self.forwarder.camera.img_height
                channels = 4
                self.rendered_img =  plt.imshow(np.zeros((width, height, channels)), cmap='gray', vmin=0, vmax=10)

            #depth_buffer = self.get_depth_img()
            #self.rendered_img.set_data(depth_buffer)
            self.rendered_img.set_data(self.img[2])
            #self.rendered_img.set_data(self.get_depth_img())
            plt.draw()
            plt.pause(.00001)
        


    def render_obs(self, img):
        
        if(self.rendered_img == None):
            width = self.forwarder.camera.img_width
            height = self.forwarder.camera.img_height
            channels = 4
            self.rendered_img =  plt.imshow(np.zeros((width, height, channels)), cmap='gray', vmin=0, vmax=10)

        #depth_buffer = self.get_depth_img()
        #self.rendered_img.set_data(depth_buffer)
        self.rendered_img.set_data(img)
        #self.rendered_img.set_data(self.get_depth_img())
        plt.draw()
        plt.pause(.00001)

    def get_depth_img(self):
        
        if(self.img != None):
            depth_buffer = self.img[3].copy()
            depth_buffer = cv2.resize(depth_buffer, (self.vis_obs_width, self.vis_obs_height), interpolation= cv2.INTER_LINEAR)
            far = self.forwarder.camera.far
            near = self.forwarder.camera.near
            depth_buffer = far * near / (far - (far - near) * depth_buffer)
            depth_img = depth_buffer
        else:
            return np.zeros([self.vis_obs_width, self.vis_obs_height])
        return depth_img

    def get_segmentation_mask(self):
        
        if(self.img != None):
            #return np.reshape(self.img[4], [self.vis_obs_width, self.vis_obs_height])
            seg_mask = self.img[4].copy().astype('float32')
            seg_mask = cv2.resize(seg_mask, (self.vis_obs_width, self.vis_obs_height), interpolation= cv2.INTER_LINEAR)
            return self.img[4].copy().resize((self.vis_obs_width, self.vis_obs_height))
        else:
            return np.zeros([self.vis_obs_width, self.vis_obs_height])


    def getLookAtPoint(self, forwarderId, parentLinkId, linkId):

        parentLink_pos, parentLink_ori, _,_,_,_ = p.getLinkState(forwarderId, parentLinkId)
        parentLink_ori = link_ori = p.getEulerFromQuaternion(parentLink_ori)
        parentLink_pos = np.array(parentLink_pos)

        link_pos, link_ori,_,_,_,_= p.getLinkState(forwarderId, linkId)
        link_pos = np.array(link_pos)
        
        link_ori = p.getEulerFromQuaternion(link_ori)
        link_ori = np.array(link_pos)

        dist = 10

        look_at_vec = (link_pos - parentLink_pos) 
        look_at_vec = look_at_vec / np.sqrt(look_at_vec[0]**2 + look_at_vec [1]**2 + look_at_vec[2]**2)
        look_at_pos = parentLink_pos + dist*look_at_vec 
        return look_at_pos

    def check_grasp(self): 

        look_at_pos = self.getLookAtPoint(self.forwarderId, 5, 6)
        parentLink_pos, _, _,_,_,_ = p.getLinkState(self.forwarderId, 6)
        parentLink_pos = np.array(parentLink_pos)

        result = p.rayTest(parentLink_pos, look_at_pos)[0]
        hit_pos = np.asarray(result[3])
        hit_dist = np.linalg.norm(hit_pos - parentLink_pos)

        # Check if grapples are closed
        grapple_pos = p.getJointState(self.forwarderId,7)[0]

        if(hit_dist < .2 and grapple_pos < .1):
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
    
    def set_frame_skip(self, frameskip):
        self.frameskip = frameskip

    def set_mode(self, mode):
        mode = mode.upper()
        if(mode == 'DIRECT'):
                self.client = p.connect(p.DIRECT)
        elif(mode =='GUI'):
                self.client = p.connect(p.GUI)

    def get_dist_to_pile(self):
        # Get grapple body pos
        end_ef = p.getLinkState(self.forwarderId, 6)
        end_ef = np.asarray(end_ef[0], dtype=np.float32)
        wood_pos = p.getBasePositionAndOrientation(self.woodPile.wood_list[0])
        return np.linalg.norm(end_ef - np.asarray(wood_pos[0],dtype=np.float32))
        


''' 
from time import sleep

fwd = ForwarderPick()

action = fwd.action_space.sample()
fwd.reset()

delta_high = 0

for i in range(100000):

    action = fwd.action_space.sample()
    obs, rew, done, _ = fwd.step(action)
    print(np.max(obs), np.min(obs))
    fwd.render()

    if (i % 200) == 0 or done:
        print("Reset at step: ", i)
        
        if (delta_high < fwd.delta_high):
            delta_high = fwd.delta_high
            print("New high delta: ", delta_high)

        fwd.reset() '''