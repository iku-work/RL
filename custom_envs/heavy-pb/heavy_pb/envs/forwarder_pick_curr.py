import gymnasium as gym
from heavy_pb.resources.forwarder.forwarder import Forwarder, WoodPile,WoodPile2, MassSensor
import pybullet as p
import numpy as np
import pybullet_data
import matplotlib.pyplot as plt
from time import sleep
import cv2
import random
import pathlib
import os

class ForwarderPickCurr(gym.Env):

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

        self.action_scale = np.array([1, 1, 1, 1, .15, .5])
        self.action_low = -1
        self.action_max = 1
        self.action_low_arr = np.full((6,), self.action_low,  dtype = np.float64) #* self.action_scale
        self.action_high_arr = np.full((6,), self.action_max,  dtype = np.float64)  #* self.action_scale

        self.update_freq =  160
        self.frameskip = 80

        self.render_mode = 'human'

        # Start the simulation
        #self.client = p.connect(p.DIRECT)# p.GUI)# 
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=70, cameraPitch=-22, cameraTargetPosition=[3,0,2])
        p.setPhysicsEngineParameter(fixedTimeStep = 1/self.update_freq,
                                    enableFileCaching=True)
        

        self.initial_wood_pos = np.array([-3.5, 2.5,0.5])
        self.initial_wood_rot = [1.54 ,0, 1.54]
        self.layer_dim = 1
        self.n_layers = 1
        self.wood_offset = 2
        self.grasp = False
        self.mass = 0

        self.rendered_img = None
        self.img = None
        self.depth_img = None

        # For wait function
        self.delta_high = 0

        self.unloading_point = np.array([.1, 3.5, 3.5])
        self.wood_pos = np.zeros(3)

        self.dist_now = 50
        self.last_smallest_dist = 50

        #self.action_space = gym.spaces.MultiDiscrete(np.array([3,3,3,3,3,3]))
        self.init_state = None
        self.forwarderId = None
        self.setWorld()

        p.enableJointForceTorqueSensor(self.forwarderId, 6)
        p.enableJointForceTorqueSensor(self.forwarderId, 7)


        self.lvl = 3
        self.max_lvls = 9
        self.level_progress = 0
        # Level progress value when considered achieved and move to new lvl
        self.learned = 50
        self.log_lost_t = 0

        self.state_filename = self.get_random_state_filename()
        self.load_state(self.state_filename)
        self.forwarder.camera.img_width = self.vis_obs_width
        self.forwarder.camera.img_height = self.vis_obs_height
        self.action_space = gym.spaces.Box(
            low=self.action_low_arr,
            high= self.action_high_arr,
            dtype=np.float32
        )
        
        #obs_low = np.zeros(shape=(4,self.vis_obs_width,self.vis_obs_height))
        #obs_high = np.zeros(shape=(4,self.vis_obs_width,self.vis_obs_height))
        #obs_high.fill(255)
        self.dummy_obs = self.reset()[0]
        obs_low = np.full(fill_value=-np.inf, shape=self.dummy_obs.shape)
        obs_high = np.full(fill_value=np.inf, shape=self.dummy_obs.shape)
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=self.dummy_obs.shape,
            dtype = self.dummy_obs.dtype
            #low=np.full((117,), -np.inf, dtype = np.float32),
            #high=np.full((117,), np.inf, dtype = np.float32),
        )
    
    def get_observation(self):

        observation = np.array([])

        # Get position, orientation in Euler angles, worldLinearVelocity and worldAngularVelocity 
        for link in range(p.getNumJoints(self.forwarderId)):
            link_pos, link_rot,_,_,worldLinVel,worldRotVel= p.getLinkState(self.forwarderId, link)
            link_rot = p.getEulerFromQuaternion(link_rot)
            link_obs = (link_pos + link_rot + worldLinVel + worldRotVel)
            observation = np.concatenate((observation, np.array(link_obs)))

        obs_lis = np.array([self.dist_now, self.lvl, self.learned, self.level_progress, self.mass,self.grasp])
        observation = np.concatenate((observation, self.wood_pos,self.unloading_point, obs_lis ))
        # Concatenate the segmentation mask from camera here
        # Get camera image, camera is attached to Extension and looks at grapple hook
        # Z-axis offset is 1 meter
        #img = self.camera.getCameraImage()
        
        # Get segmentation mask  
        #img_flat = img[4].flatten()

        #print(img_flat)

        # Add flattened image to observation

        # Calculate positions of wood bodies, and check if they are within unloading zone
        # This may be done inside of the gym environment and added there

        return observation

    def get_random_state_filename(self):
        
        current_file_dir = pathlib.Path(__file__).parent.parent
        states_dir = pathlib.Path('{}/{}'.format(str(current_file_dir), 'resources/forwarder/curriculum_states'))
        current_level_dir = '{}/lvl{}'.format(states_dir, self.lvl)
        side = bool(random.getrandbits(1))
        state_files = []

        if(side):
            side_folder = '{}/left'.format(current_level_dir)
        else:
            side_folder = '{}/right'.format(current_level_dir)

        for path in os.listdir(side_folder):
            # check if current path is a file
            if os.path.isfile(os.path.join(side_folder, path)):
                state_files.append(path)
        return '{}/{}'.format(side_folder ,random.choice(state_files))   

    def setWorld(self):
        p.resetSimulation(self.client)
        p.setGravity(0,0,-10)
        self.plane = p.loadURDF("plane.urdf")
        self.forwarder = Forwarder(self.client)
        self.forwarderId = self.forwarder.forwarder
        self.woodPile = WoodPile(self.initial_wood_pos, 
                                self.initial_wood_rot, 
                                self.layer_dim , 
                                self.n_layers, 
                                self.wood_offset
                                )

        self.massSensor = MassSensor(self.forwarderId, 
                            triggerVolDim=[4.5, 1.5, 1.5], 
                            excludedBodiesIds=[self.plane])

        for _ in range(60):
            p.stepSimulation()

    def actWithWait(self, action):
        self.forwarder.apply_action(action)
        for _ in range(self.frameskip):            
            p.stepSimulation()

    def step(self, action):
        
        reward = 0
        done = False
        info = {}

        if(type(self.action_space) == gym.spaces.MultiDiscrete):
            action = action - 1

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

        self.mass = self.massSensor.getMass()
        if(np.greater(self.mass, 0)):
            reward += self.lvl
            self.level_progress += 1
            done = True

        self.grasp = self.check_grasp()

        wood_pos = p.getBasePositionAndOrientation(self.woodPile.wood_list[0])
        self.wood_pos = np.asarray(wood_pos[0])

        if(not np.allclose(self.wood_pos, wood_pos[0], atol=.1)):
            self.wood_pos = wood_pos

        self.dist_now = self.get_dist_to_pile(self.wood_pos)

        if(self.grasp):
            self.log_lost_t = 0
            reward += ((15 - self.get_dist_to_unload()) /15)/2
        else:
            if(not np.allclose(self.wood_pos, wood_pos[0], atol=.1)):
                self.wood_pos = wood_pos
                self.last_smallest_dist = 50
            else:
                if( self.dist_now < self.last_smallest_dist):
                    reward +=  ((15 - self.dist_now)/15) #self.last_smallest_dist  #.001
                    self.last_smallest_dist = self.dist_now

        if (self.check_collision_results()):
            reward = 0

        if((not self.grasp) and np.isclose(self.mass, 0)):
            print("Log lost at: ",self.log_lost_t, ' Lvl: ', self.lvl, 'Lvl progress: ', self.level_progress)
            self.log_lost_t += 1
            if(self.log_lost_t > 5):
                done = True

        self.img = self.forwarder.camera.getCameraImage()
        #print('Reward: {}'.format(reward))
        self.render()
        obs = self.get_observation()
        #self.render_obs(obs.transpose())
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        
        # Start a new level
        if(self.level_progress > self.learned and self.lvl < self.max_lvls):
            self.lvl += 1
            self.level_progress = 0

        self.state_filename = self.state_filename = self.get_random_state_filename()
        self.load_state(self.state_filename)

        self.dist_now = 50
        self.last_smallest_dist = 50

        self.log_lost_t = 0
        
        self.img = self.forwarder.camera.getCameraImage()
        return self.get_observation(), {}
    
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
        
    def get_vis_obs(self, mode='rgb'):
        
        #255 * cmap(depth_relative)[:,:,:3]

        if(mode=='rgb'):
            return self.img[2].transpose().astype(np.uint8)

        elif(mode=='depth'):
            depth = self.get_depth_img()
            return np.dstack((depth, depth, depth, depth)).transpose()
        
        elif(mode=='seg'):

            rgb = self.img[2].copy()
            seg_mask = self.get_segmentation_mask() 
            
            seg_indx_wood = seg_mask < 2
            seg_indx_base = np.logical_not(seg_mask == 1)

            new_mask = np.zeros(shape=seg_mask.shape)
            seg_wood = np.where(seg_indx_wood, new_mask, new_mask+255)
            seg_base = np.where(seg_indx_base, new_mask, new_mask+255)

            #rgb =  cv2.resize(self.img[2], (self.vis_obs_width, self.vis_obs_height), interpolation= cv2.INTER_LINEAR)

            zero_mask = np.zeros(shape=seg_mask.shape)
            seg_mask = np.dstack((seg_base, seg_wood, zero_mask,seg_mask))
            #seg_mask = np.dstack((zero_mask, seg_wood, seg_wood, seg_wood))
            seg_mask = np.asarray(seg_mask, dtype=np.uint8)
            rgb = np.asarray(rgb, dtype=np.uint8)
            return cv2.addWeighted(rgb, 0.9, seg_mask, 0.9,0).transpose()
        

        #return self.img[2]

    def render_obs(self, img):
        
        if(self.rendered_img == None):
            width = self.forwarder.camera.img_width
            height = self.forwarder.camera.img_height
            channels = 4
            self.rendered_img =  plt.imshow(np.zeros((width, height, channels)), cmap='gray', vmin=0, vmax=10)
        
        self.rendered_img.set_data(img)
        plt.draw()
        plt.pause(.00001)

    def get_depth_img(self):
        
        if(self.img != None):
            depth_buffer = self.img[3].copy()
            #depth_buffer = cv2.resize(depth_buffer, (self.vis_obs_width, self.vis_obs_height), interpolation= cv2.INTER_LINEAR)
            far = 1000
            near = self.forwarder.camera.near
            depth_buffer = far * near / (far - (far - near) * depth_buffer)
            depth_img = depth_buffer
        else:
            return np.zeros([self.vis_obs_width, self.vis_obs_height])
        return depth_img

    def get_segmentation_mask(self):
        
        if(self.img != None):
            seg_mask = self.img[4].copy().astype('float32')
            #seg_mask = cv2.resize(seg_mask, (self.vis_obs_width, self.vis_obs_height), interpolation= cv2.INTER_LINEAR)
            return seg_mask
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

    def get_dist_to_pile(self, wood_pos):
        # Get grapple body pos
        end_ef = p.getLinkState(self.forwarderId, 7)
        end_ef = np.asarray(end_ef[0])
        return np.linalg.norm(end_ef - np.asarray(wood_pos,dtype=np.float32))
        
    def get_dist_to_unload(self):
        end_ef = p.getLinkState(self.forwarderId, 7)
        end_ef = np.asarray(end_ef[0])
        return np.linalg.norm(end_ef - np.asarray(self.unloading_point,dtype=np.float32))

    def save_state(self, save_name):
        p.saveBullet(save_name)
    
    def load_state(self, save_name):
        p.restoreState(fileName=save_name)

'''
from time import sleep
import os
if(os.name != 'posix'):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
fwd = ForwarderPickCurr()

action = fwd.action_space.sample()
fwd.reset()
fwd.increment = False

delta_high = 0

for i in range(100000):

    action = fwd.action_space.sample()
    obs, rew, done, _ = fwd.step(action)

    #fwd.render()
    fwd.render_obs(obs.transpose())

    if (i % 200) == 0 or done:
        print("Reset at step: ", i, 'Level: ', fwd.lvl,  'Level progress: ', fwd.level_progress)
        
        if (delta_high < fwd.delta_high):
            delta_high = fwd.delta_high
            print("New high delta: ", delta_high)

        fwd.reset() 
    ''' 