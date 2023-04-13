import pybullet as p
import os
import math
import numpy as np
import matplotlib as plt

from torch import from_numpy


class CameraSensor():

    def __init__(self, forwarder, mainLink, lookAtLink, offsetZ):
        
        self.mainLink = mainLink
        self.lookAtLink = lookAtLink
        self.offsetZ = offsetZ
        self.forwarder = forwarder
        
        self.camera_up_vector = [0, 0, 1.0]

        self.near = 0.02
        self.far = 30
        
        self.img_width = 60
        self.img_height = 60
        self.fov = 90
        self.aspect_ratio = 1
        

    def getCameraImage(self):

        link_pos, _,_,_,_,_= p.getLinkState(self.forwarder, self.mainLink)
        look_at_pos, _,_,_,_,_= p.getLinkState(self.forwarder, self.lookAtLink)
        look_at_pos = list(look_at_pos)
        look_at_pos[2] += self.offsetZ

        camera_pos = list(link_pos)
        camera_pos[2] += self.offsetZ

        view_matrix = p.computeViewMatrix(
                        cameraEyePosition=camera_pos,
                        cameraTargetPosition=look_at_pos,
                        cameraUpVector=self.camera_up_vector
                        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
                                    fov=self.fov, 
                                    aspect=self.aspect_ratio,
                                    nearVal=self.near,
                                    farVal=self.far
                                    )

        imgs = p.getCameraImage(self.img_width, self.img_height,
                                    view_matrix,
                                    projection_matrix, shadow=False,
                                    renderer=p.ER_TINY_RENDERER,
                                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
                                    )

        return imgs

    def getDepth(self, img):
        
        depth_matrix = img[4]

        depth_tensor = from_numpy(depth_matrix)
        depth_tensor /= self.far

        return depth_tensor
    
'''# For visual observation processing 
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.conv1  = nn.Conv2d

    def forward(self):

        return x'''



# testing purposes, delete later
import random

class Forwarder:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'Forwarder.urdf')
        self.forwarder = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0.1],
                              physicsClientId=client, 
                              #useMaximalCoordinates=True,
                              flags=p.URDF_USE_INERTIA_FROM_FILE
                                    | p.URDF_USE_SELF_COLLISION
                                    | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                              )

        #print(f_name)
        '''
        =====================================================
        ------------------      Joints     ------------------
        =====================================================
        Joint number: 0  Joint name:  cranePillar1
        Joint number: 1  Joint name:  craneArm1
        Joint number: 2  Joint name:  extension1
        Joint number: 3  Joint name:  intermediateHookJnt1
        Joint number: 4  Joint name:  grappleHook1
        Joint number: 5  Joint name:  grappleBody1
        Joint number: 6  Joint name:  grappleL1
        Joint number: 7  Joint name:  grappleR1
        -----------------------------------------------------
        '''

        # Adding one more constraint to the right grapple
        # so it mirrors the left grapple 
        joint_rev = p.createConstraint(self.forwarder, 7, 
                                 self.forwarder, 6, 
                                 p.JOINT_GEAR, 
                                 jointAxis=(0,1,0), 
                                 parentFramePosition=(0,0,0), 
                                 childFramePosition=(0,0,0))
        p.changeConstraint(joint_rev, maxForce=55000,gearRatio=1,erp=0.2)

        '''for joint in range(p.getNumJoints(self.forwarder)):
            info = p.getJointInfo(self.forwarder, joint)
            #print(info)
            print('Joint number: ', info[0], ' Joint name: ', info[12])'''


        self.camera = CameraSensor(self.forwarder, 2,4,-1)

        self.active_joints = [0,1,2,5,7]


    def apply_action(self, action):
        
        '''
        ==========================
        --- Action description ---
        ==========================
        0 - Base rotation,          jointIndex: 0
        1 - Arm rotation,           jointIndex: 1 
        2 - Extension arm rotation  jointIndex: 2
        3 - Grapple body rotation   jointIndex: 5
        4 - Open/close grapples     jointIndex: 7
        ---------------------------
        '''

        #max_velocity = [ .15, .15, .15, .5, .5 ]
        max_force = [ None, None, None, None, 5e4 ]
        active_joints = [0,1,2,5,7]
        # subject to calibration
        self.action_scale = [.08, .05, .025, .05, .5] 
        

        joints = p.getJointStates(
            self.forwarder,
            active_joints
        )
        
        for ind,jnt in enumerate(active_joints):
            if(max_force[ind] == None):
                p.setJointMotorControl2(self.forwarder, 
                                        jnt,
                                        p.POSITION_CONTROL,
                                        targetPosition=joints[ind][0] + (action[ind]*self.action_scale[ind]),
                                        #maxVelocity=max_velocity[ind],
                                        )        
            else:
                p.setJointMotorControl2(self.forwarder, 
                                        jnt,
                                        p.POSITION_CONTROL,
                                        targetPosition=joints[ind][0] + (action[ind]*self.action_scale[ind]),
                                        force=max_force[ind]
                                        )

        '''
        p.setJointMotorControl2(self.forwarder, 0,
                            p.POSITION_CONTROL,
                            targetPosition= action[0],#action[0],
                            maxVelocity=.15
                            )

        p.setJointMotorControl2(self.forwarder, 1,
                            p.POSITION_CONTROL,
                            targetPosition= action[1],#action[1],
                            maxVelocity=.15,
                            force=2e5
                            )   

        p.setJointMotorControl2(self.forwarder, 2,
                            p.POSITION_CONTROL,
                            targetPosition=action[2],
                            maxVelocity=.15
                            )     

        p.setJointMotorControl2(self.forwarder, 5,
                        p.POSITION_CONTROL,
                        targetPosition=action[3]*1.54,
                        maxVelocity=.5
                        )     
        
        p.setJointMotorControl2(self.forwarder, 7,
                        p.POSITION_CONTROL,
                        targetPosition=action[4],
                        maxVelocity=.5,
                        force=5.5e4
                        ) '''


    def get_observation(self):

        observation = np.array([])

        # Get position, orientation in Euler angles, worldLinearVelocity and worldAngularVelocity 
        for link in range(p.getNumJoints(self.forwarder)):
            link_pos, link_rot,_,_,worldLinVel,worldRotVel= p.getLinkState(self.forwarder, link)
            link_rot = p.getEulerFromQuaternion(link_rot)
            link_obs = (link_pos + link_rot + worldLinVel + worldRotVel)
            observation = np.concatenate((observation, np.array(link_obs)))

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


class WoodPile():

    def __init__(self, initialWoodPos, initialWoodRot, layerDim, nLayers, offset):
        self.initialWoodPos = initialWoodPos
        self.initialWoodRot = initialWoodRot
        self.layerDim = layerDim
        self.nLayers = nLayers
        self.offset = offset
        self.f_name = os.path.join(os.path.dirname(__file__), "wood.urdf")

        self.createWoodPile()

    def createWoodPile(self):

        layer_old = 0

        for layer in range(self.nLayers):
            for _ in range(self.layerDim):
                wood = p.loadURDF(self.f_name, 
                        self.initialWoodPos, 
                        p.getQuaternionFromEuler(self.initialWoodRot),
                        useMaximalCoordinates=True,
                        flags=p.URDF_USE_INERTIA_FROM_FILE,
                        #globalScaling=0.8
                        )
                p.changeVisualShape(wood,-1,rgbaColor=[0.8,0.8,0.8,1])
                self.initialWoodPos[1] += self.offset
                if(layer_old != layer):
                    self.initialWoodPos[1] = 1
                    layer_old = layer
            self.initialWoodPos[2] += self.offset