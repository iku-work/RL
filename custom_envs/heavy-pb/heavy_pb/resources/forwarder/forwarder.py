import pybullet as p
import os
import math
import numpy as np
import matplotlib as plt

from torch import from_numpy


class CameraSensor():

    def __init__(self, forwarder, mainLink, lookAtLink, offsetZ, lookAtOffsetZ):
        
        self.mainLink = mainLink
        self.lookAtLink = lookAtLink
        self.offsetZ = offsetZ
        self.lookAtOffsetZ = lookAtOffsetZ
        self.forwarder = forwarder
        
        self.camera_up_vector = [0, 0, 1.0]

        self.near = 0.02
        self.far = 100
        
        self.img_width = 128
        self.img_height = 128
        self.fov = 90
        self.aspect_ratio = 1

        self.link_pos = None
        self.look_at_pos = None
        self.camera_pos = None
        self.projection_matrix  = None
        self.view_matrix = None
        

    def getCameraImage(self):

        self.link_pos, _,_,_,_,_= p.getLinkState(self.forwarder, self.mainLink)
        self.look_at_pos, _,_,_,_,_= p.getLinkState(self.forwarder, self.lookAtLink)
        self.look_at_pos = np.asarray(self.look_at_pos)
        self.look_at_pos[2] += self.lookAtOffsetZ

        self.camera_pos = np.asarray(self.link_pos)
        self.camera_pos[2] += self.offsetZ

        self.view_matrix = p.computeViewMatrix(
                        cameraEyePosition=self.camera_pos,
                        cameraTargetPosition=self.look_at_pos,
                        cameraUpVector=self.camera_up_vector
                        )
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
                                    fov=self.fov, 
                                    aspect=self.aspect_ratio,
                                    nearVal=self.near,
                                    farVal=self.far
                                    )

        imgs = p.getCameraImage(self.img_width, self.img_height,
                                    self.view_matrix,
                                    self.projection_matrix, shadow=False,
                                    renderer=p.ER_TINY_RENDERER,
                                    #flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
                                    )
        return imgs

    
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
                              flags= p.URDF_USE_INERTIA_FROM_FILE
                                    | p.URDF_USE_SELF_COLLISION 
                                    | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                                    | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                              )
        
        #print(f_name)
        '''
        =====================================================
        ------------------      Joints     ------------------
        =====================================================
        Joint number: 0  Joint name:  cranePillar1
        Joint number: 1  Joint name:  craneArm1
        Joint number: 2  Joint name:  extensionArm1
        Joint number  3  Joint name:  extension1
        Joint number: 4  Joint name:  intermediateHookJnt1
        Joint number: 5  Joint name:  grappleHook1
        Joint number: 6  Joint name:  grappleBody1
        Joint number: 7  Joint name:  grappleL1
        Joint number: 8  Joint name:  grappleR1
        -----------------------------------------------------
        '''

        # Adding one more constraint to the right grapple
        # so it mirrors the left grapple 
        joint_rev = p.createConstraint(self.forwarder, 8, 
                                 self.forwarder, 7, 
                                 p.JOINT_GEAR, 
                                 jointAxis=(0,1,0), 
                                 parentFramePosition=(0,0,0), 
                                 childFramePosition=(0,0,0))
        p.changeConstraint(joint_rev, maxForce=55000,gearRatio=1,erp=0.2)

        '''for joint in range(p.getNumJoints(self.forwarder)):
            info = p.getJointInfo(self.forwarder, joint)
            #print(info)
            print('Joint number: ', info[0], ' Joint name: ', info[12])'''

        self.camera = CameraSensor(self.forwarder, 3,4,-0.5, -4)
        self.max_velocity = [ .1, .1, .1, .1, .8, .5 ]
        self.max_force = [ None, 5e5, None, None, None, 5e4 ]
        self.active_joints = [0,1,2,3,6,8]
        # subject to calibration
        

        self.action_min, self.action_max = self.getJointsLimits()

    def getJointsLimits(self):

        action_min = list()
        action_max = list()
        
        for jnt in self.active_joints:
            info = p.getJointInfo(self.forwarder, jnt)
            action_min.append(info[8])
            action_max.append(info[9])

        return action_min, action_max

    def incrementJointPosByAction(self, action, action_scale = [.15,.15,.15,.15,1,.5]):

        joints_actions = list()
        joints = p.getJointStates(
            self.forwarder,
            self.active_joints
        )

        for ind, jnt in enumerate(joints):
            joints_actions.append(joints[ind][0] + action[ind]*action_scale[ind])

        return joints_actions

    def scaleToJntsLimits(self, action, model_action_min=-1, model_action_max=1):
        joints_actions = list()
        for ind, jnt in enumerate(self.active_joints):
            #new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
            #print(model_action_min, model_action_max, model_action_max[ind], model_action_min[ind])
            new_value_1 = ( (action[ind] - model_action_min) / (model_action_max - model_action_min) )
            new_value_2 = (self.action_max[ind] - self.action_min[ind])
            new_value =  new_value_1 * new_value_2 + self.action_min[ind]
            joints_actions.append(new_value)

        return joints_actions


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

        for ind,jnt in enumerate(self.active_joints):
            if(self.max_force[ind] == None):
                p.setJointMotorControl2(self.forwarder, 
                                        jnt,
                                        p.POSITION_CONTROL,
                                        targetPosition= action[ind],
                                        maxVelocity=self.max_velocity[ind],
                                        )        
            else:
                p.setJointMotorControl2(self.forwarder, 
                                        jnt,
                                        p.POSITION_CONTROL,
                                        targetPosition=action[ind],
                                        force=self.max_force[ind],
                                        maxVelocity = self.max_velocity[ind]
                                        )
        
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

class TriggerVolume():

    def __init__(self, origin, rot, dimensions):#l, w, h):
        
        self.origin = origin
        self.z_rot = rot[-1]
        self.l = dimensions[0]
        self.w = dimensions[1]
        self.h = dimensions[2]
        self.vertices = self.getVertices()

    def getVertices(self):
        x = np.array([-self.w/2 , -self.w/2 , self.w/2 , self.w/2])
        y = np.array([self.l/2, -self.l/2, -self.l/2, self.l/2])
        vertices = np.stack((x,y), axis=1)
        c, s = np.cos(self.z_rot), np.sin(self.z_rot)
        R = np.array(((c, -s), (s, c)))

        for vertex in range(len(vertices)):
            vertices[vertex] = np.dot(R, vertices[vertex])
            vertices[vertex][0] += self.origin[0]
            vertices[vertex][1] += self.origin[1]

        return vertices

    def is_within_point(self, point):
        is_within_square = self.test_point_2d(point[0], point[1])
        z_min, z_max = self.origin[-1] - self.h/2, self.origin[-1] + self.h/2
        is_withing_z = z_max > point[-1] > z_min 
        return is_within_square and is_withing_z

    def is_on_right_side(self, x, y, xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        a = float(y1 - y0)
        b = float(x0 - x1)
        c = - a*x0 - b*y0
        return a*x + b*y + c >= 0

    def test_point_2d(self, x, y):
        num_vert = len(self.vertices)
        is_right = [self.is_on_right_side(x, y, self.vertices[i], self.vertices[(i + 1) % num_vert]) for i in range(num_vert)]
        all_left = not any(is_right)
        all_right = all(is_right)
        return all_left or all_right
    
    def is_within_body(self, bodyId):
        pos, _ = p.getBasePositionAndOrientation(bodyId)
        return self.is_within_point(pos)

    def update(self):
        self.vertices = self.getVertices()

class MassSensor():

    def __init__(self, fwdId, origin_offset=[0,0,1], triggerVolDim=[5, 2, 1.5], excludedBodiesIds=[]):
        self.fwdId  = fwdId
        self.excludedBodiesIds = excludedBodiesIds
        self.excludedBodiesIds.append(fwdId)
        self.base_pos, self.base_ori = p.getBasePositionAndOrientation(self.fwdId)
        self.base_pos = np.array(self.base_pos)
        self.base_ori = np.array(p.getEulerFromQuaternion(self.base_ori))
        self.origin = self.getOriginFromBase(origin_offset)
        self.triggerVol = TriggerVolume(self.origin, self.base_ori, triggerVolDim)
        self.aabb = p.getAABB(fwdId)
        self.aabb_min = self.aabb[0]
        self.aabb_max = self.aabb[1]

    def getOriginFromBase(self, offset=[0,0,0]):
        offset = np.array(offset)
        return self.base_pos + offset

    def getMass(self):
        #Update pos and rotation of trigger volume in case it changed
        mass = 0
        self.triggerVol.update()
        overlappingObjs=p.getOverlappingObjects(self.aabb_min, self.aabb_max)
        for _,obj in enumerate(overlappingObjs):
            if(obj[0] not in self.excludedBodiesIds):
                if(self.triggerVol.is_within_body(obj[0])):
                    dynInfo = p.getDynamicsInfo(obj[0], -1)
                    mass += dynInfo[0]
        return mass
        

class WoodPile2():

    def __init__(self, initialWoodPos, initialWoodRot, layerDim, nLayers, offset):
        self.initialWoodPos = initialWoodPos.copy()
        self.initialWoodRot = initialWoodRot.copy()
        self.layerDim = layerDim
        self.nLayers = nLayers
        self.offset = offset
        self.f_name = os.path.join(os.path.dirname(__file__), "wood.urdf")
        self.radius = .25
        self.length = 4
        self.meshScale = [1,1,1]

        self.shift = [0, -0.02, 0]
        self.shift1 = [0, 0.1, 0]
        self.shift2 = [0, 0, 0]

        self.visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                         #halfExtents=[[0, 0, 0], [0.1, 0.1, 0.1]],
                                         radius=self.radius ,
                                         length=self.length,
                                         #fileName="meshes/wood.obj",
                                         visualFramePosition=[
                                             self.shift1,
                                             self.shift2,
                                         ],
                                         meshScale=self.meshScale,
                                         )

        self.collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                radius=self.radius,
                                                height=self.length,
                                                collisionFramePosition=[
                                                   self.shift1,
                                                   self.shift2,
                                                ],
                                                meshScale=self.meshScale)
        #texUid = p.loadTexture("/Users/ilyakurinov/Downloads/Forwarder/Forwarder/meshes/textures/wood_bark_tex.jpg")
        self.wood_list = self.createWoodPile()

    def createWoodPile(self):
        wood_list = list()
        layer_old = 0
        for layer in range(self.nLayers):
            for _ in range(self.layerDim):
                mb = p.createMultiBody(baseMass=400,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseCollisionShapeIndex=self.collisionShapeId,
                                        baseVisualShapeIndex=self.visualShapeId,
                                        basePosition=self.initialWoodPos,
                                        baseOrientation=p.getQuaternionFromEuler(self.initialWoodRot),
                                        useMaximalCoordinates=False)
                p.changeVisualShape(mb, -1, rgbaColor=[0.54, .34, .129, 1])
                wood_list.append(mb)
                self.initialWoodPos[1] += self.offset
                if(layer_old != layer):
                    self.initialWoodPos[1] = 1
                    layer_old = layer
                p.changeDynamics(mb, -1, rollingFriction=0.008)

            self.initialWoodPos[2] += self.offset
        return mb


class WoodPile():

    def __init__(self, initialWoodPos, initialWoodRot, layerDim, nLayers, offset):
        self.initialWoodPos = initialWoodPos.copy()
        self.initialWoodRot = initialWoodRot.copy()
        self.layerDim = layerDim
        self.nLayers = nLayers
        self.offset = offset
        self.f_name = os.path.join(os.path.dirname(__file__), "wood.urdf")
        self.wood_list = self.createWoodPile()

    def createWoodPile(self):
        layer_old = 0
        wood_list = list()

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
                wood_list.append(wood)
                self.initialWoodPos[1] += self.offset
                if(layer_old != layer):
                    self.initialWoodPos[1] = 1
                    layer_old = layer
            self.initialWoodPos[2] += self.offset
        return wood_list