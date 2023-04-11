import pybullet as p
import os
import math
import numpy as np

class Bobcat:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'bobcat-330s.urdf')
        self.bobcat = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0.1],
                              physicsClientId=client, flags=p.URDF_USE_INERTIA_FROM_FILE)

        # Joint indices as found by p.getJointInfo()
        self.left_joints = [2,3]
        self.right_joints = [0,1]
        # Joint speed
        self.left_joint_speed = 0
        self.right_joint_speed = 0
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the bobcat
        self.c_throttle = 20

    def get_ids(self):
        return self.bobcat, self.client

    def apply_action(self, action):
        # Expects action to be two dimensional
        left_throttle, right_throttle = action

        # Clip throttle and steering angle to reasonable values
        left_throttle = left_throttle * 4.5
        right_throttle = right_throttle * 4.5

        for joint in self.left_joints:
            p.setJointMotorControl2(self.bobcat, joint,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_throttle)
        
        for joint in self.right_joints:
            p.setJointMotorControl2(self.bobcat, joint,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_throttle)


        '''# Calculate drag / mechanical resistance ourselves
        # Using velocity control, as torque control requires precise models
        left_friction = -self.left_joint_speed * (self.left_joint_speed * self.c_drag +
                                        self.c_rolling)
        left_acceleration = self.c_throttle * left_throttle + left_friction
        # Each time step is 1/240 of a second
        self.left_joint_speed = self.left_joint_speed + 1/30 * left_acceleration
        if self.left_joint_speed < 0:
            self.left_joint_speed = 0

        # Set the velocity of the wheel joints directly
        p.setJointMotorControlArray(
            bodyUniqueId=self.bobcat,
            jointIndices=self.left_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.left_joint_speed] * 2,
            forces=[1.2] * 2,
            physicsClientId=self.client)

        # Calculate drag / mechanical resistance ourselves
        # Using velocity control, as torque control requires precise models
        right_friction = -self. right_joint_speed * (self.right_joint_speed * self.c_drag +
                                        self.c_rolling)
        right_acceleration = self.c_throttle *  right_throttle +  right_friction
        # Each time step is 1/240 of a second
        self. right_joint_speed = self. right_joint_speed + 1/30 *  right_acceleration
        if self.right_joint_speed < 0:
            self. right_joint_speed = 0

        # Set the velocity of the wheel joints directly
        p.setJointMotorControlArray(
            bodyUniqueId=self.bobcat,
            jointIndices=self. right_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self. right_joint_speed] * 2,
            forces=[1.2] * 2,
            physicsClientId=self.client)'''

    def get_observation(self):
        # Get the position and orientation of the bobcat in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.bobcat, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        #print(np.array(pos)/10)
        # Get the velocity of the bobcat
        vel = p.getBaseVelocity(self.bobcat, self.client)[0][0:2]
        # Concatenate position, orientation, velocity
        observation = (pos + ori + vel)

        return observation









