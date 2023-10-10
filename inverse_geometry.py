#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
import time

from tools import setcubeplacement

DT = 1e-2

# Still coliding maybe move left and right hands
def computeqgrasppose(robot: RobotWrapper, qcurrent, cube: RobotWrapper, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    LEFT_HAND_ID = robot.model.getFrameId(LEFT_HAND)
    RIGHT_HAND_ID = robot.model.getFrameId(RIGHT_HAND)
    setcubeplacement(robot, cube, cubetarget)
    cube_LH = getcubeplacement(cube, LEFT_HOOK)
    cube_RH = getcubeplacement(cube, RIGHT_HOOK)
    q = qcurrent.copy()
    # herr = []
    for _ in range(1000): 
        # Run the algorithms that outputs values in robot.data
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        # Placement from world frame o to frame f oMtool  
        oMleft = robot.data.oMf[LEFT_HAND_ID]
        o_Jleft = pin.computeFrameJacobian(robot.model, robot.data, q, LEFT_HAND_ID, pin.LOCAL)
        left_nu = pin.log(oMleft.inverse() * cube_LH).vector

        # Placement from world frame o to frame f oMtool  
        oMright  = robot.data.oMf[RIGHT_HAND_ID]
        o_Jright = pin.computeFrameJacobian(robot.model, robot.data, q, RIGHT_HAND_ID, pin.LOCAL)
        right_nu = pin.log(oMright.inverse() * cube_RH).vector

        vq = pinv(o_Jleft) @ left_nu
        Ptool = np.eye(robot.nv) - pinv(o_Jleft) @ o_Jleft
        vq += pinv(o_Jright @ Ptool) @ (right_nu - o_Jright @ vq)

        q = pin.integrate(robot.model,q, vq * DT)
        # viz.display(q)
        # herr.append((left_nu, right_nu))
    
    # print(herr[-1])
    return q, True
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6000")
    
    q = robot.q0.copy()
    
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend  = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    # updatevisuals(viz, robot, cube, q0)
    updatevisuals(viz, robot, cube, qe)
    print(collision(robot, qe))
    
    
    
