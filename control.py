#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pinocchio as pin
from bezier import Bezier
from config import LEFT_HAND, RIGHT_HAND
from tools import setupwithpybullet, rununtil
from config import DT
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose
from path import computepath
from config import LEFT_HAND
import matplotlib.pyplot as plt

# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.0  # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)  # derivative gain (D of PD)


def plot_bezier_diff(trajs, path, total_time):
    LEFT_HAND_ID = robot.model.getFrameId(LEFT_HAND)

    def get_xyz_from_q(q, frame_id):
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.forwardKinematics(robot.model, robot.data, q)
        return robot.data.oMf[frame_id].translation.copy()

    world_goal_traj_left = np.array([get_xyz_from_q(q, LEFT_HAND_ID) for q in path])
    world_bezier_traj_left = np.array(
        [
            get_xyz_from_q(trajs[0](t), LEFT_HAND_ID)
            for t in np.linspace(0, total_time, 65)
        ]
    )
    # Plot them 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        world_goal_traj_left[:, 0],
        world_goal_traj_left[:, 1],
        world_goal_traj_left[:, 2],
        label="Goal (L)",
        color="red",
    )
    ax.plot(
        world_bezier_traj_left[:, 0],
        world_bezier_traj_left[:, 1],
        world_bezier_traj_left[:, 2],
        label="Bezier (L)",
        color="blue",
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.legend()
    plt.show()


def calculate_curve_diff(trajs, path):
    LEFT_HAND_ID = robot.model.getFrameId(LEFT_HAND)

    def get_xyz_from_q(q, frame_id):
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.forwardKinematics(robot.model, robot.data, q)
        return robot.data.oMf[frame_id].translation.copy()

    world_goal_traj_left = np.array([get_xyz_from_q(q, LEFT_HAND_ID) for q in path])
    world_bezier_traj_left = np.array(
        [
            get_xyz_from_q(trajs[0](t), LEFT_HAND_ID)
            for t in np.linspace(0, total_time, len(path))
        ]
    )

    np.linalg.norm(world_goal_traj_left - world_bezier_traj_left)


def controllaw(sim, robot, trajs, tcurrent):
    q, vq = sim.getpybulletstate()

    ref_q = trajs[0](tcurrent)
    ref_vq = trajs[1](tcurrent)
    ref_vvq = trajs[2](tcurrent)

    # These are vectors
    dq = q - ref_q
    dvq = vq - ref_vq

    b = pin.rnea(robot.model, robot.data, q, vq, ref_vvq)
    # compute mass matrix M
    M = pin.crba(robot.model, robot.data, q)

    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)

    LEFT_HAND_ID = robot.model.getFrameId(LEFT_HAND)
    o_Jleft = pin.computeFrameJacobian(
        robot.model, robot.data, q, LEFT_HAND_ID, pin.LOCAL
    )

    RIGHT_HAND_ID = robot.model.getFrameId(RIGHT_HAND)
    o_Jright = pin.computeFrameJacobian(
        robot.model, robot.data, q, RIGHT_HAND_ID, pin.LOCAL
    )

    f_c = np.array([0, -50, 30, 0, 0, 0])

    desired_vvq = ref_vvq - Kp * dq - Kv * dvq
    torques = M @ desired_vvq + b + (o_Jleft.T + o_Jright.T) @ f_c

    sim.step(torques)


def maketraj(path, T):
    q_of_t = Bezier(path, t_max=T)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)
    return q_of_t, vq_of_t, vvq_of_t


if __name__ == "__main__":
    robot, sim, cube = setupwithpybullet()
    q0, successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe, successend = computeqgrasppose(
        robot, robot.q0, cube, CUBE_PLACEMENT_TARGET, None
    )
    q_path, se3_path = computepath(
        q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube
    )

    # setting initial configuration
    sim.setqsim(q0)

    total_time = 4.0
    trajs = maketraj(q_path, total_time)

    plot_bezier_diff(trajs, q_path, total_time)

    tcur = 0.0
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur)
        tcur += DT
