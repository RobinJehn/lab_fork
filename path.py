#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv
from pinocchio.utils import rotate
from rrt import RRT_CONNECT, Node
from tools import setcubeplacement
from inverse_geometry import computeqgrasppose

from config import LEFT_HAND, RIGHT_HAND
import time

# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal, robot, cube):
    rrt = RRT_CONNECT(
        Node(cubeplacementq0.translation, None, qinit),
        Node(cubeplacementqgoal.translation, None, qgoal),
    )
    path = rrt.plan(
        collision_f(robot, qinit, cube), 0.01, 1000, sampleCubePlacements, True
    )
    se3_path = list(map(lambda n: pin.SE3(rotate("z", 0.0), n.position), path))
    q_path = np.array(
        list(
            map(
                lambda cube_placement: computeqgrasppose(
                    robot, qinit, cube, cube_placement
                )[0],
                se3_path,
            )
        )
    )

    return q_path, se3_path


def collision_f(robot, q_current, cube):
    def f(node: Node) -> bool:
        cube_placement = pin.SE3(rotate("z", 0.0), node.position)
        q_init = (
            node.parent.q
            if node.parent is not None and node.parent.q is not None
            else q_current
        )
        q, sucess = computeqgrasppose(robot, q_init, cube, cube_placement)
        node.q = q

        # cube_size = [0.1, 0.1, 0.1]
        # obstacle_size = [0.3, 0.03, 0.12]
        # obstacle_pos = [0.43, -0.1, 0.94]
        # cube_pos = node.position

        # potential_x_collision = abs(obstacle_pos[0] - cube_pos[0]) < (cube_size[0] + obstacle_size[0]) / 2
        # potential_y_collision = abs(obstacle_pos[1] - cube_pos[1]) < (cube_size[1] + obstacle_size[1]) / 2
        # potential_z_collision = abs(obstacle_pos[2] - cube_pos[2]) < (cube_size[2] + obstacle_size[2]) / 2
        # manual_collision_check = potential_x_collision and potential_y_collision and potential_z_collision

        return not sucess  # or manual_collision_check

    return f


def displaypath(robot, robot_path, cube_path, dt, viz):
    for q, se3 in zip(robot_path, cube_path):
        setcubeplacement(robot, cube, se3)
        viz.display(q)
        time.sleep(dt)


def sampleCubePlacements(start: bool) -> Node:
    # Idea, sample a lot around the obstacle because
    # Obstacle is along x axis
    # Sample just above the obstacle with enough distance for the cube
    # xrange = (0.3, 0.6)
    # zrange = (1.1, 1.2)
    # yrange = (-0.3, 0.1)
    xrange = (0.3, 0.5)
    zrange = (1.06 - 0.12, 1.2 - 0.12)
    if start:
        yrange = (-0.3, -0.3)
    else:
        yrange = (0.1, 0.1)
    x = np.random.uniform(xrange[0], xrange[1])
    y = np.random.uniform(yrange[0], yrange[1])
    z = np.random.uniform(zrange[0], zrange[1])

    return Node(np.array([x, y, z]))


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")

    q_path, se3_path = computepath(
        q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, robot, cube
    )
    displaypath(
        robot, q_path, se3_path, dt=0.2, viz=viz
    )  # you ll probably want to lower dt
