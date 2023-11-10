from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, position: np.ndarray, parent: Node = None, q=None):
        self.position = position
        self.parent = parent
        self.q = q

    def distance(self, other):
        return np.linalg.norm(self.position - other.position)

    def step(self, target: Node, epsilon, collision_f: callable) -> Node:
        if self.distance(target) < epsilon:
            new_node = Node(target.position, self)
        else:
            dir = target.position - self.position
            dir = dir / np.linalg.norm(dir) * epsilon  # Make dir length epsilon
            new_pos = self.position + dir
            new_node = Node(new_pos, self)

        if collision_f(new_node):
            return None
        else:
            return new_node

    def __hash__(self):
        return hash(self.position.tobytes())

    def __eq__(self, other) -> bool:
        return np.all(self.position == other.position)

    def __str__(self):
        return str(self.position)

    def path_to_root(self) -> list[Node]:
        if self.parent is None:
            return [self]
        else:
            return [self] + self.parent.path_to_root()


class Tree:
    def __init__(self):
        self.nodes = []
        self.edges = dict()

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_edge(self, u: Node, v: Node) -> None:
        # u -> v
        if u in self.edges:
            self.edges[u].append(v)
        else:
            self.edges[u] = [v]
        # v -> u
        if v in self.edges:
            self.edges[v].append(u)
        else:
            self.edges[v] = [u]

    def nearestNeighbor(self, q: Node) -> Node:
        min_dist = np.inf
        nearest = None
        for node in self.nodes:
            dist = q.distance(node)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    def plot(self, ax, vis_3d: bool = False, show: bool = False) -> None:
        pos = np.array(list(map(lambda n: n.position, self.nodes)))
        if vis_3d:
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
        else:
            ax.scatter(pos[:, 0], pos[:, 1])
        for key, value in self.edges.items():
            for v in value:
                if vis_3d:
                    ax.plot(
                        [key.position[0], v.position[0]],
                        [key.position[1], v.position[1]],
                        [key.position[2], v.position[2]],
                    )
                else:
                    ax.plot(
                        [key.position[0], v.position[0]],
                        [key.position[1], v.position[1]],
                    )
        if show:
            plt.show()

    @staticmethod
    def combine(tree_a: Tree, tree_b: Tree) -> Tree:
        tree = Tree()
        for node in tree_a.nodes:
            tree.add_node(node)
        for node in tree_b.nodes:
            if node not in tree.nodes:
                tree.add_node(node)
        for u, v in tree_a.edges.items():
            for w in v:
                if not u in tree.edges or w not in tree.edges[u]:
                    tree.add_edge(u, w)
        for u, v in tree_b.edges.items():
            for w in v:
                if not u in tree.edges or w not in tree.edges[u]:
                    tree.add_edge(u, w)
        return tree


class RRT_CONNECT:
    def __init__(self, start: Node, goal: Node, vis_3d: bool = True):
        self.start = start
        self.goal = goal
        self.path = []
        # Creating the trees
        self.tree_start = Tree()
        self.tree_start.add_node(self.start)
        self.tree_end = Tree()
        self.tree_end.add_node(self.goal)
        self.vis_3d = vis_3d
        self.ax = (
            plt.figure().add_subplot(projection="3d")
            if self.vis_3d
            else plt.figure().add_subplot()
        )

    @staticmethod
    def extend(tree: Tree, q: Node, collision_f: callable, step_size: float = 0.1):
        q_near = tree.nearestNeighbor(q)
        q_new = q_near.step(q, step_size, collision_f)
        if q_new is not None and np.all(q_new.position == q.position):
            tree.add_node(q_new)
            tree.add_edge(q_new, q_near)
            state = "Reached"
        elif q_new is None:
            state = "Trapped"
        else:
            tree.add_node(q_new)
            tree.add_edge(q_new, q_near)
            state = "Advanced"
        return q_new, state

    @staticmethod
    def connect(
        tree: Tree, q: Node, collision_f: callable, step_size: float
    ) -> Tuple[Node, str]:
        q_new, s = RRT_CONNECT.extend(tree, q, collision_f, step_size)
        while s == "Advanced":
            q_new, s = RRT_CONNECT.extend(tree, q, collision_f, step_size)
        return q_new, s

    def plan(
        self,
        collision_f: callable,
        step_size: float = 0.1,
        max_iter: int = 1000,
        rand_f: callable = None,
        vis=False,
    ) -> list[Node]:
        if vis:
            plt.ion()
            plt.show()
        for k in range(max_iter):
            tree_1 = self.tree_start if k % 2 == 0 else self.tree_end
            tree_2 = self.tree_end if k % 2 == 0 else self.tree_start
            extend_start = k % 2 == 0
            q_rand = rand_f(extend_start)
            q_new_1, state_1 = RRT_CONNECT.extend(
                tree_1, q_rand, collision_f, step_size
            )
            if not state_1 == "Trapped":
                q_new_2, state_2 = self.connect(tree_2, q_new_1, collision_f, step_size)
                if state_2 == "Reached":
                    path_1 = q_new_1.path_to_root()
                    path_2 = q_new_2.path_to_root()
                    if k % 2 == 0:  # path is from tree_start, path_2 from tree_end
                        self.path = list(reversed(path_1)) + path_2
                    else:  # path is from tree_end, path_2 from tree_start
                        self.path = list(reversed(path_2)) + path_1

                    if vis:
                        self.plot()
                        plt.pause(5)
                    break
            if k > 1 and k % 10 == 0:
                if vis:
                    self.plot()
        return self.path

    def plot(self, vis_path: bool = False) -> None:
        self.tree_start.plot(self.ax, self.vis_3d)
        self.tree_end.plot(self.ax, self.vis_3d)
        if vis_path:
            path = np.array(list(map(lambda n: n.position, self.path)))
            if self.vis_3d:
                self.ax.plot(path[:, 0], path[:, 1], path[:, 2], "ro-")
            else:
                self.ax.plot(path[:, 0], path[:, 1], "ro-")
        print(len(self.tree_start.nodes))
        print(len(self.tree_end.nodes))
        plt.draw()
        plt.pause(0.1)


def collission_f(node: Node) -> bool:
    position = node.position
    return (
        position[0] > 4.5 and position[0] < 5.5 and position[1] < 1 and position[1] > -1
    )


def rand_f(extend_start: bool) -> Node:
    return Node(np.array([np.random.uniform(4, 6), np.random.uniform(-2, 2)]))


if __name__ == "__main__":
    start = Node(np.array([0, 0]))
    goal = Node(np.array([10, 0]))

    rrt_con = RRT_CONNECT(start, goal, False)

    rrt_con.plan(collission_f, 0.1, 1000, rand_f, vis=True)
