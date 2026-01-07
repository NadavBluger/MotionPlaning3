import math
from sysconfig import get_path

import numpy as np
from RRTTree import RRTTree
import time

from threeD.building_blocks import BuildingBlocks3D


class RRTStarPlanner(object):

    def __init__(
        self,
        bb:BuildingBlocks3D,
        ext_mode,
        max_step_size,
        start,
        goal,
        max_itr=5000,
        stop_on_goal=None,
        k=None,
        goal_prob=0.01,
    ):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.k = k

        self.max_step_size = max_step_size

    def plan(self):
        """
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        """
        self.tree.add_vertex(self.start)
        itrs=0
        costs = []
        cost = math.inf
        while itrs < self.max_itr:
            goal_prob=0 if self.tree.is_goal_exists(self.goal) else self.goal_prob
            rand_config = self.bb.sample_random_config(goal_prob, self.goal)
            sid, nearest_config = self.tree.get_nearest_config(rand_config)
            if self.bb.config_validity_checker(rand_config) and self.bb.edge_validity_checker(nearest_config, rand_config):
                new_config = self.extend(nearest_config, rand_config)
                eid = self.tree.add_vertex(new_config)
                new_cost = self.bb.compute_distance(nearest_config, new_config) + self.tree.vertices[sid].cost
                self.tree.add_edge(sid, eid, new_cost)


                nearest_neighbors, _ = self.tree.get_k_nearest_neighbors(new_config, self.get_k())

                #Parent rewire
                potential_parents= []
                for nearest_neighbor in nearest_neighbors:
                    if self.bb.edge_validity_checker(new_config, self.tree.vertices[nearest_neighbor].config):
                        new_cost = self.bb.compute_distance(new_config, self.tree.vertices[nearest_neighbor].config) + self.tree.vertices[nearest_neighbor].cost
                        potential_parents.append((nearest_neighbor,new_cost))
                if potential_parents:
                    potential_parents.sort(key=lambda x: x[1])
                    new_parent = potential_parents[0]
                    if new_parent[1] < self.tree.vertices[eid].cost:
                        self.tree.vertices[eid].cost = new_parent[1]
                        self.tree.edges[eid] = new_parent[0]

                #Child rewire
                potential_children = []
                for nearest_neighbor in nearest_neighbors:
                    if self.bb.edge_validity_checker(new_config, self.tree.vertices[nearest_neighbor].config):
                        new_cost = self.bb.compute_distance(new_config, self.tree.vertices[nearest_neighbor].config) + \
                                   self.tree.vertices[eid].cost
                        potential_children.append((nearest_neighbor, new_cost))
                if potential_children:
                    potential_children.sort(key=lambda x: x[1])
                    new_child = potential_children[0]
                    if new_child[1] < self.tree.vertices[eid].cost:
                        self.tree.vertices[new_child[0]].cost = new_child[1]
                        self.tree.edges[new_child] = eid
            itrs +=1
            # if self.tree.is_goal_exists(self.goal) and self.compute_cost(self.get_path()) < cost:
            #     cost = self.compute_cost(self.get_path())
            #     costs.append((itrs, cost))
            #     print(costs[-1])
            if itrs%200==0:
                cost =self.compute_cost(self.get_path())
                costs.append((itrs, cost))
                print(costs[-1])
        return self.get_path(), costs

    def get_path(self):
        if not self.tree.is_goal_exists(self.goal):
            return []
        current = self.tree.get_vertex_for_config(self.goal)
        path = [current.config]
        while self.tree.get_idx_for_config(current.config) in self.tree.edges:
            current = self.tree.vertices[self.tree.edges[self.tree.get_idx_for_config(current.config)]]
            path.append(current.config)
        return np.array(path)[::-1]


    def compute_cost(self, plan):
        cost = 0
        for i in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return cost

    def extend(self, x_near, x_rand):
        if self.bb.compute_distance(x_near, x_rand) < self.max_step_size:
            new_conf = x_rand
        else:
            new_conf = x_near + ((x_rand-x_near)/self.bb.compute_distance(x_near, x_rand)) * self.max_step_size
        return np.array(new_conf)
    def get_k(self):
        if self.k:
            return  min(self.k, len(self.tree.vertices)-1)
        else:
            i = len(self.tree.vertices)
            d = len(self.goal)
            # k = e^(1+1/d)*log i
            k = int(math.exp(1 + 1 / d) * math.log(i))
            k = k if k > 1 else 1
            return min(k, len(self.tree.vertices) - 1)