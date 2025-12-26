import numpy as np
from RRTTree import RRTTree
import time


class RRTMotionPlanner(object):

    def __init__(self, bb, ext_mode, goal_prob, start, goal):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        self.tree.add_vertex(self.start)
        while not self.tree.is_goal_exists(self.goal):
            rand_config = self.bb.sample_random_config(goal_prob=self.goal_prob, goal=self.goal)
            self.extend(self.tree.get_nearest_config(rand_config)[1], rand_config)

        plan = np.empty((0, 2))
        current = self.tree.get_vertex_for_config(self.goal)
        while (current.config!=self.start).all():
            plan=np.vstack((plan, current.config))
            current =self.tree.vertices[self.tree.edges[self.tree.get_idx_for_config(current.config)]]
        plan=np.vstack((plan, self.start))
        return plan

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        #maybe need to return just the cost of the goal vertex
        return sum(vertex.cost for vertex in plan)


    def extend(self, near_config: np.ndarray, rand_config: np.ndarray):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        #E1
        if self.ext_mode == "E1":
            if not self.bb.env.config_validity_checker(rand_config) or not self.bb.env.edge_validity_checker(
                    near_config, rand_config):
                return
            eid = self.tree.add_vertex(rand_config)
            sid = self.tree.get_idx_for_config(near_config)
            self.tree.add_edge(sid, eid, self.bb.compute_distance(near_config, rand_config))
        else:
            new_config = near_config + ((rand_config - near_config) / self.bb.compute_distance(rand_config, near_config))*0.5
            if not self.bb.env.config_validity_checker(new_config) or not self.bb.env.edge_validity_checker(
                    near_config, new_config):
                return
            eid = self.tree.add_vertex(new_config)
            sid = self.tree.get_idx_for_config(near_config)
            self.tree.add_edge(sid, eid, self.bb.compute_distance(new_config, near_config))

