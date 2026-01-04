import random

import numpy as np
from RRTTree import RRTTree
import time


class RRTInspectionPlanner(object):

    def __init__(self, bb, start, ext_mode, goal_prob, coverage):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb, task="ip")
        self.start = start

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage

        # set step size - remove for students
        self.step_size = min(self.bb.env.xlimit[-1] / 50, self.bb.env.ylimit[-1] / 200)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        self.tree.add_vertex(self.start, self.bb.get_inspected_points(self.start))
        #TODO: make near_conf the conf with the most seen points not the nearest (that is nearest? maybe with E1 that's not needed) or maybe only when config sampling make it so
        best_config = np.array([0,0,-np.pi/2,0])
        best_coverage = self.bb.compute_coverage(self.bb.get_inspected_points(best_config))
        while self.tree.max_coverage < self.coverage:
            rand_config = self.bb.sample_random_config(self.goal_prob, best_config)
            #self.extend(self.tree.get_nearest_config(rand_config)[1], rand_config)
            self.extend(self.tree.vertices[self.tree.max_coverage_id].config, rand_config)
            already_inspected = self.tree.vertices[self.tree.max_coverage_id].inspected_points
            new_points = self.bb.get_inspected_points(rand_config)
            if len(already_inspected) == 0:
                combined_points = new_points
            elif len(new_points) == 0:
                combined_points = already_inspected
            else:
                combined_points = np.unique(np.vstack((already_inspected, new_points)), axis=0)
            if (self.bb.compute_coverage(combined_points) > best_coverage):
                best_coverage = self.bb.compute_coverage(combined_points)
                best_config = rand_config
            #print(self.tree.max_coverage, best_coverage)

        current = self.tree.vertices[self.tree.max_coverage_id]
        plan = [current.config]
        while np.any(current.config != self.start):
            current = self.tree.vertices[self.tree.edges[self.tree.get_idx_for_config(current.config)]]
            plan.append(current.config)

        return np.array(plan)[::-1]

    def get_goal(self):
        while True:
            goal = random.choice(self.bb.env.inspection_points)
            if not len(self.tree.vertices[self.tree.max_coverage_id].inspected_points):
                return self.bb.compute_inverse_kinematics(goal)
            if goal not in self.tree.vertices[self.tree.max_coverage_id].inspected_points:
                return self.bb.compute_inverse_kinematics(goal)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        cost = 0
        for i in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        if self.ext_mode == "E1":
            if not self.bb.config_validity_checker(rand_config) or not self.bb.edge_validity_checker(
                    near_config, rand_config):
                return
            sid = self.tree.get_idx_for_config(near_config)
            new_inspected_points = self.bb.compute_union_of_points(
                self.bb.get_inspected_points(rand_config), self.tree.vertices[sid].inspected_points)
            eid = self.tree.add_vertex(rand_config, new_inspected_points)
            self.tree.add_edge(sid, eid, self.bb.compute_distance(near_config, rand_config))
        else:
            if self.bb.compute_distance(rand_config, near_config) < self.step_size:
                new_config = rand_config
            else:
                new_config = near_config + ((rand_config - near_config) / self.bb.compute_distance(rand_config, near_config)) * self.step_size
            if not self.bb.config_validity_checker(new_config) or not self.bb.edge_validity_checker(
                    near_config, new_config):
                return

            sid = self.tree.get_idx_for_config(near_config)
            new_inspected_points = self.bb.compute_union_of_points(
                self.bb.get_inspected_points(new_config), self.tree.vertices[sid].inspected_points)
            eid = self.tree.add_vertex(new_config, new_inspected_points)

            self.tree.add_edge(sid, eid, self.bb.compute_distance(new_config, near_config))
