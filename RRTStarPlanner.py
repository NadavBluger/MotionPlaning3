import numpy as np
from RRTTree import RRTTree
import time


class RRTStarPlanner(object):

    def __init__(
        self,
        bb,
        ext_mode,
        step_size,
        start,
        goal,
        max_itr=None,
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
        self.increment=5

        self.max_step_size = step_size

    def plan(self):
        """
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        """
        self.tree.add_vertex(self.start)
        while not self.tree.is_goal_exists(self.goal):
            rand_config = self.bb.sample_random_config(goal_prob=self.goal_prob, goal=self.goal)
            self.extend(self.tree.get_nearest_config(rand_config)[1], rand_config)

        current = self.tree.get_vertex_for_config(self.goal)
        plan = [current.config]
        while np.any(current.config != self.start):
            current = self.tree.vertices[self.tree.edges[self.tree.get_idx_for_config(current.config)]]
            plan.append(current.config)

        return np.array(plan)[::-1]

    def compute_cost(self, plan):
        cost = 0
        for i in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return cost

    def extend(self, x_near, x_rand):
        if self.ext_mode == "E1":
            if not self.bb.config_validity_checker(x_rand) or not self.bb.edge_validity_checker(
                    x_near, x_rand):
                return
            eid = self.tree.add_vertex(x_rand)
            sid = self.tree.get_idx_for_config(x_near)
            self.tree.add_edge(sid, eid, self.bb.compute_distance(x_near, x_rand))
            new_config = x_rand
        else:
            if self.bb.compute_distance(x_near, self.goal) < self.increment:
                new_config = self.goal
            else:
                new_config = x_near + ((x_rand - x_near) / self.bb.compute_distance(x_rand, x_near)) * self.increment
            if not self.bb.config_validity_checker(new_config) or not self.bb.edge_validity_checker(x_near, new_config):
                return

            eid = self.tree.add_vertex(new_config)
            sid = self.tree.get_idx_for_config(x_near)
            self.tree.add_edge(sid, eid, self.bb.compute_distance(new_config, x_near))
        nearest_neighbors_ids, nearest_neighbors_configs = self.tree.get_k_nearest_neighbors(new_config, k=min(self.k, len(self.tree.vertices)-1))
        for parent_id, parent_config in zip(nearest_neighbors_ids, nearest_neighbors_configs):
            if self.bb.edge_validity_checker(parent_config, new_config):
                c = self.bb.compute_distance(parent_config, new_config)
                if self.tree.get_vertex_for_config(parent_config).cost+c < self.tree.get_vertex_for_config(new_config).cost:
                    self.tree.edges[eid] = parent_id
                    self.tree.vertices[eid].set_cost(self.tree.vertices[parent_id].cost+c)

