import math
from math import inf

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
        max_itr=4000,
        stop_on_goal=False,
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
        self.rewires=0

        self.max_step_size = step_size

    def plan(self):
        """
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        """
        self.tree.add_vertex(self.start)
        itrs=0
        costs = []
        cost=inf
        start = time.time()
        #while not (self.tree.is_goal_exists(self.goal) and self.stop_on_goal) and itrs <self.max_itr:
        while not (self.tree.is_goal_exists(self.goal) and self.stop_on_goal) and time.time()-start < 30.5:
            rand_config = self.bb.sample_random_config(self.goal_prob, self.goal)
            # print(rand_config)
            self.extend(self.tree.get_nearest_config(rand_config)[1], rand_config)
            itrs+=1
            if self.tree.is_goal_exists(self.goal) and self.compute_cost(self.get_path()) <cost:
                cost = self.compute_cost(self.get_path())
                costs.append((time.time()-start, cost))
                print(costs[-1])
            # if itrs %200 ==0:
            #     paths.append(self.get_path())
        print(time.time()-start)
        print(costs)
        print(len(self.tree.edges))
        print(f"{self.rewires=}")
        return self.get_path(), costs

    def get_path(self):
        if self.tree.is_goal_exists(self.goal):
            current = self.tree.get_vertex_for_config(self.goal)
        else:
            return []
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
            if self.bb.compute_distance(x_near, x_rand) < self.max_step_size:
                new_config = x_rand
            else:
                new_config = x_near + ((x_rand - x_near) / self.bb.compute_distance(x_rand, x_near)) * self.max_step_size
            if not self.bb.config_validity_checker(new_config):
                return
            if not self.bb.edge_validity_checker(x_near, new_config):
                return

            eid = self.tree.add_vertex(new_config)
            sid = self.tree.get_idx_for_config(x_near)
            self.tree.add_edge(sid, eid, self.bb.compute_distance(new_config, x_near))
        if self.k is None:
            i = len(self.tree.vertices)
            d = len(new_config)
            k = (5*d*int((math.log(i)/i)**(1/d)))
            k = k if k > 1 else 1
        else:
            k = min(self.k, len(self.tree.vertices)-1)
        nearest_neighbors_ids, _ = self.tree.get_k_nearest_neighbors(new_config, k=k)
        for parent_id in nearest_neighbors_ids:
            self.rewire(parent_id, eid)
        for parent_id in nearest_neighbors_ids:
            self.rewire(eid, parent_id)

    def rewire(self, pp_id, n_id):
        pp = self.tree.vertices[pp_id].config
        n = self.tree.vertices[n_id].config
        c = self.bb.compute_distance(pp,n)
        if self.tree.vertices[pp_id].cost+c < self.tree.vertices[n_id].cost:
            if self.bb.edge_validity_checker(pp, n):
                self.tree.edges[n_id] = pp_id
                self.tree.vertices[n_id].set_cost(self.tree.vertices[pp_id].cost+c)
                self.propagate_cost_to_children(n_id)
                self.rewires +=1
                print(self.compute_cost(self.get_path()))

    def propagate_cost_to_children(self, parent_id):
        for child_id, pid in self.tree.edges.items():
            if pid == parent_id:
                #print(child_id, pid, parent_id)
                dist = self.bb.compute_distance(self.tree.vertices[parent_id].config, self.tree.vertices[child_id].config)
                self.tree.vertices[child_id].set_cost(self.tree.vertices[parent_id].cost + dist)
                self.propagate_cost_to_children(child_id)
