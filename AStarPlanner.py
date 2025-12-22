import numpy as np
import heapq


class AStarPlanner(object):
    def __init__(self, bb, start, goal):
        self.bb = bb
        self.start = start
        self.goal = goal

        self.nodes = dict()

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []

        # define all directions the agent can take - order doesn't matter here
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]

        self.epsilon = 1
        plan, cost = self.a_star(self.start, self.goal)
        return np.array(plan)

    # compute heuristic based on the planning_env
    def compute_heuristic(self, state):
        '''
        Return the heuristic function for the A* algorithm.
        @param state The state (position) of the robot.
        '''
        #TODO i actually did f function calculation noe heuristic, need to split
        return self.bb.env.compute_distance(state, self.goal)

    def a_star(self, start_loc, goal_loc):
        start_node = tuple(start_loc)
        goal_node = tuple(goal_loc)

        open_set = []
        
        # Calculate initial f-score
        # compute_heuristic returns g + h. We extract h to apply epsilon.
        h = self.compute_heuristic(start_node)
        f_score = 0 + self.epsilon * h
        
        heapq.heappush(open_set, (f_score, start_node))

        came_from = {}
        g_score = {start_node: 0}
        expanded_set = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current in expanded_set:
                continue

            expanded_set.add(current)
            self.expanded_nodes.append(np.array(current))

            if current == goal_node:
                path = []
                while current in came_from:
                    path.append(np.array(current))
                    current = came_from[current]
                path.append(np.array(start_node))
                return path[::-1], g_score[goal_node]

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if not self.bb.env.config_validity_checker(neighbor):
                    continue

                if not self.bb.env.edge_validity_checker(current, neighbor):
                    continue

                tentative_g_score = g_score[current] + self.bb.env.compute_distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    # Calculate f-score
                    h = self.compute_heuristic(neighbor)
                    f_score = tentative_g_score + self.epsilon * h
                    
                    heapq.heappush(open_set, (f_score, neighbor))

        return [], 0
