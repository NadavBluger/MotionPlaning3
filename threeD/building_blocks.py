import numpy as np
import random
import math

class BuildingBlocks3D(object):
    """
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    """

    def __init__(self, transform, ur_params, env, resolution=0.1):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        self.obstacles_np = np.array(self.env.obstacles)
        
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechamical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [
            ["shoulder_link", "forearm_link"],
            ["shoulder_link", "wrist_1_link"],
            ["shoulder_link", "wrist_2_link"],
            ["shoulder_link", "wrist_3_link"],
            ["upper_arm_link", "wrist_1_link"],
            ["upper_arm_link", "wrist_2_link"],
            ["upper_arm_link", "wrist_3_link"],
            ["forearm_link", "wrist_2_link"],
            ["forearm_link", "wrist_3_link"],
        ]

    def sample_random_config(self, goal_prob, goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        if random.random() < goal_prob:
            return goal_conf
        return [random.uniform(limit[0], limit[1]) for limit in self.ur_params.mechamical_limits.values()]

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return False if in collision
        @param conf - some configuration
        """
        spheres = self.transform.conf2sphere_coords(conf)
        
        # Convert to numpy arrays and check x > 0.4 constraint
        spheres_arrays = {}
        for name, link_spheres in spheres.items():
            S = np.array(link_spheres)
            if np.any(S[:, 0] > 0.4):
                return False
            spheres_arrays[name] = S

        # link link collision
        for plc in self.possible_link_collisions:
            name1, name2 = plc
            S1 = spheres_arrays[name1]
            S2 = spheres_arrays[name2]
            r1 = self.ur_params.sphere_radius[name1]
            r2 = self.ur_params.sphere_radius[name2]
            
            # Vectorized distance check
            dists_sq = np.sum((S1[:, None, :] - S2[None, :, :]) ** 2, axis=-1)
            if np.any(dists_sq < (r1 + r2) ** 2):
                return False

        # link obstacle collision
        if len(self.obstacles_np) > 0:
            obs_r = self.env.radius
            for name, S in spheres_arrays.items():
                r = self.ur_params.sphere_radius[name]
                dists_sq = np.sum((S[:, None, :] - self.obstacles_np[None, :, :]) ** 2, axis=-1)
                if np.any(dists_sq < (obs_r + r) ** 2):
                    return False

        # link floor collision
        for name, S in spheres_arrays.items():
            if name == "shoulder_link":
                continue
            if np.any(S[:, 2] < self.ur_params.sphere_radius[name]):
                return False

        return True


    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        """check for collisions between two configurations - return True if transition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        """
        #length = self.compute_distance(current_conf, prev_conf)
        length = np.linalg.norm(current_conf - prev_conf)
        amount = max(int(length / self.resolution), 2)
        current = prev_conf.copy()
        increment = (current_conf - prev_conf) / amount
        for i in range(amount + 1):
            if self.config_validity_checker(current):
                current += increment
            else:
                return False
        return True

    def compute_distance(self, conf1, conf2):
        """
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        """
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5
