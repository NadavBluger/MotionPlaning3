import json

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from twoD.environment import MapEnvironment
from twoD.dot_environment import MapDotEnvironment
from twoD.dot_building_blocks import DotBuildingBlocks2D
from twoD.building_blocks import BuildingBlocks2D
from twoD.dot_visualizer import DotVisualizer
from threeD.environment import Environment
from threeD.kinematics import UR5e_PARAMS, Transform
from threeD.building_blocks import BuildingBlocks3D
from threeD.visualizer import Visualize_UR
from AStarPlanner import AStarPlanner
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner
from RRTStarPlanner import RRTStarPlanner
from twoD.visualizer import Visualizer

# MAP_DETAILS = {"json_file": "twoD/map1.json", "start": np.array([10,10]), "goal": np.array([4, 6])}
MAP_DETAILS = {"json_file": "twoD/map2.json", "start": np.array([360, 150]), "goal": np.array([100, 200])}


def run_dot_2d_astar():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = AStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, expanded_nodes=planner.expanded_nodes, show_map=True, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_dot_2d_rrt():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E1", goal_prob=0.20)

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_dot_2d_rrt_star():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.2, k=10, step_size=15)

    # execute plan
    plan,_ = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)
def run_2d_rrt_star_motion_planning():
    MAP_DETAILS = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal": np.array([0.3, 0.15, 1.0, 1.1]),
    }
    costs_results=[]
    for _ in range(10):
        planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
        bb = BuildingBlocks2D(planning_env)
        planner = RRTStarPlanner(
            bb=bb,
            start=MAP_DETAILS["start"],
            goal=MAP_DETAILS["goal"],
            ext_mode="E1",
            goal_prob=0.2,
            step_size=0.05,
            stop_on_goal=False,
            k=10
        )
        # execute plan
        plan, costs = planner.plan()
        costs_results.append(costs)
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_motion_planning():
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.01)
    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_inspection_planning():
    MAP_DETAILS = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0]),
                   "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E2", goal_prob=0.01, coverage=0.5)

    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"])

def run_3d(max_step_size, p_bias ):
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.1)
    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    env2_start = np.deg2rad([110,-70, 90, -90, -90, 0])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------

    rrt_star_planner = RRTStarPlanner(step_size=max_step_size,
                                      start=env2_start,
                                      goal=env2_goal,
                                      max_itr=2000,
                                      stop_on_goal=False,
                                      bb=bb,
                                      goal_prob=p_bias,
                                      ext_mode="E2")
    paths = rrt_star_planner.plan()

    if paths[-1] is not None:

        # create a folder for the experiment
        # Format the time string as desired (YYYY-MM-DD_HH-MM-SS)
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # create the folder
        exps_folder_name = os.path.join(os.getcwd(), "exps")
        if not os.path.exists(exps_folder_name):
            os.mkdir(exps_folder_name)
        exp_folder_name = os.path.join(exps_folder_name, "exp_pbias_"+ str(rrt_star_planner.goal_prob) + "_max_step_size_" + str(rrt_star_planner.max_step_size) + "_" + time_str)
        if not os.path.exists(exp_folder_name):
            os.mkdir(exp_folder_name)

        # save the path
        np.save(os.path.join(exp_folder_name, 'path'), paths[-1])

        # save the cost of the path and time it took to compute
        with open(os.path.join(exp_folder_name, 'stats'), "w") as file:
            file.write("Path cost: {} \n".format(rrt_star_planner.compute_cost(paths[-1])))

        time.sleep(10)
        print("showing path")
        visualizer.show_path(paths[-1])
        return paths

def run_trials_2d_manipulator(ext_mode, goal_prob, trials=10, k=5, step_size= 5):
    MAP_DETAILS = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal":  np.array([0.3,  0.15, 1.0, 1.1]),
    }

    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)

    costs = []
    times = []

    for i in range(trials):
        planner = RRTStarPlanner(
            bb=bb,
            ext_mode=ext_mode,
            step_size=step_size,
            start=MAP_DETAILS["start"],
            goal=MAP_DETAILS["goal"],
            max_itr=20000,
            stop_on_goal=True,
            k=k,
            goal_prob=goal_prob,
        )

        t0 = time.perf_counter()
        plan = planner.plan()
        t1 = time.perf_counter()

        exec_time = t1 - t0
        cost = planner.compute_cost(plan)

        times.append(exec_time)
        costs.append(cost)

        # Representative visualization for each parameter combo: save/attach manually if needed
        # easiest: visualize the FIRST run for each combo outside this loop (see below)

    times = np.array(times, dtype=float)
    costs = np.array(costs, dtype=float)

    return {
        "ext_mode": ext_mode,
        "goal_prob": goal_prob,
        "times": times,
        "costs": costs,
        "mean_time": float(times.mean()),
        "std_time": float(times.std(ddof=1)),   # sample stdev
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std(ddof=1)),   # sample stdev
    }
def experiment_2d_manipulator_all():
    trials = 10
    k = 5
    step_size = 5

    results = []

    for ext_mode in ["E1", "E2"]:
        r5  = run_trials_2d_manipulator(ext_mode, 0.05, trials=trials, k=k, step_size=step_size)
        r20 = run_trials_2d_manipulator(ext_mode, 0.20, trials=trials, k=k, step_size=step_size)
        results.append((r5, r20))
        print("here")

        # Print report
        print(f"\n==== 2D Manipulator | extend={ext_mode} ====")
        print(f"Goal bias 5% :  mean time={r5['mean_time']:.4f}s  stdev={r5['std_time']:.4f}s | "
              f"mean cost={r5['mean_cost']:.4f}  stdev={r5['std_cost']:.4f}")
        print(f"Goal bias 20%: mean time={r20['mean_time']:.4f}s stdev={r20['std_time']:.4f}s | "
              f"mean cost={r20['mean_cost']:.4f} stdev={r20['std_cost']:.4f}")

        # Scatter plot (20 points) for this extend mode
        plt.figure()
        plt.scatter(r5["times"],  r5["costs"],  label="goal bias 5%")
        plt.scatter(r20["times"], r20["costs"], label="goal bias 20%")
        plt.xlabel("Time to solution (s)")
        plt.ylabel("Path cost")
        plt.title(f"RRT* outcomes (extend={ext_mode})")
        plt.legend()
        plt.show()

    return results
def visualize_representatives_2d_manipulator():
    MAP_DETAILS = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal":  np.array([0.3,  0.15, 1.0, 1.1]),
    }
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)

    for ext_mode in ["E1", "E2"]:
        for goal_prob in [0.05, 0.20]:
            planner = RRTStarPlanner(
                bb=bb,
                ext_mode=ext_mode,
                step_size=0.1,
                start=MAP_DETAILS["start"],
                goal=MAP_DETAILS["goal"],
                max_itr=20000,
                stop_on_goal=True,
                k=5,
                goal_prob=goal_prob,
            )
            plan = planner.plan()
            Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])


def dot_tree_figures_all():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)

    for ext_mode in ["E1", "E2"]:
        for goal_prob in [0.05, 0.20]:
            planner = RRTStarPlanner(
                bb=bb,
                start=MAP_DETAILS["start"],
                goal=MAP_DETAILS["goal"],
                ext_mode=ext_mode,
                goal_prob=goal_prob,
                k=5,
                step_size=None
            )
            plan = planner.plan()
            DotVisualizer(bb).visualize_map(
                plan=plan,
                tree_edges=planner.tree.get_edges_as_states(),
                show_map=True
            )
if __name__ == "__main__":
    # dot_tree_figures_all()
    # run_dot_2d_astar()
    # run_dot_2d_rrt()
    run_dot_2d_rrt_star()
    # run_2d_rrt_motion_planning()
    # run_2d_rrt_inspection_planning()
    # run_2d_rrt_star_motion_planning()
    # res =dict()
    # for p in [0.05, 0.2]:
    #     for m in [0.05, 0.075, 0.1, 0.125, 0.2, 0.25,0.3,0.4]:
    #         for i in range(20):
    #             print(f"{p=}, {m=} {i=}")
    #             res[f"{p}_{m}_{i}"]=run_3d(m, p)
    #
    # with open("res") as f:
    #     json.dump(res, f)
