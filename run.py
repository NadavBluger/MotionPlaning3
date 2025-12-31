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
from building_blocks import BuildingBlocks3D
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
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E1", goal_prob=0.2, k=5, step_size=None)

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)
def run_2d_rrt_star_motion_planning():
    MAP_DETAILS = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal": np.array([0.3, 0.15, 1.0, 1.1]),
    }
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(
        bb=bb,
        start=MAP_DETAILS["start"],
        goal=MAP_DETAILS["goal"],
        ext_mode="E2",
        goal_prob=0.5,
        max_step_size=0.1,
    )
    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_motion_planning():
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=.2)
    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_inspection_planning():
    MAP_DETAILS = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E2", goal_prob=0.01, coverage=0.5)

    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"])

def run_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.1 )

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------

    rrt_star_planner = RRTStarPlanner(max_step_size=0.5,
                                      start=env2_start,
                                      goal=env2_goal,
                                      max_itr=4000,
                                      stop_on_goal=True,
                                      bb=bb,
                                      goal_prob=0.05,
                                      ext_mode="E2")

    path = rrt_star_planner.plan()

    if path is not None:

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
        np.save(os.path.join(exp_folder_name, 'path'), path)

        # save the cost of the path and time it took to compute
        with open(os.path.join(exp_folder_name, 'stats'), "w") as file:
            file.write("Path cost: {} \n".format(rrt_star_planner.compute_cost(path)))

        visualizer.show_path(path)

def run_all_trials_and_plot(num_trials_per_setting):
    """
    Runs trials for all four settings:
      (E1, 5%), (E1, 20%), (E2, 5%), (E2, 20%)
    """


    color_map = {
        ("E1", 0.05): "blue",
        ("E1", 0.20): "cyan",
        ("E2", 0.05): "red",
        ("E2", 0.20): "orange",
    }

    results = {cfg: {"times": [], "costs": []} for cfg in [("E1", 0.05), ("E1", 0.20), ("E2", 0.05), ("E2", 0.20)]}

    # Run trials
    for ext_mode, goal_prob in [("E1", 0.05), ("E1", 0.20), ("E2", 0.05), ("E2", 0.20)]:
        for _ in range(num_trials_per_setting):
            t, c = run_2d_rrt_motion_planning_trial(ext_mode, goal_prob)
            results[(ext_mode, goal_prob)]["times"].append(t)
            results[(ext_mode, goal_prob)]["costs"].append(c)

    # Plot all outcomes together
    plt.figure()
    for (ext_mode, goal_prob), data in results.items():
        plt.scatter(
            data["times"],
            data["costs"],
            color=color_map[(ext_mode, goal_prob)],
            label=f"{ext_mode}, {int(goal_prob*100)}%"
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Cost")
    plt.title("RRT outcomes: time vs cost (all runs)")
    plt.legend()
    plt.show()

    # OVERALL stats across ALL runs
    all_times = []
    all_costs = []
    for data in results.values():
        all_times.extend(data["times"])
        all_costs.extend(data["costs"])

    all_times = np.array(all_times, dtype=float)
    all_costs = np.array(all_costs, dtype=float)

    print("\n=== OVERALL STATISTICS (all iterations combined) ===")
    print(f"Total runs: {len(all_times)}")
    print(f"Average time (overall): {all_times.mean():.4f} seconds")
    print(f"Standard deviation of time (overall): {all_times.std(ddof=1):.4f} seconds")
    print(f"Average cost (overall): {all_costs.mean():.4f}")
    print(f"Standard deviation of cost (overall): {all_costs.std(ddof=1):.4f}")

    return results


def run_trials_2d_manipulator():
    # 5 runs per setting -> 4 settings -> 20 total outcomes
    return run_all_trials_and_plot(num_trials_per_setting=10)
def run_2d_rrt_motion_planning_trial(ext_mode, goal_probs):
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=goal_probs)

    t0 = time.perf_counter()
    plan = planner.plan()
    t1 = time.perf_counter()
    print("complete iter")
    return (t1 - t0), planner.compute_cost(plan)


if __name__ == "__main__":
    #run_trials_2d_manipulator()
    # run_dot_2d_astar()
   # run_dot_2d_rrt()
    #run_dot_2d_rrt_star()
    run_2d_rrt_motion_planning()
    #run_2d_rrt_inspection_planning()
    # run_2d_rrt_star_motion_planning()
    #run_3d()