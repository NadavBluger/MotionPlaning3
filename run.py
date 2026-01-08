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
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.2, k=None, max_step_size=15)

    # execute plan
    plan, _ = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)
def run_2d_rrt_star_motion_planning(i):
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
            ext_mode="E2",
            goal_prob=i,
            max_step_size=0.1,
            stop_on_goal=False,
            k=None
        )
        # execute plan
        plan, costs = planner.plan()
        print(plan)
        print(planner.compute_cost(plan))
        costs_results.append(costs)
    if len(plan):
        Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])
    with open(f"costs_{i}", mode='w')as f:
        f.write(json.dumps(costs_results))

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
    #visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    env2_start = np.deg2rad([110,-70, 90, -90, -90, 0])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------

    rrt_star_planner = RRTStarPlanner(max_step_size=max_step_size,
                                      start=env2_start,
                                      goal=env2_goal,
                                      max_itr=2000,
                                      stop_on_goal=False,
                                      bb=bb,
                                      goal_prob=p_bias,
                                      ext_mode="E2")
    paths, costs = rrt_star_planner.plan()

    if paths is not None:

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
        np.save(os.path.join(exp_folder_name, 'path'), paths)
        np.save(os.path.join(exp_folder_name, 'costs'), costs)


        # save the cost of the path and time it took to compute
        with open(os.path.join(exp_folder_name, 'stats'), "w") as file:
            file.write("Path cost: {} \n".format(rrt_star_planner.compute_cost(paths)))

        #time.sleep(10)
        #print("showing path")
        #visualizer.show_path(paths)
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
                max_itr=2000,
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

def get_best_exp_dir():
    """
    Iterates through the 'exps' directory, reads 'stats' files, and returns the
    directory path that contains the smallest non-zero path cost.
    """
    exps_folder_name = os.path.join(os.getcwd(), "exps")
    if not os.path.exists(exps_folder_name):
        return None

    min_cost = float('inf')
    best_dir = None
    for dirname in os.listdir(exps_folder_name):
        dir_path = os.path.join(exps_folder_name, dirname)
        if os.path.isdir(dir_path):
            stats_path = os.path.join(dir_path, "stats")
            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    content = f.read()
                    if "Path cost:" in content:
                        try:
                            cost = float(content.split("Path cost:")[1].strip())
                            if cost > 0 and cost < min_cost:
                                min_cost = cost
                                best_dir = dir_path
                        except ValueError:
                            pass
    return best_dir

def plot_results():
    """
    Parses the 'exps' directory and generates Cost vs Iteration and Success Rate vs Iteration
    graphs for each p_bias value, comparing different max_step_sizes.
    """
    exps_folder = os.path.join(os.getcwd(), "exps")
    if not os.path.exists(exps_folder):
        print("No 'exps' folder found.")
        return

    # Structure: results[p_bias][max_step_size] = list of numpy arrays (costs)
    results = {}

    for dirname in os.listdir(exps_folder):
        dir_path = os.path.join(exps_folder, dirname)
        if not os.path.isdir(dir_path):
            continue
        
        try:
            parts = dirname.split('_')
            # Format: exp_pbias_<val>_max_step_size_<val>_<timestamp>
            if 'pbias' in parts and 'size' in parts:
                p_bias = float(parts[parts.index('pbias') + 1])
                max_step_size = float(parts[parts.index('size') + 1])
                
                costs_path = os.path.join(dir_path, 'costs.npy')
                if os.path.exists(costs_path):
                    # Load (N, 2) array: [[iteration, cost], ...]
                    costs_data = np.load(costs_path, allow_pickle=True)
                    
                    if p_bias not in results:
                        results[p_bias] = {}
                    if max_step_size not in results[p_bias]:
                        results[p_bias][max_step_size] = []
                    results[p_bias][max_step_size].append(costs_data)
        except Exception:
            continue

    # Generate Plots
    for p_bias in sorted(results.keys()):
        fig_cost, ax_cost = plt.subplots(figsize=(10, 5))
        fig_succ, ax_succ = plt.subplots(figsize=(10, 5))
        
        for step_size in sorted(results[p_bias].keys()):
            runs = results[p_bias][step_size]
            if not runs: continue
            
            # Get all unique iteration checkpoints
            iterations = sorted(list(set(row[0] for run in runs for row in run)))
            avg_costs, success_rates = [], []
            
            for it in iterations:
                # Extract cost at this iteration for each run if it exists
                costs_at_it = [run[run[:, 0] == it][0, 1] for run in runs if run[run[:, 0] == it].size > 0]
                # Filter for valid paths (cost > 0)
                valid_costs = [c for c in costs_at_it if c is not None and c > 0]
                
                success_rates.append(len(valid_costs) / len(runs) * 100)
                avg_costs.append(np.mean(valid_costs) if valid_costs else np.nan)
            
            ax_cost.plot(iterations, avg_costs, marker='.', label=f'Step Size {step_size}')
            ax_succ.plot(iterations, success_rates, marker='.', label=f'Step Size {step_size}')
        
        for ax, title in zip([ax_cost, ax_succ], ["Cost", "Success Rate"]):
            ax.set_title(f"{title} vs Iteration (p_bias={p_bias})")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(f"Average {title}" if title == "Cost" else "Success Rate (%)")
            ax.legend(); ax.grid(True)
        plt.show()

def plot_cost_vs_time_from_files():
    """
    Reads costs_0.05 and costs_0.2 files and plots Cost vs Time.
    """
    for bias in [0.05, 0.2]:
        filename = f"costs_{bias}"
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            continue
        
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filename}")
                continue
        
        plt.figure(figsize=(10, 6))
        for i, run in enumerate(data):
            if not run:
                continue
            run_arr = np.array(run)
            # run_arr is (N, 2) -> time, cost
            plt.plot(run_arr[:, 0], run_arr[:, 1], marker='.', label=f'Run {i+1}')
        
        plt.xlabel("Time (s)")
        plt.ylabel("Cost")
        plt.title(f"Cost vs Time (Goal Bias {int(bias*100)}%)")
        plt.legend()
        plt.grid(True)
        plt.show()

def visualize_representative_2d_final():
    """
    Runs one representative instance for each goal bias and visualizes it.
    """
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    
    for bias in [0.05, 0.2]:
        print(f"Running representative plan for bias {bias}...")
        planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=bias, max_step_size=0.1, stop_on_goal=False)
        plan, _ = planner.plan()
        if len(plan) > 0:
            Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

if __name__ == "__main__":
    # dot_tree_figures_all()
    # run_dot_2d_astar()
    # run_dot_2d_rrt()
    # run_dot_2d_rrt_star()
    # run_2d_rrt_motion_planning()
    # run_2d_rrt_inspection_planning()
    # for i in [0.05, 0.20]:
    #     run_2d_rrt_star_motion_planning(i)
    plot_cost_vs_time_from_files()
    visualize_representative_2d_final()
