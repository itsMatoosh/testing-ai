"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from envs.highway_env_utils import run_episode

from search.base_search import sample_random_config



# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """

    # our objectives include:crash count, min distance to any vehicle, risk objective (for how long was the agent in risk), minimal lane-local distance
    crash_count = 0
    min_distance = float('inf')
    min_lane_distance = float('inf')
    risk_sum = 0.0

    eps = 1e-3  # numerical stability

    for frame in time_series:
        if frame["crashed"]:
            crash_count = 1
            break

        ego = frame["ego"]
        ego_pos = ego["pos"]
        ego_lane = ego["lane_id"]

        frame_min_dist = float('inf')

        for other in frame["others"]:
            other_pos = other["pos"]

            distance = np.sqrt(
                (ego_pos[0] - other_pos[0]) ** 2 +
                (ego_pos[1] - other_pos[1]) ** 2
            )

            # global minimum distance (over all time)
            if distance < min_distance:
                min_distance = distance

            # lane-local minimum distance
            if other["lane_id"] == ego_lane:
                if distance < min_lane_distance:
                    min_lane_distance = distance

            # frame-local minimum (for risk)
            if distance < frame_min_dist:
                frame_min_dist = distance

        # accumulate risk over time (how long agent is close to others)
        risk_sum += 1.0 / (frame_min_dist + eps)

    if crash_count == 1:
        min_distance = 0.0
        min_lane_distance = 0.0

    return {
        "crash_count": crash_count,
        "min_distance": min_distance,
        "risk_sum": risk_sum,
        "min_lane_distance": min_lane_distance
    }


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """

    # fitness now takes into consideration: crash_count and min_distance (as before) + uses risk_sum and optionally min_lane_distance with little weight
    if objectives["crash_count"] >= 1:
        return -1e6  #not -1 because no guarantee that non-crashing will always be >=-1

    # small weights chosen to keep min_distance dominant
    w_risk = 0.05
    w_lane = 0.1

    return (
            objectives["min_distance"]
            - w_risk * objectives["risk_sum"]
            + w_lane * objectives["min_lane_distance"]
    )


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """

    cfg_copy = copy.deepcopy(cfg)

    # decide how many parameters to mutate
    num_mutations = rng.integers(1, 4)

    # choose parameters to mutate
    mutable_params = list(param_spec.keys())
    params_to_mutate = rng.choice(
        mutable_params, size=num_mutations, replace=False
    )
    for param in params_to_mutate:
        spec = param_spec[param]
        current_value = cfg_copy[param]

        if spec["type"] == "float":
            span = spec["max"] - spec["min"]
            # 70% small step, 30% large step
            if rng.random() < 0.7:
                sigma = 0.1 * span
            else:
                sigma = 0.4 * span
            new_value = rng.normal(current_value, sigma)

        elif spec["type"] == "int":
            step = rng.choice([-1, 1])
            new_value = current_value + step

        else:
            continue

        new_value = np.clip(new_value, spec["min"], spec["max"])

        if spec["type"] == "int":
            if param == "vehicles_count":
                step = rng.integers(3, 8) * rng.choice([-1, 1])
                new_value = current_value + step

            elif param == "lanes_count":
                step = rng.choice([-2, -1, 1, 2])
                new_value = current_value + step

            elif param == "initial_lane_id":
                # relocate ego vehicle agressively
                new_value = rng.integers(0, cfg_copy["lanes_count"])
                cfg_copy[param] = int(new_value)
                continue
            else:
                step = rng.choice([-1, 1])
                new_value = current_value + step

        #debug print statmement
        cfg_copy[param] = new_value

    # if lanes_count changed, ensure the initial_lane_id is valid
    if "lanes_count" in params_to_mutate:
        cfg_copy["initial_lane_id"] = int(
            np.clip(
                cfg_copy["initial_lane_id"],
                0,
                cfg_copy["lanes_count"] - 1
            )
        )

    return cfg_copy

# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)

    # TODO (students): choose initialization (base_cfg or random scenario)
    current_cfg = sample_random_config(rng, param_spec, base_cfg)

    # Evaluate initial solution (seed_base used for reproducibility)
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base

    history = [best_fit]

    print(f"Initial configuration: {best_cfg['initial_spacing']}" )

    # TODO (students): implement HC loop
    # - generate neighbors
    # - evaluate
    # - pick best
    # - accept if improved
    # - early stop on crash (optional)
    stagnation_counter = 0
    stagnation_limit = 3
    for i in range(1, iterations + 1):
        print(f"Iteration {i}, best fitness so far: {best_fit}")

        best_neighbor_cfg = None
        best_neighbor_obj = None
        best_neighbor_fit = cur_fit

        # Generate and evaluate neighbors
        for _ in range(neighbors_per_iter):
            nn = mutate_config(current_cfg, param_spec, rng)

            seed_eval = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, nn, policy, defaults, seed_eval)

            obj = compute_objectives_from_time_series(ts)
            fit = compute_fitness(obj)

            print(f"nn spacing: {nn['initial_spacing']}, fitness: {fit}, obj: {obj}")

            if fit <= best_neighbor_fit:
                best_neighbor_cfg = nn
                best_neighbor_fit = fit
                best_neighbor_obj = obj
        improved_global_best = False
        # accept move if improvement or plateau
        if best_neighbor_cfg is not None:
            current_cfg = copy.deepcopy(best_neighbor_cfg)
            cur_fit = best_neighbor_fit

            if cur_fit <= best_fit:
                best_cfg = copy.deepcopy(current_cfg)
                best_fit = cur_fit
                best_obj = dict(best_neighbor_obj)
                best_seed_base = seed_eval
                improved_global_best = True
        if improved_global_best:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        if stagnation_counter >= stagnation_limit:
            print("refreshing search since no improvement")

            current_cfg = sample_random_config(rng, param_spec, base_cfg)

            seed_eval = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_eval)
            obj = compute_objectives_from_time_series(ts)
            cur_fit = compute_fitness(obj)

            stagnation_counter = 0

        history.append(best_fit)

        # Early stop on crash
        if best_obj["crash_count"] >= 1:
            break

    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "history": history
    }
