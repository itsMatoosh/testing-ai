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

    # just the basic implementation
    min_distance = float('inf')
    for frame in time_series:
        if frame["crashed"]:
            return {"crash_count": 1, "min_distance": 0.0}
        else:
            ego_pos = frame["ego"]["pos"]
            min_distance = float('inf')
            for other in frame["others"]:
                other_pos = other["pos"]
                distance = np.sqrt((ego_pos[0] - other_pos[0])**2 + (ego_pos[1] - other_pos[1])**2)
                if distance < min_distance:
                    min_distance = distance
    return {"crash_count": 0, "min_distance": min_distance}


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

    # I hate this definition of fitness, fitness means up is better, loss is smaller better
    if objectives["crash_count"] >= 1:
        return -1.0  # best possible fitness
    else:
        return objectives["min_distance"]  # smaller is better


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

    # instead of hard coding std maybe do 0.5 * stagnated_iterations?
    # stagnated iterations could be smuggled in via param_spec if desired
    new_spacing = rng.normal(cfg_copy['initial_spacing'], 1)

    # np.clip?
    if new_spacing < param_spec['initial_spacing']['min']:
        new_spacing = param_spec['initial_spacing']['min']
    elif new_spacing > param_spec['initial_spacing']['max']:
        new_spacing = param_spec['initial_spacing']['max']

    cfg_copy['initial_spacing'] = new_spacing
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
    end = False
    max_iterations = iterations
    i = 1
    while not end:
        print(f"Iteration {i}, best fitness so far: {best_fit}")
        i += 1
        neighbours = []
        for _ in range(neighbors_per_iter):
            nn = mutate_config(best_cfg, param_spec, rng)
            neighbours.append(nn)

        # neighbours.append(best_cfg)  # elitism
        for n in neighbours:
            crashed, ts = run_episode(env_id, n, policy, defaults, seed_base)
            obj = compute_objectives_from_time_series(ts)
            cur_fit = compute_fitness(obj)
            print(f"nn spacing: {n['initial_spacing']}, fitness: {cur_fit}, obj: {obj}")
            if cur_fit < best_fit:
                best_fit = cur_fit
                best_cfg = copy.deepcopy(n)
                best_obj = dict(obj)
                if obj["crash_count"] >= 1:
                    break

        history.append(best_fit)
        if i >= max_iterations or best_obj["crash_count"] >= 1:
            end = True


    return {
          "best_cfg": best_cfg,
          "best_objectives": best_obj,
          "best_fitness": best_fit,
          "best_seed_base": seed_base,
          "history": history
        }
