"""
Benchmark script for search methods.

Runs a specified search method X times and records how many iterations
it took to find a crash for each run. Outputs results as both a console
table and CSV file.

Usage:
    python benchmark.py --method hill_climb --runs 10 --output results.csv
    python benchmark.py --method random --runs 5 --iterations 100
    python benchmark.py --method hill_climb --runs 10 --parallel 4  # Run 4 in parallel
"""

import argparse
import csv
import sys
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from functools import partial

import numpy as np

from config.search_space import param_spec, base_cfg


# ============================================================
# Crash Iteration Detection
# ============================================================

CRASH_FITNESS = -1e6  # From hill_climbing.py compute_fitness()


def find_crash_iteration_from_history(history: List[float]) -> Optional[int]:
    """
    Find the iteration where crash occurred by looking for crash fitness in history.
    Returns iteration number (0-indexed) or None if no crash found.
    """
    for i, fitness in enumerate(history):
        if fitness <= CRASH_FITNESS:
            return i
    return None


# ============================================================
# Worker Functions for Parallel Execution
# ============================================================

def _worker_hill_climb(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel hill_climb execution.
    Loads policy/env in each worker to avoid pickling issues.
    """
    run_number, seed, iterations, neighbors_per_iter, total_runs = args
    
    # Import and load in worker process
    from policies.pretrained_policy import load_pretrained_policy
    from envs.highway_env_utils import make_env
    from search.hill_climbing import hill_climb
    
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    _, defaults = make_env(env_id)
    
    print(f"[Worker {run_number}] Starting hill_climb (seed={seed})...")
    
    result = hill_climb(
        env_id=env_id,
        base_cfg=base_cfg,
        param_spec=param_spec,
        policy=policy,
        defaults=defaults,
        seed=seed,
        iterations=iterations,
        neighbors_per_iter=neighbors_per_iter,
    )
    
    crash_found = result["best_objectives"]["crash_count"] >= 1
    crash_iteration = find_crash_iteration_from_history(result["history"])
    
    if crash_found:
        print(f"[Worker {run_number}] CRASH FOUND at iteration {crash_iteration}")
    else:
        print(f"[Worker {run_number}] NO CRASH (max iterations reached)")
    
    return {
        "run": run_number,
        "crash_found": crash_found,
        "iterations_to_crash": crash_iteration,
        "seed": seed,
    }


def _worker_random_search(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel random search execution.
    """
    run_number, seed, n_scenarios, n_eval, total_runs = args
    
    # Import and load in worker process
    from policies.pretrained_policy import load_pretrained_policy
    from envs.highway_env_utils import make_env, run_episode
    from search.random_search import RandomSearch
    
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    _, defaults = make_env(env_id)
    
    print(f"[Worker {run_number}] Starting RandomSearch (seed={seed})...")
    
    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    rng = np.random.default_rng(seed)
    
    for scenario_num in range(1, n_scenarios + 1):
        cfg = search.sample_random_config(rng)
        
        for j in range(n_eval):
            s = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, cfg, policy, defaults, s)
            
            if crashed:
                print(f"[Worker {run_number}] CRASH FOUND at scenario {scenario_num}")
                return {
                    "run": run_number,
                    "crash_found": True,
                    "iterations_to_crash": scenario_num,
                    "seed": seed,
                }
    
    print(f"[Worker {run_number}] NO CRASH (max scenarios reached)")
    return {
        "run": run_number,
        "crash_found": False,
        "iterations_to_crash": None,
        "seed": seed,
    }


# ============================================================
# Sequential Benchmark Runner Functions
# ============================================================

def run_hill_climb_benchmark(
    env_id: str,
    policy,
    defaults: Dict[str, Any],
    seed: int,
    iterations: int,
    neighbors_per_iter: int,
    run_number: int,
    total_runs: int,
) -> Tuple[bool, Optional[int]]:
    """
    Run hill climbing using the existing implementation (sequential).
    """
    from search.hill_climbing import hill_climb
    
    print(f"Running hill_climb (seed={seed}, iterations={iterations}, neighbors={neighbors_per_iter})...")
    
    result = hill_climb(
        env_id=env_id,
        base_cfg=base_cfg,
        param_spec=param_spec,
        policy=policy,
        defaults=defaults,
        seed=seed,
        iterations=iterations,
        neighbors_per_iter=neighbors_per_iter,
    )
    
    crash_found = result["best_objectives"]["crash_count"] >= 1
    crash_iteration = find_crash_iteration_from_history(result["history"])
    
    if crash_found:
        print(f">>> Run {run_number}/{total_runs} complete: CRASH FOUND at iteration {crash_iteration}")
    else:
        print(f">>> Run {run_number}/{total_runs} complete: NO CRASH (max iterations reached)")
    
    return crash_found, crash_iteration


def run_random_search_benchmark(
    env_id: str,
    policy,
    defaults: Dict[str, Any],
    seed: int,
    n_scenarios: int,
    n_eval: int,
    run_number: int,
    total_runs: int,
) -> Tuple[bool, Optional[int]]:
    """
    Run random search (sequential).
    """
    from envs.highway_env_utils import run_episode
    from search.random_search import RandomSearch
    
    print(f"Running RandomSearch (seed={seed}, scenarios={n_scenarios}, n_eval={n_eval})...")
    
    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    rng = np.random.default_rng(seed)
    
    for scenario_num in range(1, n_scenarios + 1):
        print(f"\rScenario {scenario_num}/{n_scenarios}...", end='', flush=True)
        
        cfg = search.sample_random_config(rng)
        
        for j in range(n_eval):
            s = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, cfg, policy, defaults, s)
            
            if crashed:
                print(f"\r>>> Run {run_number}/{total_runs} complete: CRASH FOUND at scenario {scenario_num}          ")
                return True, scenario_num
    
    print(f"\r>>> Run {run_number}/{total_runs} complete: NO CRASH (max scenarios reached)          ")
    return False, None


# ============================================================
# Output Functions
# ============================================================

def print_results_table(results: List[Dict[str, Any]], method: str) -> None:
    """Print a frequency table of evaluations to find a crash."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    # Sort results by run number for display
    results = sorted(results, key=lambda x: x["run"])
    
    # Count frequencies of each iteration count
    iterations_list = [r["iterations_to_crash"] for r in results if r["iterations_to_crash"] is not None]
    no_crash_count = sum(1 for r in results if r["iterations_to_crash"] is None)
    
    freq_counter = Counter(iterations_list)
    sorted_evals = sorted(freq_counter.keys())
    
    # Determine label based on method
    eval_label = "iterations" if method == "hill_climb" else "scenarios"
    
    # Header
    print(f"+{'-' * 40}+{'-' * 12}+")
    print(f"| {'Number of ' + eval_label + ' to find a crash':^38} | {'Frequency':^10} |")
    print(f"+{'-' * 40}+{'-' * 12}+")
    
    # Data rows (sorted by evaluation count)
    for eval_count in sorted_evals:
        freq = freq_counter[eval_count]
        print(f"| {eval_count:^38} | {freq:^10} |")
    
    # Add row for "No crash found" if applicable
    if no_crash_count > 0:
        print(f"| {'N/A (no crash)':^38} | {no_crash_count:^10} |")
    
    print(f"+{'-' * 40}+{'-' * 12}+")
    
    # Summary statistics
    crash_count = len(iterations_list)
    total_runs = len(results)
    
    print(f"\nSummary: {crash_count}/{total_runs} runs found crashes")
    
    if iterations_list:
        mean_iter = np.mean(iterations_list)
        median_iter = np.median(iterations_list)
        std_iter = np.std(iterations_list) if len(iterations_list) > 1 else 0
        min_iter = min(iterations_list)
        max_iter = max(iterations_list)
        
        print(f"Mean {eval_label} to crash: {mean_iter:.2f}")
        print(f"Median {eval_label} to crash: {median_iter:.2f}")
        print(f"Std dev: {std_iter:.2f}")
        print(f"Min: {min_iter}, Max: {max_iter}")
    else:
        print(f"No crashes found - no {eval_label} statistics available")
    
    print("=" * 60)


def save_results_csv(results: List[Dict[str, Any]], filename: str) -> None:
    """Save frequency results to a CSV file."""
    iterations_list = [r["iterations_to_crash"] for r in results if r["iterations_to_crash"] is not None]
    no_crash_count = sum(1 for r in results if r["iterations_to_crash"] is None)
    
    freq_counter = Counter(iterations_list)
    sorted_evals = sorted(freq_counter.keys())
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['number_of_evaluations_to_find_crash', 'frequency'])
        
        for eval_count in sorted_evals:
            freq = freq_counter[eval_count]
            writer.writerow([eval_count, freq])
        
        if no_crash_count > 0:
            writer.writerow(['N/A', no_crash_count])
    
    print(f"\nResults saved to: {filename}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark search methods for finding crashes"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hill_climb", "random"],
        required=True,
        help="Search method to benchmark (hill_climb or random)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of times to run the search (default: 10)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Max iterations per run (default: 50 for hill_climb, 30 for random)"
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Neighbors per iteration for hill_climb (default: 5)"
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=1,
        help="Number of evaluations per scenario for random search (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV filename (default: benchmark_results.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)"
    )
    
    args = parser.parse_args()
    
    # Set default iterations based on method
    if args.iterations is None:
        args.iterations = 50 if args.method == "hill_climb" else 30
    
    print(f"\nBenchmarking {args.method} with {args.runs} runs")
    print(f"Max iterations/scenarios per run: {args.iterations}")
    if args.method == "hill_climb":
        print(f"Neighbors per iteration: {args.neighbors}")
    if args.parallel > 1:
        print(f"Parallel workers: {args.parallel}")
    print("-" * 60)
    
    # Generate seeds for each run
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        seeds = [int(rng.integers(1e9)) for _ in range(args.runs)]
    else:
        seeds = [int(np.random.default_rng().integers(1e9)) for _ in range(args.runs)]
    
    results = []
    
    if args.parallel > 1:
        # Parallel execution
        print(f"\nStarting {args.runs} runs in parallel with {args.parallel} workers...\n")
        
        if args.method == "hill_climb":
            worker_args = [
                (run_num, seeds[run_num - 1], args.iterations, args.neighbors, args.runs)
                for run_num in range(1, args.runs + 1)
            ]
            with mp.Pool(processes=args.parallel) as pool:
                results = pool.map(_worker_hill_climb, worker_args)
        else:  # random
            worker_args = [
                (run_num, seeds[run_num - 1], args.iterations, args.n_eval, args.runs)
                for run_num in range(1, args.runs + 1)
            ]
            with mp.Pool(processes=args.parallel) as pool:
                results = pool.map(_worker_random_search, worker_args)
    else:
        # Sequential execution
        print("Loading environment and policy...")
        from policies.pretrained_policy import load_pretrained_policy
        from envs.highway_env_utils import make_env
        
        env_id = "highway-fast-v0"
        policy = load_pretrained_policy("agents/model")
        env, defaults = make_env(env_id)
        
        for run_num in range(1, args.runs + 1):
            print(f"\n=== Run {run_num}/{args.runs} (seed: {seeds[run_num - 1]}) ===")
            
            if args.method == "hill_climb":
                crash_found, iterations = run_hill_climb_benchmark(
                    env_id=env_id,
                    policy=policy,
                    defaults=defaults,
                    seed=seeds[run_num - 1],
                    iterations=args.iterations,
                    neighbors_per_iter=args.neighbors,
                    run_number=run_num,
                    total_runs=args.runs,
                )
            else:  # random
                crash_found, iterations = run_random_search_benchmark(
                    env_id=env_id,
                    policy=policy,
                    defaults=defaults,
                    seed=seeds[run_num - 1],
                    n_scenarios=args.iterations,
                    n_eval=args.n_eval,
                    run_number=run_num,
                    total_runs=args.runs,
                )
            
            results.append({
                "run": run_num,
                "crash_found": crash_found,
                "iterations_to_crash": iterations,
                "seed": seeds[run_num - 1],
            })
    
    # Output results
    print_results_table(results, args.method)
    save_results_csv(results, args.output)


if __name__ == "__main__":
    main()
