from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import hill_climb


def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    print(param_spec)
    print(base_cfg)
    n_tests = 20
    crash_iterations = []
    for i in range(n_tests):
        print("==== Test ", i+1, " ====")
        search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
        crash_iteration = search.run_search(seed=None)
        crash_iterations.append(crash_iteration)
    print(crash_iterations)

    n_evaluations = []
    for i in range(n_tests):
        print("==== Hill Climb Test ", i+1, " ====")
        hc = hill_climb(
            env_id,
            base_cfg,
            param_spec,
            policy,
            defaults,
            seed=None,
            iterations=50,
            neighbors_per_iter=5,
        )
        n_evaluations.append(hc['n_eval'])

    print(n_evaluations)
    print(crash_iterations)


if __name__ == "__main__":
    main()
