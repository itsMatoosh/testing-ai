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

    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    # crashes = search.run_search(n_scenarios=2, seed=11)

    hc = hill_climb(
        env_id,
        base_cfg,
        param_spec,
        policy,
        defaults,
        seed=1,
        iterations=10,
        neighbors_per_iter=10,
    )

    # print(f"✅ Found {len(crashes)} crashes.")
    print(hc)
    #if crashes:
    #    print(crashes)


if __name__ == "__main__":
    main()
