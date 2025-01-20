import numpy as np


def simulate_coin_flips(p, N):
    flips = np.random.choice([1, 0], size=N, p=[p, 1-p])
    cumulative_heads = np.cumsum(flips)
    return cumulative_heads.tolist()


def compute_variance_at_trails(p, N, R):
    cumulative_heads_list = []
    for _ in range(R):
        cumulative_heads = simulate_coin_flips(p, N)
        cumulative_heads_list.append(cumulative_heads)

    cumulative_heads_array = np.array(cumulative_heads_list)

    mean_heads = np.mean(cumulative_heads_array, axis=0)
    variance_heads = np.var(cumulative_heads_array, axis=0)

    return variance_heads


p = 0.5
N = 2000
R = 1000

variance = compute_variance_at_trails(p, N, R)
