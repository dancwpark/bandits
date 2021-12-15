import numpy as np

def solve(bandits, max_iters=100):
    total_reward = 0
    n_arms = len(bandits)
    probs = np.ones(n_arms, dtype=float)
    for _ in range(max_iters):
        # Find "optimal" arm
        arm = np.argmax(probs)
        # Pull arm
        reward = bandits[arm].pull()
        # Get rewarded
        total_reward += reward
        # Update probability
        n = bandits[arm].n
        probs[arm] = (float(n) / (n+1)) * probs[arm] + (1. / (n+1)) * reward
    
    return total_reward

