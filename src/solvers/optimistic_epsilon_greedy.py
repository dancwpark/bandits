import numpy as np
import random

def solve(bandits, epsilon=0.2, max_iters=100):
    try:
        assert (epsilon >=0) and (epsilon <=1)
    except ValueError:
        raise ValueError("Epsilon value must be between 0 and 1")
    total_reward = 0
    n_arms = len(bandits)
    probs = np.ones(n_arms, dtype=float)
    for _ in range(max_iters):
        r = random.random()
        if r <= epsilon:
            # Explore
            arm = random.randrange(n_arms)
        else:
            # Exploit
            # Find "optimal" arm
            arm = np.argmax(probs)
        # Pull arm
        reward = bandits[arm].pull()
        # Get rewarded
        total_reward += reward
        # Update probability
        n = bandits[arm].n
        probs[arm] = (float(n) / (n+1)) * probs[arm] + (1. / (n+1)) * reward
    
    return total_reward, probs

