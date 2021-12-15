import numpy as np
import matplotlib.pyplot as plt
import bandit
from solvers import optimistic_greedy

probs = [0.4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0.7, 0.8]

def main():
    bandits = []
    for i in range(10):
        bandits.append(bandit.Arm(probs[i]))
    
    score = optimistic_greedy.solve(bandits)
    print("Total Score: {}".format(score))
    for i, arm in enumerate(bandits):
        print("Arm {} was pulled {} times".format(i, arm.n))

if __name__ == '__main__':
    main()