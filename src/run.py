import numpy as np
import matplotlib.pyplot as plt
import bandit
from solvers import optimistic_greedy

probs = [0.4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0.7, 0.8]

def main():
    bandits = []
    for i in range(10):
        bandits.append(bandit.Arm(probs[i]))
    
    score, learned_probs = optimistic_greedy.solve(bandits)
    print("Total Score: {}".format(score))
    for i, arm in enumerate(bandits):
        print("Arm {} was pulled {} times. It's prob is {:.2f} and we learned {:.2f}".format(i, arm.n, probs[i], learned_probs[i]))
    
    # TODO: run a solution multiple times to gather mean and standard deviation
    # TODO: plot the mean with fill_between using (mean-std, mean+std)

if __name__ == '__main__':
    main()