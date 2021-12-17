import numpy as np
import matplotlib.pyplot as plt
import bandit
from solvers import optimistic_greedy, epsilon_greedy, optimistic_epsilon_greedy

probs = [0.4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0.7, 0.8]
solver = optimistic_epsilon_greedy

def experiment(n_iters=1000):
    bandits = []
    for i in range(10):
        bandits.append(bandit.Arm(probs[i]))
    
    score, learned_probs = solver.solve(bandits, max_iters=n_iters)
    print("Total Score: {}".format(score))
    for i, arm in enumerate(bandits):
        print("Arm {} was pulled {} times. It's prob is {:.2f} and we learned {:.2f}".format(i, arm.n, probs[i], learned_probs[i]))
    
    return bandits, learned_probs, score

def main():
    iters = np.linspace(100, 10000, 100).astype(int)
    all_learned_probs = np.array([[]])
    all_scores = np.array([])
    for iter in iters:
        _, probs, score = experiment(iter)
        if all_learned_probs.size == 0:
            all_learned_probs = np.array([probs])
        else:
            all_learned_probs = np.append(all_learned_probs, np.array([probs]), axis=0)
        all_scores = np.append(all_scores, score)
    
    # Graph stuff now
    figure, axis = plt.subplots(2, 1)

    axis[0].plot(iters, all_scores)
    axis[0].set_title("Score over iterations")
    axis[0].set_ylabel("Score")
    axis[0].set_xlabel("Iterations")

    for i, prob in enumerate(probs): 
        label = "Arm " + str(i)
        axis[1].plot(iters, abs(prob - all_learned_probs.T[i]), label=label)
    axis[1].set_title("Deviation from true probabilities over iterations")
    axis[1].set_ylabel("Deviation from True Probability")
    axis[1].set_xlabel("Iterations")
    axis[1].legend()

    plt.show()



if __name__ == '__main__':
    main()