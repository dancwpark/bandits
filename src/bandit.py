import numpy as np
import random
import matplotlib.pyplot as plt

class Arm(object):
    def __init__(self, prob):
        try:
            assert (prob >= 0 and prob <= 1)
        except AssertionError:
            raise ValueError("Incorrect range on probability for bandit -- range is [0, 1]")
        self.prob = prob
        self.win = 1
        self.loss = 0
        self.n = 0

    def pull(self):
        self.n += 1
        rand = random.random()
        if rand <= self.prob:
            return self.win
        else:
            return self.loss
    
    def stats(self):
        print("Pulled {} times".format(self.n))

# For Testing
if __name__ == "__main__":
    a = Arm(0.7)
    b = a.pull()
    print(b)
