import numpy as np

LEARNING_RATE = 0.05

class LinearPerceptron:

    def __init__(self, dim):
        self.weights = [0.0 for _ in range(dim + 1)]

    def classify(self, sample):
        x = [1.0] + sample
        return np.array(self.weights) @ np.array(x)

    def train(self, x, y):
        output = self.classify(x)
        error = LEARNING_RATE * (y - output) * np.array([1.0] + list(map(float, x)))
        self.weights += error

    def loss(self, xs, ys):
        mistakes = 0
        for x, y in zip(xs, ys):
            mistakes += 0 if max(0, self.classify(x)) == y else 1

        return mistakes / len(xs)

