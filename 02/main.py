import scipy.io
import numpy as np
import copy
from matplotlib import pyplot as plt
LEARNING_RATE = 0.07

def cost(theta: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    m = len(theta)
    res = 0.0
    for x_i, y_i in zip(xs, ys):
        res += (theta.T @ x_i - y_i) ** 2.0
    return 1/2 * 1/m * res

def gradient_descent(theta: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    res = np.array([0.0]* len(theta))
    for x_i, y_i in zip(xs, ys):
        res = np.add(res, (theta.T @ x_i - y_i) * x_i)
    return theta - LEARNING_RATE / len(xs) * res

def train(initial_theta: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    theta = copy.deepcopy(initial_theta)
    iterations = 0
    last_cost = np.Infinity
    current_const = np.Infinity
    history = []
    while iterations < MAX_ITERATIONS and (last_cost - (current_const := cost(theta, xs, ys))) >= COST_EPSILON:
        theta = gradient_descent(theta, xs, ys)
        history.append(theta)
        iterations += 1
        last_cost = current_const
    return theta, history

MAX_ITERATIONS = 1000
COST_EPSILON = 0.001
def h_1(xs: np.ndarray, ys: np.ndarray):
    # gradient descent
    theta = np.array([3.0])
    theta, history = train(theta, xs, ys)
    history_cost = np.array(list(map(lambda x: cost(x, xs, ys), history)))

    # brute force
    interval = np.arange(0.0, 3.0, 3.0/1000.0)
    cost_values = np.array(list(map(lambda x: cost(np.array([x]), xs, ys), interval)))
    theta_b = interval[np.argmin(cost_values)]
    if False:
        plt.plot(interval, cost_values)
        plt.plot(theta_b, cost(np.array([theta_b]), xs, ys), "ro")
        plt.plot(history, history_cost, "g.")
        plt.show()
    print(f"Model for h_1: gd: {theta} bf: {theta_b}")

def h_2(xs: np.ndarray, ys: np.ndarray):
    # gradient descent:
    theta = np.array([2.0, 4.0])
    xs_ = np.hstack([np.ones(xs.shape), xs])
    theta, history = train(theta, xs_, ys)
    history_cost = np.array(list(map(lambda x: cost(x, xs_, ys), history)))

    # brute force:
    xr = np.arange(-2.0, 4.0, 6.0/1000.0)
    yr = np.arange(-2.0, 4.0, 6.0/1000.0)
    xx, yy = np.meshgrid(xr, yr, sparse=False)
    z = []
    for x, y in zip(xx, yy):
        z.append(cost(np.array([x, y]), xs_, ys))
    z = np.array(z)
    plt.contour(xx, yy, z, 20)
    plt.axis([-2.0, 4.0, -2.0, 4.0])
    plt.plot(history, history_cost, "go")
    plt.show()
    print(f"Model for h_2: {theta}")

def h_3(xs: np.ndarray, ys: np.ndarray):
    theta = np.array([.0, .0, .0])
    xs_ = np.hstack([np.ones(xs.shape), xs, np.square(xs)])
    theta, history = train(theta, xs_, ys)
    print(f"Model for h_3: {theta}")

def main():
    mat = scipy.io.loadmat("data_ex7.mat")
    xs = mat["x"]
    ys = mat["y"]
    h_1(xs, ys)
    h_2(xs, ys)
    h_3(xs, ys)

if __name__ == "__main__":
    main()