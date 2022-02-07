from unittest.main import MODULE_EXAMPLES
import numpy as np
from matplotlib import pyplot as plt

def calc_z(x, y, wx = 4, wy = 2):
    return wx * x + wy * y

def ex01():
    px = np.arange(0.0, 31.0, 1.0)
    py = np.arange(-1.5, 1.6, 0.1)

    # add some noise to the x data and y
    sigma_x = 2.0
    sigma_y = 0.2

    px_ = np.array(list(map(lambda x: x + np.random.normal(scale=sigma_x), px)))
    py_ = np.array(list(map(lambda x: x + np.random.normal(scale=sigma_y), py)))
    pz_ = np.array(list(map(lambda x: calc_z(x[0], x[1]), zip(px_, py_))))

    # 3d plot data
    fig = plt.figure()
    ax = fig.add_subplot(211, projection="3d")
    ax2 = fig.add_subplot(212)
 
    # fit line with noisy data
    A = np.column_stack([px_, py_])
    covariance_matrix = A.T @ A
    new_z = A.T @ pz_

    params = np.linalg.inv(covariance_matrix) @ new_z
    print(f"{params=}")

    calced_z = np.array(list(map(lambda x: calc_z(x[0], x[1], params[0], params[1]), zip(px, py))))

    ax.scatter(px_, py_, pz_)
    ax.plot(px, py, calced_z, "r-")

    # plot contour lines of loss function
    def loss(theta0, theta1):
        loss = 0.0
        for i in range(len(px)):
            loss += (theta0 * px_[i] + theta1 * py_[i] - calc_z(px_[i], py_[i])) ** 2 

        return loss
    
    xlist = np.linspace(-30.0, 30.0, 10)
    ylist = np.linspace(-30.0, 30.0, 10)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.array([loss(X[i], Y[i]) for i in range(len(X))])

    ax2.contour(X, Y, Z)

    print(f"Condition of A.T @ A: {np.linalg.cond(covariance_matrix)}")

    # normalize data
    mu_x = np.mean(px_)
    mu_y = np.mean(py_)
    si_x = np.sqrt(np.var(px_))
    si_y = np.sqrt(np.var(py_))
    
    print(f"{mu_x=}, {mu_y=}, {si_x=}, {si_y=}")

    normalized_x = (px_ - np.ones(px_.shape) * mu_x) / si_x
    normalized_y = (py_ - np.ones(py_.shape) * mu_y) / si_y
    
    normalized_z = np.array(list(map(lambda x: calc_z(x[0], x[1]), zip(normalized_x, normalized_y))))

    ax.scatter(normalized_x, normalized_y, normalized_z)

    A = np.column_stack([normalized_x, normalized_y])
    covariance_matrix = A.T @ A
    new_z = A.T @ normalized_z
    
    params = np.linalg.inv(covariance_matrix) @ new_z    
    print(f"{params=}")

    calced_z = np.array(list(map(lambda x: calc_z(x[0], x[1], params[0], params[1]), zip(px, py))))
    
    ax.plot(px, py, calced_z, "bx")

    print(f"Condition of A2.T @ A2: {np.linalg.cond(covariance_matrix)}")
    
    # pca sphering
    A = np.column_stack([px_, py_])
    U, s, V = np.linalg.svd(A)
    s_inv = np.diag(np.array(list(map(lambda x: 1 / x, s))))
    PCA_A = s_inv @ V.T @ A.T

    print(f"Condition of A.T @ A after PCA: {np.linalg.cond(PCA_A.T @ PCA_A)}")
    
    plt.show()

def main():
    ex01()

if __name__ == "__main__":
    main()