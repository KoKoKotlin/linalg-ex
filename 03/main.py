import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow.keras as keras
import scipy.io
import numpy as np

from matplotlib import pyplot as plt

def ex41(xs, ys):

    model = keras.Sequential(name="geilo")
    
    node_count = 1
    input_shape = (1, 1)
    
    model.add(keras.layers.Dense(node_count, input_shape=input_shape))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1), loss="mean_absolute_error")

    model.fit(xs, ys, epochs=5, batch_size=10, validation_split=0.2)
    pxs = np.array([[0.0], [3.5]])
    pys = model.predict(pxs)
    
    pys_ = []
    for y in pys:
        pys_.append(y[0][0])
    
    plt.scatter(xs, ys)
    plt.plot([0.0, 3.5], pys_, color="r")
    plt.show()


def main():
    mat = scipy.io.loadmat("data_ex7.mat")
    xs = mat["x"]
    ys = mat["y"]
    
    # ex41(xs, ys)
    ex42(xs, ys)

def ex42(xs, ys):
    model = keras.Sequential(name="geilo2")
    
    node_count = 1
    input_shape = (1, 1)
    
    model.add(keras.layers.Dense(node_count, input_shape=input_shape))
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mean_absolute_error")

    model.fit(xs, ys, epochs=10, batch_size=10, validation_split=0.2)
    pxs = np.arange(0.0, 4.0, .1).reshape((40, 1))

    pys = model.predict(pxs)
    
    pys_ = []
    for y in pys:
        pys_.append(y[0][0])
    
    plt.scatter(xs, ys)
    plt.scatter(pxs, pys_, color="r")
    plt.show()

if __name__ == "__main__":
    main()