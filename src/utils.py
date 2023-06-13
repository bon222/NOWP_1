import numpy as np
import matplotlib.pyplot as plt

def plot_contour(func, limits, heading, vectors=None, levels=20, paths=None, labels=None):
    x = np.linspace(limits[0], limits[1], 500)
    y = np.linspace(limits[2], limits[3], 500)
    X, Y = np.meshgrid(x, y)
    Z = func(np.vstack([X.ravel(), Y.ravel()]))[0]  # Compute f values only
    Z = Z.reshape(X.shape)  # Reshape Z to be 2D

    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z, levels=levels)

    if vectors is not None:
        for vector in vectors:
            plt.plot(vector[0], vector[1], 'ro')

    if paths and labels:
        for path, label in zip(paths, labels):
            plt.plot(*path.T, label=label)
        plt.legend()

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Contour Plot: "+ heading)
    plt.grid(True)
    plt.show()

def plot_function_values(function_values,heading):
    plt.figure(figsize=(8, 8))
    x_axis=[]
    y_axis=[]
    i=1
    for val in function_values:
        x_axis.append(i)
        i=i+1
        y_axis.append(val)
    plt.plot(x_axis, y_axis)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Function Values vs. Iteration: '+heading)
    plt.show()