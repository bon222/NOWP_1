import numpy as np

def quadratic_1(v, hess=True):
    Q = np.array([[1, 0], [0, 1]])
    f = np.sum(v * (Q @ v), axis=0)
    g = 2 * (Q @ v)
    h = 2 * Q if hess else None
    return f, g, h

def quadratic_2(v, hess=True):
    Q = np.array([[2, 1], [1, 2]])
    f = np.sum(v * (Q @ v), axis=0)
    g = 2 * (Q @ v)
    h = 2 * Q if hess else None
    return f, g, h

def quadratic_3(v, hess=True):
    Q = np.array([[2, 0], [0, 1]])
    f = np.sum(v * (Q @ v), axis=0)
    g = 2 * (Q @ v)
    h = 2 * Q if hess else None
    return f, g, h

def rosenbrock(x, hess=True):
    f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    g = np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                  200 * (x[1] - x[0] ** 2)])
    h = np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]],
                  [-400 * x[0], 200]])
    return f, g, h if hess else None

def linear(v, hess=False):
    f = np.array([1,1])*v
    g = np.array([1,1])
    h = None
    return f, g, h

def triangles(v, hess=True):
    x, y = v
    f = np.exp(x + 3 * y - 0.1) + np.exp(x - 3 * y - 0.1) + np.exp(-x - 0.1)
    g = np.array([np.exp(x + 3 * y - 0.1) + np.exp(x - 3 * y - 0.1) - np.exp(-x - 0.1), 3*np.exp(x+3 * y - 0.1)-3*np.exp(x-3*y-0.1)])
    h = np.array([[np.exp(x+3*y-0.1)+np.exp(x-3*y-0.1)+np.exp(-x-0.1),3*np.exp(x+3*y-0.1)-3*np.exp(x-3*y-0.1)],[3*np.exp(x+3*y-0.1)-3*np.exp(x-3*y-0.1), 9*np.exp(x+3*y-0.1)+9*np.exp(x-3*y-0.1)]])
    return f, g, None