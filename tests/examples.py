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

def rosenbrock(v, hess=True):
    x, y = v
    f = 100 * (y - x**2)**2 + (1 - x)**2
    g = np.array([400*x**3 + 400*x*y +2*x - 2, 200 * (y - x**2)])
    h = np.array([[1200*x**2 - 400*y+2, -400*x],[-400 * x, 200]])if hess else None
    return f, g, h

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