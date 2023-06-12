import numpy as np

def line_search(f, x0,method):
    current_x = x0
    function = f(current_x,hess=False)
    pk = -function[1]
    a = 1
    c = 0.5
    new_function = f(current_x + a * pk)
    while new_function[0] > function[0] + c * a * np.dot(function[1], pk):
        a = a- 0.01
        new_function = f(current_x + a * pk)
    return current_x+a*pk

def minimize(f, x0=np.array([11, 11]),obj_tol=np.power(10.0, -12), param_tol=np.power(10.0, -8), max_iter=100,method="gd"):
    lst_x = []
    lst_fx = []
    i=0
    current_x=x0
    current_fx = f(current_x)[0]
    lst_x.append(current_x)
    lst_fx.append(current_fx)
    while i<max_iter:
        new_x = line_search(f,current_x,method)
        new_fx = f(new_x)[0]
        lst_x.append(new_x)
        lst_fx.append(new_fx)
        distance = ((current_x[0] - new_x[0])**2 + (current_x[0] - current_x[1])**2)**0.5
        if distance<param_tol or np.abs(f(new_x)[0]-f(current_x)[0])<obj_tol:
            return new_x, f(new_x)[0], lst_x, lst_fx, True
        else:
            current_x = new_x
            current_fx = new_fx
        i+=1
    return new_x, f(new_x)[0], lst_x, lst_fx, False 

def line_search_newton(f, x0,method):
    current_x = x0
    function = f(current_x)
    pk = -np.linalg.inv(function[2])@function[1]
    a = 1
    c = 0.5
    print(pk)
    new_function = f(current_x + a * pk)
    while new_function[0] > function[0] + c * a * np.dot(function[1], pk):
        a = a- 0.01
        new_function = f(current_x + a * pk)
    return current_x+a*pk

def minimize_newton(f, x0=np.array([11, 11]),obj_tol=np.power(10.0, -12), param_tol=np.power(10.0, -8), max_iter=100,method="gd"):
    lst_x = []
    lst_fx = []
    i=0
    current_x=x0
    current_fx = f(current_x)[0]
    lst_x.append(current_x)
    lst_fx.append(current_fx)
    while i<max_iter:
        new_x = line_search_newton(f,current_x,method)
        new_fx = f(new_x)[0]
        lst_x.append(new_x)
        lst_fx.append(new_fx)
        distance = ((current_x[0] - new_x[0])**2 + (current_x[0] - current_x[1])**2)**0.5
        if distance<param_tol or np.abs(f(new_x)[0]-f(current_x)[0])<obj_tol:
            return new_x, f(new_x)[0], lst_x, lst_fx, True
        else:
            current_x = new_x
            current_fx = new_fx
        i+=1
    return new_x, f(new_x)[0], lst_x, lst_fx, False