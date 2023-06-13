import numpy as np

def line_search(f, x0,method):
    current_x = x0
    if method == "newton":
        function = f(current_x,hess=True)
    else:
        function = f(current_x)
    if method == "grad_desc":
        pk = -function[1]
    else:
        pk = -np.linalg.inv(function[2])@function[1]
    a = 1
    c = 0.5
    new_function = f(current_x + a * pk)
    while new_function[0] > function[0] + c * a * np.dot(function[1], pk):
        a = a- 0.01
        new_function = f(current_x + a * pk)
    return current_x+a*pk

def minimize(f,method, x0=np.array([1, 1]),obj_tol=np.power(10.0, -12), param_tol=np.power(10.0, -8), max_iter=100):
    lst_x = []
    lst_fx = []
    i=0
    prev_x=x0
    current_x=x0
    current_fun = f(current_x)
    current_fx=current_fun[0]
    lst_x.append(current_x)
    lst_fx.append(current_fx)
    prev_hess = None
    while i<max_iter:
        if method == "SR1" or method == "BFGS":
            if i==0:
                fun = f(current_x,hess=True)
                prev_hess = fun[2]
            elif method == "SR1":
                h = sr1_hess(f,current_x,prev_hess)
                prev_hess=h
            else:
                h = bfgs_hess(f,current_x,prev_hess,prev_x)
                prev_hess=h

            pk = -np.linalg.inv(prev_hess)@fun[1]
            a = 1
            c  = 0.5
            new_function = f(current_x + a * pk)
            while new_function[0] > current_fun[0] + c * a * np.dot(current_fun[1], pk):
                a = a- 0.01
                new_function = f(current_x + a * pk)
            new_x= current_x+a*pk
        else:
            new_x = line_search(f,current_x,method)
        new_fun = f(new_x)
        new_fx=new_fun[0]
        lst_x.append(new_x)
        lst_fx.append(new_fx)
        distance = ((current_x[0] - new_x[0])**2 + (current_x[1] - new_x[1])**2)**0.5
        if distance<param_tol or np.abs(f(new_x)[0]-f(current_x)[0])<obj_tol:
            return new_x, f(new_x)[0], lst_x, lst_fx, True
        else:
            prev_x=current_x
            current_x = new_x
            current_fun = f(current_x)
            current_fx = current_fun[0]
        i+=1
    return new_x, f(new_x)[0], lst_x, lst_fx, False 

def sr1_hess(f, x, prev_hess):
    fun = f(x, hess=False)
    g = fun[1]
    h = np.outer(g, g) / np.dot(g, prev_hess @ g)  # SR1 update formula

    # Return the updated Hessian.
    return prev_hess + h

def bfgs_hess(f, x, prev_hess, prev_x):
    fun = f(x, hess=False)
    g = fun[1]
    y = g.reshape(-1, 1)
    s = (x - prev_x).reshape(-1, 1)
    h = prev_hess - (prev_hess @ s @ s.T @ prev_hess) / (s.T @ prev_hess @ s) + (y @ y.T) / (y.T @ s)

    # Return the updated Hessian.
    return h

