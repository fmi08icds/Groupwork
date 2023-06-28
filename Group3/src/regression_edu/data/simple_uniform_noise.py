import numpy as np

def simple_uniform(f, N, range, rho):
    """
    Simple method. Given an input function, generate a dataset with a uniform fuzz. 
    
    f (func): python lambda function to model 
    N (int): number of datapoints
    range (tuple): interval for input data. 
    rho (int): noise scaling factor on y-values rho \in (0,1] -> Think of this as the amplitude of the output data 

    Returns: 
        x (np array): x-values 
        y (np array): generated y-values
    """
    # retrieve range
    a,b = range
    assert a <= b, f"invalid range, expected (a,b) with a <= b, got a = {a}, b = {b}"     

    # create uniform x-value distribution, compute function values 
    x = np.random.default_rng().uniform(a,b,N)
    y_pre = f(x)

    # scale the fuzz given rho input 
    # create and apply uniform noise to each value 
    # TODO: think about how to scale amount of fuzz. For now using interval length. Maybe a better idea: min and max value of function within range
    min = np.min(y_pre)
    max = np.max(y_pre) 
    
    scale = rho*(max - min)
    
    fuzz = np.random.default_rng().uniform(0, scale, N)
    
    stretch = np.random.default_rng().choice([-1,1],N) # vector of -1's and 1's to fuzz in both directions 
    fuzz = fuzz * stretch 
    y = y_pre + fuzz 

    return x,y 