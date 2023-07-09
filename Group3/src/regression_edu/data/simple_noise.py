import numpy as np

def generate_x(f,distr_x:str="normal", **kwargs):
    """
    Given a distribution name and its parameters, generate an array of x-values.
    
    distr_x (str): name of the distribution to use
    **kwargs: parameters for the distribution
    Returns:
        x (np array): array of x-values
    """
    
    collect_x = getattr(np.random,str.lower(distr_x))
    x = collect_x(**kwargs)
    # compute function values
    y = f(x) # call the lambda function with the input x to evaluate the function
    return x,y

def add_noise(y, distr_eps:str="normal", **kwargs):
    """
    Given an input function and an array of x-values, generate a dataset with uniform noise.
    
    f (func): python lambda function to model 
    x (np array): array of x-values
    rho (int): noise scaling factor on y-values rho \in (0,1] -> Think of this as the amplitude of the output data 
    distr_eps (str): name of the distribution to use for the noise
    **kwargs: parameters for the distribution
    Returns: 
        y (np array): generated y-values
    """

    if distr_eps != 'Heteroscedastic':
        collect_err = getattr(np.random,str.lower(distr_eps))
        err = collect_err(**kwargs)
        stretch = np.random.default_rng().choice([-1,1],len(err)) # vector of -1's and 1's to fuzz in both directions 
        fuzz = err * stretch 
    else:
        # depending on the value between -1 and 1, return the noise with a quadratic magnitude
        # this is to simulate heteroscedastic noise

        noise = np.random.default_rng().normal(-1,1,len(y))
        heteroscedacity = kwargs['heteroscedacity']
        # sort y values
        # scale y values from 0 to 1
        y_sorted = (y - np.min(y))/(np.max(y) - np.min(y))
        y_sorted = np.sort(y_sorted)
        # noise
        fuzz = noise * (y_sorted**2)*heteroscedacity
    y = y + fuzz 

    return y
