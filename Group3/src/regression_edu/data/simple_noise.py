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
        fuzz = collect_err(**kwargs)
    else:
        # depending on the value between -1 and 1, return the noise with a quadratic magnitude
        # this is to simulate heteroscedastic noise
        size = kwargs['size']
        sd_deviation = np.random.uniform(0, 4, size=size) + (np.array(y) / size)
        reversed_ = True if kwargs['heteroscedacity'] < 0 else False
        y_heteroskedastic = np.array(y) + np.random.normal(loc=0, scale = np.abs(kwargs['heteroscedacity'] * (1 +  np.array(sorted(y, reverse=reversed_)))), size=size)
        fuzz = y_heteroskedastic
    y = y + fuzz 

    return y