import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## draft, not finished, gonna improve later


def generate_data():
    polydict = {
        'f1':{'degree':1,'coefficient':20},
        'f2':{'degree':1,'coefficient':-4},
        'f3':{'degree':2,'coefficient':1.5},
        'f4':{'degree':3,'coefficient':1.02}
        }

    intercept = 20

    data_range = [-25, 25]
    y_true = range(data_range[0], data_range[1])
    size = len(y_true)

    vars = pd.DataFrame(
        {'intercept':[(intercept) for x in y_true]}
        )

    for func in polydict:
        degree = polydict[func]['degree']
        coefficient = polydict[func]['coefficient']
        vars[func] = evaluate_polynomial(y_true, *(coefficient for x in range(degree+1)))

    y = vars['intercept'] + vars['f1'] + vars['f2'] + vars['f3'] + vars['f4']


    ## add noise to y
    mean = 0
    std = y.std() / 1.5
    eps = np.random.normal(mean, std, size)

    y_noised = y - eps

    # scatter plot the different polynomials against y_noised in one plot.

    vars['y'] = y
    vars['y_noised'] = y_noised

    vars.to_csv('./Groupwork/Group3/data/synth/easy_polynomials.csv', index=False)


def evaluate_polynomial(x, *coeffs):
    degree = len(coeffs) - 1
    powers = np.arange(degree, -1, -1)
    x_powers = np.power.outer(x, powers)
    return np.dot(coeffs, x_powers)


generate_data()