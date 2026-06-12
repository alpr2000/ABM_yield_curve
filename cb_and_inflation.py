import numpy as np
import numpy.random as random

def set_interest_rate_taylor_rule(inflation_rate, r_star, target_inflation, phi_pi):
    """
    Sets the nominal interest rate based on the Taylor rule
    """
    return r_star + phi_pi * (inflation_rate - target_inflation)

def inflation_ar_process(starting_inflation, i_lr, phi, sigma, periods, seed=None):
    """
    Simulates an AR(1) process for inflation
    """
    if seed is not None:
        random.seed(seed)
    inflation_series = [starting_inflation]
    for t in range(1, periods):
        new_inflation = i_lr + phi * (inflation_series[-1] - i_lr) + random.normal(0, sigma)
        inflation_series.append(new_inflation)
    return np.array(inflation_series)

def inflation_asymmetric_process(starting_inflation, i_lr, phi, sigma, periods, seed=None):
    """
    Simulates an AR(1) process for inflation with asymmetric shocks (larger positive shocks)
    """
    if seed is not None:
        random.seed(seed)
    inflation_series = [starting_inflation]
    for t in range(1, periods):
        shock = random.normal(0, sigma)
        # Make positive shocks larger than negative shocks
        if shock > 0:
            shock *= 1.5
        new_inflation = i_lr + phi * (inflation_series[-1] - i_lr) + shock
        inflation_series.append(new_inflation)
    return np.array(inflation_series)