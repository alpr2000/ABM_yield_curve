"""

Here we generate the government bond auction

"""
import numpy as np
import numpy.random as random
from parameters import *

def stochastic_bond_supply(quantity, maturity_spectrum):
    supply = np.random.uniform(0, 1, size=maturity_spectrum.shape)
    supply = supply/np.sum(supply)*quantity
    return supply


# This function find the clearing price for all maturities    
def find_clearing_price_auction(demand_pfs, supply, price_spectrum):
    # demand_pfs is a 2D stacked array
    
