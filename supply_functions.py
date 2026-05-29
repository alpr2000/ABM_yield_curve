import numpy as np
import numpy.random as random

import numpy as np
import numpy.random as random

"""
Generates supply functions for each agent type

NOTE: All prices are expressed as the return on a 100 unit nominal bond
NOTE: Base rate expressed as the return on a 100 unit nominal bond over 1 month
      Eg. 101 means a 1% return over 1 month, which annualizes to approximately 12.68%
"""

# Load variables from parameters.py
from parameters import maturity_spectrum
from parameters import price_spectrum
from parameters import scale_demand
from parameters import term_premium
from parameters import hf_random_type
from parameters import hf_heterogeneity
from parameters import pf_liquidity_buffer

################## Hedge Fund Supply Functions ##################

# Produces the nested array of supply functions for each maturity
def hf_supply_all_maturities(price_spectrum, maturity_spectrum, fair_prices, scale_demand, HF_holdings):
    """
    Generates supply curves for hedge funds across all maturities
    HFs supply more as price exceeds their fair price estimate
    Supply is capped by their current holdings
    """
    price_spectrum = np.asarray(price_spectrum)
    fair_prices = np.asarray(fair_prices)
    HF_holdings = np.asarray(HF_holdings)

    # Reshape for broadcasting to (25, 360, 200)
    price_grid = price_spectrum  # shape (200,)
    fair_grid = fair_prices[:, :, np.newaxis]  # shape (25, 360, 1)
    holdings_grid = HF_holdings[:, :, np.newaxis]  # shape (25, 360, 1)

    # Supply increases quadratically when price > fair_price
    supply_array = price_grid - fair_grid
    supply_array = (scale_demand * supply_array) ** 2
    supply_array[price_grid <= fair_grid] = 0

    # Cap supply by holdings
    supply_array = np.minimum(supply_array, holdings_grid)
    
    # Round to integer bond quantities
    supply_array = np.floor(supply_array)

    return supply_array


################## Noise Trader Supply Functions ##################

def nt_supply_all_maturities(price_spectrum, maturity_spectrum, NT_holdings, fair_prices, nt_cash_perc ):
    """
    Generates supply curves for noise traders across all maturities
    NT supplies uniformly distributed across prices based on their holdings
    """
    price_spectrum = np.asarray(price_spectrum)
    NT_holdings = np.asarray(NT_holdings)
    fair_prices = np.asarray(fair_prices)
    

    fair_grid = fair_prices[:, np.newaxis]  # shape (1, 360, 1)
    
    supply_array = np.exp(-0.5 * ((price_spectrum - fair_grid) / 5) ** 2)

    supply_array = NT_holdings.reshape(-1,1) * supply_array * nt_cash_perc
    
    supply_array = np.floor(supply_array)

    return supply_array
