import numpy as np
import numpy.random as random

"""
Generates demand functions for each agent type

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

################## Pension Fund Demand Function ##################


# Generates new liabilities which the PF must meet
def pf_gen_liability(maturity_spectrum):
    # Only add liabilities for the second half of the maturity spectrum
    new_liabilities = random.uniform(0, 1, size=len(maturity_spectrum))
    new_liabilities[:len(maturity_spectrum) // 2] = 0
    return new_liabilities

# Performs the pension fund's estimate of the fair price
def pf_fair_price(base_rate, term_premium, maturity_spectrum):
    #As all bonds are zero-coupon, the fair price is simply the discounted value of the liability at maturity
    fair_price = (100/base_rate)**(maturity_spectrum)*100
    term_premium_curve = (1+term_premium)**maturity_spectrum
    fair_price = fair_price/term_premium_curve 
    return (fair_price)

# Calculates the pension fund's demand for a bond of a particular liability
# This is actually redundant as we can just calculate the demand for all maturities at once, but it is useful for testing and visualization purposes
def pf_demand_individual(price_spectrum, fair_price, liability, time_to_liability):
    if time_to_liability == 0:
        demand_curve=0*price_spectrum
    else:
        #Implement a fermi-dirac-esque demand function
        urgency = 0.5/(time_to_liability)*fair_price # The more urgent the liability, the higher the urgency factor

        # We assume that if the time to liability is short, the pension fund demands more, pushing their fair price up
        demand_curve = liability/(1 + np.exp((price_spectrum - (fair_price+urgency )/1 )))
    return demand_curve

# Produces the nested array of demand functions for each maturity
def pf_demand_all_maturities(price_spectrum, maturity_spectrum, fair_prices, liability_spectrum):
    
    price_spectrum = np.asarray(price_spectrum)
    fair_prices = np.asarray(fair_prices)
    maturity_spectrum = np.asarray(maturity_spectrum)

    # Liability spectrum has the shape (N_pension_funds, len(maturity_spectrum))

    liability_spectrum = np.asarray(liability_spectrum)

    price_grid = price_spectrum[np.newaxis, :]
    fair_grid = fair_prices[:, np.newaxis]
    liability_grid = liability_spectrum[:,:, np.newaxis]
    time_grid = maturity_spectrum[:, np.newaxis]

    urgency = 0.5 / time_grid * fair_grid
    demand_array = liability_grid / (1 + np.exp(price_grid - fair_grid - urgency))
    demand_array = np.where(time_grid == 0, 0.0, demand_array)

    return demand_array


################## Hedge Fund Demand Functions ##################

def hf_curve_random(fair_price, maturity_spectrum, hf_random_type, hf_heterogeneity):
    if hf_random_type == 0:
        # Simple slope deviation
        random_slope = random.uniform(1-hf_heterogeneity, 1+hf_heterogeneity)
        random_curve = random_slope**maturity_spectrum * fair_price
        return random_curve
    else:
        raise ValueError("Invalid hedge fund random type")
    
# Create function to create each hedge fund proprietary pricing curve based on the current base rate and a random seed
def hf_fund_fair_price(base_rate, term_premium, maturity_spectrum, N_hedge_funds):
    # Base curve is simply the fair price curve based on the current base rate
    maturity_grid = np.asarray(maturity_spectrum)
    N_funds_grid = np.ones(N_hedge_funds)[:, np.newaxis]
    fair_price = (100/base_rate)**(maturity_spectrum)*100
    # Add some term premium
    term_premium_curve = (1+term_premium)**maturity_spectrum
    fair_price = fair_price/term_premium_curve  

    # Now we add a random component to the pricing curve for each fund
    fair_price = fair_price[np.newaxis, :]
    # We want a random curve for each of N_funds, so we repeat the fair price curve N_funds times and then apply the randomization function to each one
    fair_price = np.repeat(fair_price, N_hedge_funds, axis=0)

    fair_prices = hf_curve_random(fair_price, maturity_spectrum, hf_random_type, hf_heterogeneity)/100

    return fair_prices


# Produces the demand curve for a single hedge fund and a single maturity
def hf_demand_single(price_spectrum, fair_price, funds, scale_demand):
    demand_curve = -price_spectrum + fair_price
    demand_curve = (scale_demand*demand_curve) ** 2
    # Cut off demand at fair price < price and above the funds available
    demand_curve[price_spectrum > fair_price] = 0
    demand_curve = np.minimum(demand_curve, funds)
    return demand_curve

# Produces the nested array of demand functions for each maturity
def hf_demand_all_maturities(price_spectrum, maturity_spectrum, fair_prices, funds, scale_demand):
    price_spectrum = np.asarray(price_spectrum)
    fair_prices = np.asarray(fair_prices)
    maturity_spectrum = np.asarray(maturity_spectrum)

    price_grid = price_spectrum[np.newaxis, :]
    fair_grid = fair_prices[:, np.newaxis]

    demand_array = -price_grid + fair_grid
    demand_array = (scale_demand * demand_array) ** 2
    demand_array[price_grid > fair_grid] = 0
    demand_array = np.minimum(demand_array, funds)

    return demand_array


################### Noise Trader Demand Function ##################

# The noise trader just demands a normal distribution around the fair price

def nt_fair_price(base_rate, term_premium, maturity_spectrum):
    fair_price = (100/base_rate)**(maturity_spectrum)*100
    term_premium_curve = (1+term_premium)**maturity_spectrum
    fair_price = fair_price/term_premium_curve 
    return fair_price


def noise_trader_demand(price_spectrum, fair_prices, scale_noise_trader):
    # Generate a normal distribution around the fair price
    fair_prices = np.asarray(fair_prices)
    price_spectrum = np.asarray(price_spectrum)
    price_grid = price_spectrum[np.newaxis, :]
    fair_grid = fair_prices[:, np.newaxis]
    demand_array = np.exp(-0.5 * ((price_grid - fair_grid) / 5) ** 2)
    demand_array = demand_array / demand_array.sum(axis=1, keepdims=True) * cash
    return demand_array




