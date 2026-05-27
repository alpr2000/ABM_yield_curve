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
def pf_gen_liability(maturity_spectrum, scale_liability, N_pension_funds):
    # Only add liabilities for the last 3 quarters of the maturity spectrum
    # Only allow integer multiples of the price granularity, to avoid issues with the demand function
    new_liabilities = np.random.randint(0, scale_liability, size=(N_pension_funds, len(maturity_spectrum)))
    new_liabilities[:, :len(maturity_spectrum) // 4] = 0
    return new_liabilities*scale_liability

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

def hf_curve_random(fair_price, maturity_spectrum, hf_random_type, hf_heterogeneity, seed):
    random.seed(seed)
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
    fair_price = (100/base_rate)**(maturity_spectrum)*100
    # Add some term premium
    term_premium_curve = (1+term_premium)**maturity_spectrum
    fair_price = fair_price/term_premium_curve  

    # Generate N_hedge_funds different slopes, one per fund
    random_slopes = random.uniform(1-hf_heterogeneity, 1+hf_heterogeneity, size=N_hedge_funds)
    random_slopes = random_slopes[:, np.newaxis]  # Shape (N_hedge_funds, 1) for broadcasting
    
    # Apply each slope to the fair price curve to get N_hedge_funds independent curves
    fair_prices = (random_slopes ** maturity_spectrum) * fair_price

    return fair_prices


# Produces the demand curve for a single hedge fund and a single maturity
def hf_demand_single(price_spectrum, fair_price, funds, scale_demand):
    demand_curve = -price_spectrum + fair_price
    demand_curve = (scale_demand*demand_curve) ** 2
    # Cut off demand at fair price < price and above the funds available
    demand_curve[price_spectrum > fair_price] = 0
    demand_curve = np.minimum(demand_curve, funds)
    return demand_curve

def allocate_hf_funds(base_rate, term_premium, maturity_spectrum, HF_fair_prices, HF_cash):
    fair_price = (100/base_rate)**(maturity_spectrum)*100
    term_premium_curve = (1+term_premium)**maturity_spectrum
    fair_price = fair_price/term_premium_curve

    price_diff = HF_fair_prices - fair_price[np.newaxis,:]
    
    # Find maturities with biggest deviations (top 20%) for EACH FUND
    abs_price_diff = np.abs(price_diff)
    
    # Compute 80th percentile for each fund (along maturities axis)
    cutoff_values = np.percentile(abs_price_diff, 80, axis=1)
    cutoff_values = cutoff_values[:, np.newaxis]  # Reshape to (N_funds, 1)
    
    # Keep only values where |deviation| >= cutoff, preserving the original sign
    price_diff_filtered = np.where(abs_price_diff >= cutoff_values, price_diff, 0)

    return price_diff_filtered


# Produces the nested array of demand functions for each maturity
def hf_demand_all_maturities(price_spectrum, maturity_spectrum, fair_prices, scale_demand):
    price_spectrum = np.asarray(price_spectrum)
    fair_prices = np.asarray(fair_prices)
    maturity_spectrum = np.asarray(maturity_spectrum)

    # Reshape for broadcasting to (25, 360, 200)
    price_grid = price_spectrum  # shape (200,)
    fair_grid = fair_prices[:, :, np.newaxis]  # shape (25, 360, 1)

    demand_array = -price_grid + fair_grid
    demand_array = (scale_demand * demand_array) ** 2
    demand_array[price_grid > fair_grid] = 0


    return demand_array


################### Cash Constraint Function ##################

def cap_demand_by_cash(demand_array, price_spectrum, cash_holdings, liqudity_buffer):
    """
    Caps demand curves to ensure price*quantity spending doesn't exceed available cash.
    Scales down all demands proportionally for agents that would overspend.
    
    Parameters:
    -----------
    demand_array : ndarray, shape (N_agents, N_maturities, N_prices)
        Demand quantities across prices for each agent and maturity
    price_spectrum : ndarray, shape (N_prices,)
        Prices corresponding to each column of demand_array
    cash_holdings : ndarray, shape (N_agents,)
        Available cash for each agent
    
    Returns:
    --------
    capped_demand : ndarray, shape (N_agents, N_maturities, N_prices)
        Demand array with overspending capped by available cash
    """
    demand_array = np.asarray(demand_array)
    price_spectrum = np.asarray(price_spectrum)
    cash_holdings = np.asarray(cash_holdings)
    
    N_agents, N_maturities, N_prices = demand_array.shape
    
    # Reshape price_spectrum for broadcasting: (N_prices,) -> (1, 1, N_prices)
    price_grid = price_spectrum[np.newaxis, np.newaxis, :]
    
    # Calculate spending value for each (agent, maturity) taking the max possible spending across prices (worst case)
    spending_array = demand_array * price_grid  # shape (N_agents, N_maturities, N_prices)
    spending_array = np.max(spending_array, axis=2, keepdims=True)  # shape (N_agents, N_maturities, 1)
    
    # Sum spending across all maturities and prices for each agent
    # Shape: (N_agents,)
    total_spending = np.sum(spending_array, axis=(1, 2))

    # As long as cash is greater than the liquidity buffer, they can spend the difference

    cash_available_for_spending = cash_holdings - liqudity_buffer
    cash_available_for_spending = np.maximum(cash_available_for_spending, 0)
    
    # Calculate scale factor for each agent
    # If total_spending <= cash, scale_factor = 1.0
    # If total_spending > cash, scale_factor = cash / total_spending
    scale_factors = np.minimum(1.0, cash_available_for_spending / (total_spending + 1e-10))
    
    # Apply scaling: reshape for broadcasting (N_agents,) -> (N_agents, 1, 1)
    scale_factors_reshaped = scale_factors[:, np.newaxis, np.newaxis]
    capped_demand = demand_array * scale_factors_reshaped
    
    return capped_demand

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
    demand_array = demand_array / demand_array.sum(axis=1, keepdims=True)
    return demand_array




