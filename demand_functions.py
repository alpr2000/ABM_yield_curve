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

################## Pension Fund Demand Function ##################


def pf_gen_liability(maturity_spectrum):
    # Only add liabilities for the second half of the maturity spectrum
    new_liabilities = random.uniform(0, 1, size=len(maturity_spectrum))
    new_liabilities[:len(maturity_spectrum) // 2] = 0
    return new_liabilities

def pf_fair_price(base_rate, term_premium):
    #As all bonds are zero-coupon, the fair price is simply the discounted value of the liability at maturity
    fair_price = (100/base_rate)**(maturity_spectrum)*100


def pf_demand_individual(price_spectrum, fair_price, liability, time_to_liability):
    #Implement a fermi-dirac-esque demand function
    urgency = 0.25/(1+time_to_liability)*fair_price # The more urgent the liability, the higher the urgency factor
    # We assume that if the time to liability is short, the pension fund demands more, pushing their fair price up
    demand_curve = liability/(1 + np.exp((price_spectrum - (fair_price+urgency )/1 )))
    return demand_curve

def pf_demand_all_maturities(price_spectrum, maturity_spectrum, ):


################## Hedge Fund Demand Functions ##################

def hf_demand_single(price_spectrum, fair_price, funds, scale_demand):
    demand_curve = -price_spectrum + fair_price
    demand_curve = (scale_demand*demand_curve) ** 2
    # Cut off demand at fair price < price and above the funds available
    demand_curve[price_spectrum > fair_price] = 0
    demand_curve = np.minimum(demand_curve, funds)
    return demand_curve

def hf_curve_random(fair_price, maturity_spectrum, hf_random_type, hf_heterogeneity):
    if hf_random_type == 0:
        # Simple slope deviation
        random_slope = random.uniform(1-hf_heterogeneity, 1+hf_heterogeneity)
        random_curve = random_slope**maturity_spectrum * fair_price
        return random_curve
    else:
        raise ValueError("Invalid hedge fund random type")

# Create function to create a hedge fund proprietary pricing curve based on the current base rate and a random seed
def hf_fund_fair_price(base_rate, term_premium, maturity_spectrum):
    # Base curve is simply the fair price curve based on the current base rate
    fair_price = (100/base_rate)**(maturity_spectrum)*100
    # Add some term premium
    term_premium_curve = (1+term_premium)**maturity_spectrum
    fair_price = fair_price/term_premium_curve  

    # Now we add a random component to the pricing curve
    fair_price = fair_price*hf_curve_random(fair_price, maturity_spectrum, hf_random_type, hf_heterogeneity)/100

    return fair_price



