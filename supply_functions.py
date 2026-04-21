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
