import numpy as np

# INPUTS
################################
N_years = 30
price_granularity = 1

N_pension_funds = 10
N_hedge_funds = 25
N_noise_traders = 100

scale_demand = 1

term_premium = 0.002 #monthly term premium

hf_random_type = 0
# 0: Simple slope deviation

hf_heterogeneity = 0.001

########################################

maturity_spectrum = np.linspace(1, 12 * N_years, 12 * N_years)  # Maturities from 1 to 30 years

price_spectrum = np.arange(0, 200, price_granularity)  # Price spectrum from 0 to 200 with specified granularity

