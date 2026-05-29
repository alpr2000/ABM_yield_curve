import numpy as np

# INPUTS
################################
simulation_periods = 10

N_years = 30
price_granularity = 0.01

N_pension_funds = 10
N_hedge_funds = 25
N_noise_traders = 1 # Can use 1 representative noise trader since they have no strategic behavior
scale_demand = 1

term_premium = 0.000002 #monthly term premium

hf_random_type = 0
# 0: Simple slope deviation

hf_heterogeneity = 0.0001
hf_liquidity_buffer = 0.2

pf_margin = 1.25
pf_liquidity_buffer =0.2

nt_cash_perc = 0.25
scale_noise_trader = 1000

#Government bond parameters
gov_auc_q = 100000

########################################

maturity_spectrum = np.linspace(1, 12 * N_years, 12 * N_years)  # Maturities from 1 to 30 years

price_spectrum = np.arange(0, 125, price_granularity)  # Price spectrum from 0 to 200 with specified granularity

