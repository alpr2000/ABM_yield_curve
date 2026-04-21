import numpy as np
import numpy.random as random

"""
Gilt market agent based model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This model simulates five agents:

1. Government: Inititates government bond auctions

2. Central Bank: Set interest rates, engages in OMO in the advanced model

3. "Pension Funds": Representing LDI, the buy in auctions and operate in the secondary market

4. "Hedge Funds": Representing leveraged investors, they only operate in the secondary market, profit motivated

5. "Noise Traders": Only operating in the secondary market, they trade randomly, representing retail investors and other non-professional market participants

Step 1: (Every Month 1-12)
    Central Bank sets interest rates based on a Taylor rule and a stochastic inflation rate based on a simple AR(1) process

Step 2: (Every Month 1)
    Government initiates bond auction, pension funds buy in based on their demand function

Step 3: (Every Month)
    Secondary market trading occurs, with pension funds, hedge funds, and noise traders all participating based on their respective strategies

Step 4: (Optional)
    The central bank may engage in open market operations (OMO) which will be executed as auctions in the secondary market, with pension funds and hedge funds participating based on their demand functions

A NOTE ON SCALE:
This model considers only zero-coupon bonds of an arbitrary value of 100.
Bond holdings are expressed in term of the number of bonds


"""

# Load functions from other files
from demand_functions import *
from supply_functions import *
from bond_auction import *
from parameters import *

###### Initialial conditions ######

base_rate = 100.2 # This means a 0.2% return over 1 month, which annualizes to approximately 2.43%


######
# Generate empty arrays for cash holdings and bond holdings
######

PF_cash = np.zeros(N_pension_funds)
PF_bonds = np.zeros((N_pension_funds, len(maturity_spectrum)))

HF_cash = np.zeros(N_hedge_funds)
HF_bonds = np.zeros((N_hedge_funds, len(maturity_spectrum)))

NT_cash = np.zeros(N_noise_traders)
NT_bonds = np.zeros((N_noise_traders, len(maturity_spectrum)))

################################
# Initialisation type 1, funds hold a random distribution of bonds across the
# maturity spectrum. 
################################

PF_bonds = np.random.uniform(0, 100, size=(N_pension_funds, len(maturity_spectrum)))
HF_bonds = np.random.uniform(0, 100, size=(N_hedge_funds, len(maturity_spectrum)))
NT_bonds = np.random.uniform(0, 100, size=(N_noise_traders, len(maturity_spectrum)))

PF_cash = np.random.uniform(0, 10000, size=N_pension_funds)
HF_cash = np.random.uniform(0, 10000, size=N_hedge_funds)
NT_cash = np.random.uniform(0, 10000, size=N_noise_traders)

PF_liabilities = np.zeros((N_pension_funds, len(maturity_spectrum)))
PF_liabilities = np.random.uniform(0, 100, size=(N_pension_funds, len(maturity_spectrum)))


################################
# Generate the demand functions for each agent type
################################

################## Pension Fund Demand Functions ##################
PF_demand = pf_demand_all_maturities(price_spectrum, maturity_spectrum, pf_fair_price(base_rate, term_premium, maturity_spectrum), PF_liabilities)


#################### Hedge Fund Demand Functions ##################

HF_fair_prices = hf_fund_fair_price(base_rate, term_premium, maturity_spectrum)

HF_demand = hf_demand_all_maturities(price_spectrum,
                                     maturity_spectrum,
