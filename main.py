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
from utils import *
from market_clearing_functions import *

###### Initial conditions ######

base_rate = 100.6 # This means a 0.6% return over 3 months, which annualizes to approximately 2.41%


################################
# Initialisation type 1, funds hold a random distribution of bonds across the
# maturity spectrum. 
################################

# A note on scale:
# Funds hold up to ~100 bonds at each maturity, each of which is priced at an average of ~50
# Their total holdings are therefore of the order of 2e6
# We give them cash holdings of the order 5e5




###### Pensions funds
#Give initially random but matched liabilities

PF_cash = np.random.uniform(0, 500000, size=N_pension_funds)

PF_liabilities = np.zeros((N_pension_funds, len(maturity_spectrum)))
PF_liabilities = np.random.randint(0, 100, size=(N_pension_funds, len(maturity_spectrum)))
PF_holdings=PF_liabilities
#Generate new liabilities
new_liabilities = pf_gen_liability(maturity_spectrum, 5, N_pension_funds)
PF_liabilities=PF_liabilities+new_liabilities

# PFs get cash proportional to the new liabilities*fair_price of each liability and scaled up by PF_margin
PF_cash = PF_cash + np.sum(new_liabilities*pf_fair_price(base_rate, term_premium, maturity_spectrum), axis=1)*pf_margin

###### Hedge funds and noise traders

HF_holdings = np.random.randint(0, 100, size=(N_hedge_funds, len(maturity_spectrum)))
NT_holdings = np.random.randint(0, 100, size=(N_noise_traders, len(maturity_spectrum)))

HF_cash = np.random.uniform(0, 500000, size=N_hedge_funds)
NT_cash = np.random.uniform(400000, 500000, size=N_noise_traders)


####### ALL VARIABLES ARE NOW INITIALISED, WE CAN PROCEED TO SIMULATION STEPS IN main.py

save_yield_curve = True
save_holdings = True
save_liabilities = True
save_cash = True

yield_curves =[]
holdings_over_time = []
liabilities_over_time = []
cash_over_time = []
clearing_prices_over_time = []

for i in range(simulation_periods):
    #print(f"Simulation period {i+1} of {simulation_periods}")
    

    # Get demand and supply

        
    #######################
    #######################
    # DEMAND FUNCTIONS
    #######################
    #######################

    new_liabilities = pf_gen_liability(maturity_spectrum, 5, N_pension_funds)
    PF_liabilities=PF_liabilities+new_liabilities

    # PFs get cash proportional to the new liabilities*fair_price of each liability and scaled up by PF_margin
    PF_cash = PF_cash + np.sum(new_liabilities*pf_fair_price(base_rate, term_premium, maturity_spectrum), axis=1)*pf_margin


    ################## Pension Fund Demand Functions ##################
    PF_demand = pf_demand_all_maturities(price_spectrum,
                                        maturity_spectrum,
                                        pf_fair_price(base_rate,
                                                    term_premium,
                                                    maturity_spectrum),
                                        PF_liabilities-PF_holdings)

    PF_demand = cap_demand_by_cash(PF_demand, price_spectrum, PF_cash, pf_liquidity_buffer)

    #################### Hedge Fund Demand Functions ##################

    HF_fair_prices = hf_fund_fair_price(base_rate,
                                        term_premium,
                                        maturity_spectrum,
                                        N_hedge_funds)

    HF_demand = hf_demand_all_maturities(price_spectrum,
                                        maturity_spectrum,
                                        HF_fair_prices,
                                        scale_demand,
                                        HF_holdings)

    HF_demand = cap_demand_by_cash(HF_demand, price_spectrum, HF_cash, hf_liquidity_buffer)


    #################### Noise Trader Demand Functions ##################

    NT_fair_prices = nt_fair_price(base_rate, term_premium, maturity_spectrum)

    NT_demand = noise_trader_demand(price_spectrum, NT_fair_prices, scale_noise_trader, NT_cash, nt_cash_perc)


    #######################
    #######################
    # SUPPLY FUNCTIONS
    #######################
    #######################

    #################### Hedge Fund Supply Functions ##################

    HF_supply = hf_supply_all_maturities(price_spectrum,
                                        maturity_spectrum,
                                        HF_fair_prices,
                                        scale_demand,
                                        HF_holdings)


    #################### Noise Trader Supply Functions ##################

    NT_supply = nt_supply_all_maturities(price_spectrum,
                                        maturity_spectrum,
                                        NT_holdings, 
                                        NT_fair_prices,
                                        nt_cash_perc)




    gov_supply = stochastic_bond_supply(gov_auc_q, maturity_spectrum)
    #Primary market auction
    clearing_prices_gov, PF_holdings_1, HF_holdings_1 = find_clearing_price_auction(PF_demand, HF_demand, PF_holdings, HF_holdings, gov_supply, price_spectrum)

    # Reduce cash
    PF_cash = PF_cash - np.sum((PF_holdings_1 - PF_holdings)*clearing_prices_gov, axis=1)
    HF_cash = HF_cash - np.sum((HF_holdings_1 - HF_holdings)*clearing_prices_gov, axis=1)


    #Update PF and HF holdings based on primary market auction results
    PF_holdings = PF_holdings_1
    HF_holdings = HF_holdings_1

    #Reduce demand for PFs and HFs based on primary market results
    PF_demand = pf_demand_all_maturities(price_spectrum,
                                     maturity_spectrum,
                                     pf_fair_price(base_rate,
                                                   term_premium,
                                                   maturity_spectrum),
                                     PF_liabilities-PF_holdings)
    PF_demand = cap_demand_by_cash(PF_demand, price_spectrum, PF_cash, pf_liquidity_buffer)

    HF_demand = hf_demand_all_maturities(price_spectrum,
                                     maturity_spectrum,
                                     HF_fair_prices,
                                     scale_demand,
                                     HF_holdings)
    HF_demand = cap_demand_by_cash(HF_demand, price_spectrum, HF_cash, hf_liquidity_buffer)

    #Secondary market clearing
    clearing_prices, PF_holdings2, HF_holdings2, NT_holdings2 = clear_secondary(PF_demand, HF_demand, NT_demand, HF_supply, NT_supply, price_spectrum, PF_holdings, HF_holdings, NT_holdings)
    

    #Check that the secondary market was zero-sum
    #print("Change in total holdings across all agents (should be 0): ", PF_holdings2.sum() + HF_holdings2.sum() + NT_holdings2.sum() - PF_holdings.sum() - HF_holdings.sum() - NT_holdings.sum())
    
    
    PF_holdings = PF_holdings2
    HF_holdings = HF_holdings2
    NT_holdings = NT_holdings2

    ##### BONDS MATURE #####
    PF_cash += 100*PF_holdings[:,0]
    HF_cash += 100*HF_holdings[:,0]
    NT_cash += 100*NT_holdings[:,0]

    PF_cash -= 100*PF_liabilities[:,0]

    #Shift holdings down one month, with zero holdings at the longest maturity
    PF_holdings[:, :-1] = PF_holdings[:, 1:]
    PF_holdings[:, -1] = 0
    HF_holdings[:, :-1] = HF_holdings[:, 1:]
    HF_holdings[:, -1] = 0
    NT_holdings[:, :-1] = NT_holdings[:, 1:]
    NT_holdings[:, -1] = 0
    
    # Also shift liabilities to match
    PF_liabilities[:, :-1] = PF_liabilities[:, 1:]
    PF_liabilities[:, -1] = 0

    
    if save_yield_curve:
        yield_curve = get_yield_curve(clearing_prices, maturity_spectrum)
        yield_curves.append(yield_curve)
    
    if save_holdings:
        holdings_over_time.append((PF_holdings.copy(), HF_holdings.copy(), NT_holdings.copy()))

    if save_liabilities:
        liabilities_over_time.append(PF_liabilities.copy())
    
    clearing_prices_over_time.append(clearing_prices.copy())

    if save_cash:
        cash_over_time.append((PF_cash.copy(), HF_cash.copy(), NT_cash.copy()))

# ANALYSIS AND PLOTTING

# Plot yield curves overlayed
plot_yield_curves(yield_curves, maturity_spectrum)

plot_pf_holdings_over_time(holdings_over_time, pf_index=0)
plot_pf_liabilities_over_time(liabilities_over_time, pf_index=0)
plot_pf_cash_over_time(cash_over_time, pf_index=0)