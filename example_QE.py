import numpy as np
import numpy.random as random


# Load functions from other files
from demand_functions import *
from supply_functions import *
from bond_auction import *
from parameters import *
from utils import *
from market_clearing_functions import *
from cb_and_inflation import *

###### Initial conditions ######

base_rate = 100.2 # This means a 0.2% return over 3 months, which annualizes to approximately 0.8% per year, which is a low interest rate environment

# Low inflation path to warrant a low interest rate environment
#inf = inflation_asymmetric_process(starting_inflation=1.005, i_lr=1.005, phi=0.9, sigma=0.0002, periods=simulation_periods, seed=12)
# Flat inflation path to investigate pure effects of QE
inf = np.ones(simulation_periods) * 1.005

target_inflation = 1.005


# Get taylor rule interest rate path based on inflation path
interest_rate_path = set_interest_rate_taylor_rule(inf, base_rate/100, target_inflation, 1.5)*100


################################
# Initialisation type 1, funds hold a random distribution of bonds across the
# maturity spectrum. 
################################a

# A note on scale:
# Funds hold up to ~100 bonds at each maturity, each of which is priced at an average of ~50
# Their total holdings are therefore of the order of 2e6
# We give them cash holdings of the order 5e5
 
# 20000 bonds at ~50$ are auctioned every period
# = 1,000,000$ in total implying a total government debt
# of around 100$ million = 2,000,000 bonds at ~50.
# PFs own ~30% of total debt = 600,000 bonds
# Over 10 PFs and 120 bins they should hold ~ 500 bonds per bin
# Their cash holdings = 10% of their total holdings = 3,000,000$
# over 10 funds = 300,000$ per fund in cash

# The same logic applies for hedgefunds but at 10%
# Of total debt and 25 funds
# = 200,000 bonds over 25 funds = 8000 bonds per fund
# Over 120 bins = ~ 67 bonds per bin
# Cash holdings = 100,000$ over 25 funds = 4,000$ per fund in cash

###### Pensions funds
#Give initially random but matched liabilities



PF_cash = np.random.uniform(0, 300000, size=N_pension_funds)

PF_liabilities = np.zeros((N_pension_funds, len(maturity_spectrum)))
PF_liabilities = np.random.randint(0, 1000, size=(N_pension_funds, len(maturity_spectrum)))
PF_holdings=PF_liabilities
#Generate new liabilities
#new_liabilities = pf_gen_liability(maturity_spectrum, 5, N_pension_funds)
#PF_liabilities=PF_liabilities+new_liabilities

# PFs get cash proportional to the new liabilities*fair_price of each liability and scaled up by PF_margin
#PF_cash = PF_cash + np.sum(new_liabilities*pf_fair_price(base_rate, term_premium, maturity_spectrum, inf[0], target_inflation), axis=1)*pf_margin

###### Hedge funds and noise traders

HF_holdings = np.random.randint(0, 250, size=(N_hedge_funds, len(maturity_spectrum)))
NT_holdings = np.random.randint(0, 1000, size=(N_noise_traders, len(maturity_spectrum)))

HF_cash = np.random.uniform(0, 50000, size=N_hedge_funds)
NT_cash = np.random.uniform(400000, 500000, size=N_noise_traders)


#####################################
# Central bank intervention
#####################################


t_CB = 10 # Period at which the central bank starts QE
maturity_skew = 0.01 # The more positive this is, the more the central bank will buy longer maturity bonds relative to shorter maturity bonds
skew_constant = 0.5 # The more positive this is, the more the central bank will buy shorter maturity bonds relative to longer maturity bonds
quantity_cb = 100000 # The total quantity of bonds the central bank will buy in the QE program  

####### ALL VARIABLES ARE NOW INITIALISED, WE CAN PROCEED TO SIMULATION STEPS IN main.py

save_yield_curve = True
save_holdings = True
save_liabilities = True
save_cash = True
print_market_stats = True  # Set to True to print primary/secondary market volume and composition each period
save_hf_trades = True  # Set to True to save hedge fund trades for analysis
save_primary_clearing_prices = True  # Set to True to save primary market clearing prices for analysis

yield_curves =[]
holdings_over_time = []
liabilities_over_time = []
cash_over_time = []
clearing_prices_over_time = []
hf_trades_over_time = []
primary_clearing_prices_over_time = []  # List to store primary market clearing prices over time

for i in range(20):
    #print(f"Simulation period {i+1} of {simulation_periods}")


    

    # Get demand and supply

        
    #######################
    #######################
    # DEMAND FUNCTIONS
    #######################
    #######################
    inflation_rate = inf[i]
    base_rate = interest_rate_path[i]

    new_liabilities = pf_gen_liability(maturity_spectrum, 20, N_pension_funds)
    PF_liabilities=PF_liabilities+new_liabilities

    # PFs get cash proportional to the new liabilities*fair_price of each liability and scaled up by PF_margin
    PF_cash = PF_cash + np.sum(new_liabilities*pf_fair_price(base_rate, term_premium, maturity_spectrum, inflation_rate, target_inflation), axis=1)*pf_margin


    ################## Pension Fund Demand Functions ##################
    PF_demand = pf_demand_all_maturities(price_spectrum,
                                        maturity_spectrum,
                                        pf_fair_price(base_rate,
                                                    term_premium,
                                                    maturity_spectrum,
                                                    inflation_rate,
                                                    target_inflation),
                                        PF_liabilities-PF_holdings)

    PF_demand = cap_demand_by_cash(PF_demand, price_spectrum, PF_cash, pf_liquidity_buffer)

    #################### Hedge Fund Demand Functions ##################

    hf_persistence = np.random.uniform(0.9, 0.99, size=N_hedge_funds) # Each HF has a random persistence in their inflation expectations, between 0.9 and 0.99
    hf_long_term_inflation = np.random.uniform(1.002, 1.008, size=N_hedge_funds) # Each HF has a random long term inflation expectation between 0.2% and 0.8% per month

    HF_fair_prices = hf_fund_fair_price_hetero_expect(base_rate,
                                        term_premium,
                                        maturity_spectrum,
                                        N_hedge_funds,
                                        hf_persistence,
                                        hf_long_term_inflation,
                                        inflation_rate)

    HF_demand = hf_demand_all_maturities(price_spectrum,
                                        maturity_spectrum,
                                        HF_fair_prices,
                                        scale_demand,
                                        HF_holdings)

    HF_demand = cap_demand_by_cash(HF_demand, price_spectrum, HF_cash, hf_liquidity_buffer)


    #################### Noise Trader Demand Functions ##################

    NT_fair_prices = nt_fair_price(base_rate, term_premium, maturity_spectrum, inflation_rate, target_inflation)

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



    ##### SECONDARY MARKET CLEARING ########################################################
    
    # Central Bank Intervention
    if i == t_CB:
        cb_demand = cb_demand_QE(quantity_cb, maturity_spectrum, maturity_skew, skew_constant)
    else:
        cb_demand = np.zeros_like(maturity_spectrum)

    #Secondary market clearing
    clearing_prices, PF_holdings_sec, HF_holdings_sec, NT_holdings_sec = clear_secondary_cb(cb_demand, PF_demand, HF_demand, NT_demand, HF_supply, NT_supply, price_spectrum, PF_holdings, HF_holdings, NT_holdings)

    secondary_pf_trade = PF_holdings_sec - PF_holdings
    secondary_hf_trade = HF_holdings_sec - HF_holdings
    secondary_nt_trade = NT_holdings_sec - NT_holdings

    secondary_pf_bought = np.sum(np.maximum(secondary_pf_trade, 0))
    secondary_pf_sold = np.sum(np.maximum(-secondary_pf_trade, 0))
    secondary_hf_bought = np.sum(np.maximum(secondary_hf_trade, 0))
    secondary_hf_sold = np.sum(np.maximum(-secondary_hf_trade, 0))
    secondary_nt_bought = np.sum(np.maximum(secondary_nt_trade, 0))
    secondary_nt_sold = np.sum(np.maximum(-secondary_nt_trade, 0))
    secondary_cb_bought = np.sum(np.maximum(cb_demand, 0))
    secondary_cb_sold = 0

    secondary_total_bought = secondary_pf_bought + secondary_hf_bought + secondary_nt_bought + secondary_cb_bought
    secondary_total_sold = secondary_pf_sold + secondary_hf_sold + secondary_nt_sold + secondary_cb_sold

    secondary_pf_bought_share = secondary_pf_bought / secondary_total_bought if secondary_total_bought > 0 else 0.0
    secondary_pf_sold_share = secondary_pf_sold / secondary_total_sold if secondary_total_sold > 0 else 0.0
    secondary_hf_bought_share = secondary_hf_bought / secondary_total_bought if secondary_total_bought > 0 else 0.0
    secondary_hf_sold_share = secondary_hf_sold / secondary_total_sold if secondary_total_sold > 0 else 0.0
    secondary_cb_bought_share = secondary_cb_bought / secondary_total_bought if secondary_total_bought > 0 else 0.0
    secondary_cb_sold_share = secondary_cb_sold / secondary_total_sold if secondary_total_sold > 0 else 0.0
    secondary_nt_bought_share = secondary_nt_bought / secondary_total_bought if secondary_total_bought > 0 else 0.0
    secondary_nt_sold_share = secondary_nt_sold / secondary_total_sold if secondary_total_sold > 0 else 0.0

    if print_market_stats:
        print("Secondary market trade summary")
        print(f"  Bought: {secondary_total_bought:.0f} | Sold: {secondary_total_sold:.0f}")
        print(f"  PF share of bought/sold: {secondary_pf_bought_share:.1%} / {secondary_pf_sold_share:.1%}")
        print(f"  HF share of bought/sold: {secondary_hf_bought_share:.1%} / {secondary_hf_sold_share:.1%}")
        print(f"  CB share of bought/sold: {secondary_cb_bought_share:.1%} / {secondary_cb_sold_share:.1%}")
        print(f"  NT share of bought/sold: {secondary_nt_bought_share:.1%} / {secondary_nt_sold_share:.1%}")
    
    # Reduce cash based on secondary market trades
    PF_cash = PF_cash - np.sum((PF_holdings_sec - PF_holdings)*clearing_prices, axis=1)
    HF_cash = HF_cash - np.sum((HF_holdings_sec - HF_holdings)*clearing_prices, axis=1)
    NT_cash = NT_cash - np.sum((NT_holdings_sec - NT_holdings)*clearing_prices, axis=1)

    
    

    #Update PF and HF holdings based on primary market auction results
    PF_holdings = PF_holdings_sec
    NT_holdings = NT_holdings_sec
    HF_holdings = HF_holdings_sec

    #Reduce demand for PFs and HFs based on secondary market results
    PF_demand = pf_demand_all_maturities(price_spectrum,
                                     maturity_spectrum,
                                     pf_fair_price(base_rate,
                                                   term_premium,
                                                   maturity_spectrum,
                                                   inflation_rate,
                                                   target_inflation),
                                     PF_liabilities-PF_holdings)
    PF_demand = cap_demand_by_cash(PF_demand, price_spectrum, PF_cash, pf_liquidity_buffer)

    HF_demand = hf_demand_all_maturities(price_spectrum,
                                     maturity_spectrum,
                                     HF_fair_prices,
                                     scale_demand,
                                     HF_holdings)
    HF_demand = cap_demand_by_cash(HF_demand, price_spectrum, HF_cash, hf_liquidity_buffer)

    #Check that the secondary market was zero-sum
    #print("Change in total holdings across all agents (should be 0): ", PF_holdings_sec.sum() + HF_holdings_sec.sum() + NT_holdings_sec.sum() - PF_holdings.sum() - HF_holdings.sum() - NT_holdings.sum())
    


    ############### Primary market auction #####################
    gov_supply = stochastic_bond_supply(gov_auc_q, maturity_spectrum)

    clearing_prices_gov, PF_holdings_prim, HF_holdings_prim = find_clearing_price_auction(PF_demand, HF_demand, PF_holdings, HF_holdings, gov_supply, price_spectrum)

    if save_primary_clearing_prices:
        primary_clearing_prices_over_time.append(clearing_prices_gov)

    primary_pf_trade = PF_holdings_prim - PF_holdings
    primary_hf_trade = HF_holdings_prim - HF_holdings

    primary_pf_bought = np.sum(np.maximum(primary_pf_trade, 0))
    primary_pf_sold = np.sum(np.maximum(-primary_pf_trade, 0))
    primary_hf_bought = np.sum(np.maximum(primary_hf_trade, 0))
    primary_hf_sold = np.sum(np.maximum(-primary_hf_trade, 0))

    primary_total_bought = primary_pf_bought + primary_hf_bought
    primary_total_sold = primary_pf_sold + primary_hf_sold

    primary_pf_bought_share = primary_pf_bought / primary_total_bought if primary_total_bought > 0 else 0.0
    primary_pf_sold_share = primary_pf_sold / primary_total_sold if primary_total_sold > 0 else 0.0
    primary_hf_bought_share = primary_hf_bought / primary_total_bought if primary_total_bought > 0 else 0.0
    primary_hf_sold_share = primary_hf_sold / primary_total_sold if primary_total_sold > 0 else 0.0

    if print_market_stats:
        print(f"\nPeriod {i+1}:")
        print("Primary market trade summary")
        print(f"  Bought: {primary_total_bought:.0f} | Sold: {primary_total_sold:.0f}")
        print(f"  PF share of bought/sold: {primary_pf_bought_share:.1%} / {primary_pf_sold_share:.1%}")
        print(f"  HF share of bought/sold: {primary_hf_bought_share:.1%} / {primary_hf_sold_share:.1%}")
        print(f"  CB share of bought/sold: 0.0% / 0.0%")
        print(f"  NT share of bought/sold: 0.0% / 0.0%")

    # Reduce cash
    PF_cash = PF_cash - np.sum((PF_holdings_prim - PF_holdings)*clearing_prices_gov, axis=1)
    HF_cash = HF_cash - np.sum((HF_holdings_prim - HF_holdings)*clearing_prices_gov, axis=1)


    PF_holdings = PF_holdings_prim
    HF_holdings = HF_holdings_prim

    ##### BONDS MATURE ################################################################
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

    if save_hf_trades:
        hf_trades_over_time.append((secondary_hf_trade.copy()))
# ANALYSIS AND PLOTTING

# Plot yield curves overlayed
plot_yield_curves(yield_curves, maturity_spectrum)

plot_yield_curves_overlayed(yield_curves, maturity_spectrum, time_intervention=t_CB)
plot_tenor_yield_over_time(yield_curves, maturity_spectrum, tenor_months=12)

plot_pf_holdings_over_time(holdings_over_time, pf_index=0)
plot_pf_liabilities_over_time(liabilities_over_time, pf_index=0)
plot_pf_cash_over_time(cash_over_time, pf_index=0)