import numpy as np
import matplotlib.pyplot as plt

def get_yield_curve(prices, maturity_spectrum):
    """
    Converts prices to yields for each maturity
    """
    return (100/prices)**(1/maturity_spectrum) - 1

def plot_yield_curves(yield_curves, maturity_spectrum):
    """
    Plots the yield curve for each time step
    """
    plt.figure(figsize=(10, 6))
    for i, yield_curve in enumerate(yield_curves):
        plt.plot(maturity_spectrum, yield_curve, label=f'Time {i}')
    plt.xlabel('Maturity (months)')
    plt.ylabel('Yield')
    plt.title('Yield Curves Over Time')
    plt.legend()
    plt.grid()
    plt.show()

# A function for viewing the holdings of a specific pension fund over the simulation periods
def plot_pf_holdings_over_time(holdings_over_time, pf_index):
    """
    Plots the holdings of a specific pension fund over time
    """
    all_holdings = []
    for holding in holdings_over_time:
        pf_holdings = holding[0]
        pf_specific = pf_holdings[pf_index]
        all_holdings.append(pf_specific)
    plt.figure(figsize=(10, 6))
    plt.plot(all_holdings)

def plot_hf_holdings_over_time(holdings_over_time, hf_index):
    """
    Plots the holdings of a specific hedge fund over time
    """
    all_holdings = []
    for holding in holdings_over_time:
        hf_holdings = holding[1]
        hf_specific = hf_holdings[hf_index]
        all_holdings.append(hf_specific)
    plt.figure(figsize=(10, 6))
    plt.plot(all_holdings)

def plot_pf_liabilities_over_time(liabilities_over_time, pf_index):
    """
    Plots the liabilities of a specific pension fund over time
    """
    all_liabilities = []
    for liability in liabilities_over_time:
        pf_liabilities = liability
        pf_specific = pf_liabilities[pf_index]
        all_liabilities.append(pf_specific)
    plt.figure(figsize=(10, 6))
    plt.plot(all_liabilities)


def plot_pf_cash_over_time(cash_over_time, pf_index):
    """
    Plots the cash of a specific pension fund over time
    """
    all_cash = []
    for cash in cash_over_time:
        pf_cash = cash[0]
        pf_specific = pf_cash[pf_index]
        all_cash.append(pf_specific)
    plt.figure(figsize=(10, 6))
    plt.plot(all_cash)


def get_hf_returns(holdings_over_time, cash_over_time, clearing_prices_over_time):
    """
    Calculates the returns of each hedge fund over time based on their holdings and cash
    """
    hf_returns = []
    for t in range(1, len(holdings_over_time)):
        hf_holdings = holdings_over_time[t][1]
        hf_cash = cash_over_time[t][1]
        hf_holdings_prev = holdings_over_time[t-1][1]
        hf_cash_prev = cash_over_time[t-1][1]
        
        clearing_prices = clearing_prices_over_time[t]
        clearing_prices_prev = clearing_prices_over_time[t-1]

        # Calculate the value of holdings at current and previous time step using actual clearing prices
        hf_value = np.sum(hf_holdings * clearing_prices, axis=1) + hf_cash
        hf_value_prev = np.sum(hf_holdings_prev * clearing_prices_prev, axis=1) + hf_cash_prev

        # Calculate returns
        returns = (hf_value - hf_value_prev) / hf_value_prev
        hf_returns.append(returns)
    
    return np.array(hf_returns)