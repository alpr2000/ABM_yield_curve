import numpy as np
import numpy.random as random


def clear_secondary(PF_demands, HF_demands, NT_demands, HF_supplys, NT_supplys, price_spectrum, PF_holdings, HF_holdings, NT_holdings):
    """
    Finds the market clearing price for each maturity by summing demand and supply across all agents
    Returns the price at which demand equals supply for each maturity
    """
    NT_demands = NT_demands[np.newaxis, :, :]
    NT_supplys = NT_supplys[np.newaxis, :, :]

    #Add along the maturity and price dimensions to get total demand and supply for each maturity and price
    total_demand = PF_demands.sum(axis=0) + HF_demands.sum(axis=0) + NT_demands.sum(axis=0)

    total_supply = HF_supplys.sum(axis=0) + NT_supplys.sum(axis=0)

    # Find the price where demand equals supply for each maturity
    # Note we take the price where demand ~ supply but strictly demand<supply
    demand_supply = total_demand - total_supply
    demand_supply[demand_supply < 0] = np.inf  # We only want to consider prices where demand >= supply
    price_indices = np.argmin(np.abs(demand_supply), axis=1)

    clearing_prices = price_spectrum[price_indices]

    # Find the change in holdings for each agent based on the clearing price
    PF_trades = PF_demands[:, np.arange(len(price_indices)), price_indices]
    PF_trades[:,total_supply[:, 0] == 0] = 0
    PF_trades[:,total_demand[:, 0] == 0] = 0
    
    HF_trades = HF_demands[:, np.arange(len(price_indices)), price_indices] - HF_supplys[:, np.arange(len(price_indices)), price_indices]
    HF_trades[:,total_supply[:, 0] == 0] = 0
    HF_trades[:,total_demand[:, 0] == 0] = 0

    NT_trades = NT_demands[:, np.arange(len(price_indices)), price_indices] - NT_supplys[:, np.arange(len(price_indices)), price_indices]
    NT_trades[:,total_supply[:, 0] == 0] = 0
    NT_trades[:,total_demand[:, 0] == 0] = 0

    # Trade at clearing prices, randomly allocating trades to demanders and suppliers if there is excess demand or supply at the clearing price
    #PF_trades = np.zeros_like(PF_holdings)
    #HF_trades = np.zeros_like(HF_holdings)
    #NT_trades = np.zeros_like(NT_holdings)

    surplus_demand = demand_supply[np.arange(len(demand_supply)), price_indices]
    
    # Reduce 

    # Update holdings based on trades
    PF_holdings = PF_holdings + PF_trades
    HF_holdings = HF_holdings + HF_trades
    NT_holdings = NT_holdings + NT_trades

    return clearing_prices, PF_holdings, HF_holdings, NT_holdings