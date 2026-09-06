import numpy as np
import matplotlib.pyplot as plt

from demand_functions import *
from supply_functions import *
from bond_auction import *
from parameters import *
from utils import *
from market_clearing_functions import *
from cb_and_inflation import *


def run_qe_experiment(apply_qe=True, simulation_periods=25, seed=42, t_CB=10,
                     maturity_skew=0.01, skew_constant=0.5, quantity_cb=100000):
    """Follow the same flow as example_QE.py, but allow QE to be switched on/off."""
    np.random.seed(seed)

    base_rate = 100.2
    inf = np.ones(simulation_periods) * 1.005
    target_inflation = 1.005
    interest_rate_path = set_interest_rate_taylor_rule(inf, base_rate / 100, target_inflation, 1.5) * 100

    PF_cash = np.random.uniform(0, 300000, size=N_pension_funds)
    PF_liabilities = np.zeros((N_pension_funds, len(maturity_spectrum)))
    PF_liabilities = np.random.randint(0, 1000, size=(N_pension_funds, len(maturity_spectrum)))
    PF_holdings = PF_liabilities.copy()

    HF_holdings = np.random.randint(0, 250, size=(N_hedge_funds, len(maturity_spectrum)))
    NT_holdings = np.random.randint(0, 1000, size=(N_noise_traders, len(maturity_spectrum)))

    HF_cash = np.random.uniform(0, 50000, size=N_hedge_funds)
    NT_cash = np.random.uniform(400000, 500000, size=N_noise_traders)

    yield_curves = []

    for i in range(simulation_periods):
        inflation_rate = inf[i]
        base_rate = interest_rate_path[i]

        new_liabilities = pf_gen_liability(maturity_spectrum, 20, N_pension_funds)
        PF_liabilities = PF_liabilities + new_liabilities
        PF_cash = PF_cash + np.sum(
            new_liabilities * pf_fair_price(base_rate, term_premium, maturity_spectrum, inflation_rate, target_inflation),
            axis=1,
        ) * pf_margin

        PF_demand = pf_demand_all_maturities(
            price_spectrum,
            maturity_spectrum,
            pf_fair_price(base_rate, term_premium, maturity_spectrum, inflation_rate, target_inflation),
            PF_liabilities - PF_holdings,
        )
        PF_demand = cap_demand_by_cash(PF_demand, price_spectrum, PF_cash, pf_liquidity_buffer)

        hf_persistence = np.random.uniform(0.9, 0.99, size=N_hedge_funds)
        hf_long_term_inflation = np.random.uniform(1.002, 1.008, size=N_hedge_funds)

        HF_fair_prices = hf_fund_fair_price_hetero_expect(
            base_rate,
            term_premium,
            maturity_spectrum,
            N_hedge_funds,
            hf_persistence,
            hf_long_term_inflation,
            inflation_rate,
            hf_fair_price_adjustment=hf_fair_price_adjustment,
        )

        HF_demand = hf_demand_all_maturities(price_spectrum, maturity_spectrum, HF_fair_prices, scale_demand, HF_holdings)
        HF_demand = cap_demand_by_cash(HF_demand, price_spectrum, HF_cash, hf_liquidity_buffer)

        NT_fair_prices = nt_fair_price(base_rate, term_premium, maturity_spectrum, inflation_rate, target_inflation)
        NT_demand = noise_trader_demand(price_spectrum, NT_fair_prices, scale_noise_trader, NT_cash, nt_cash_perc)

        HF_supply = hf_supply_all_maturities(price_spectrum, maturity_spectrum, HF_fair_prices, scale_demand, HF_holdings)
        NT_supply = nt_supply_all_maturities(price_spectrum, maturity_spectrum, NT_holdings, NT_fair_prices, nt_cash_perc)

        if apply_qe and i >= t_CB:
            cb_demand = cb_demand_QE(quantity_cb, maturity_spectrum, maturity_skew, skew_constant)
        else:
            cb_demand = np.zeros_like(maturity_spectrum)

        clearing_prices, PF_holdings_sec, HF_holdings_sec, NT_holdings_sec = clear_secondary_cb(
            cb_demand,
            PF_demand,
            HF_demand,
            NT_demand,
            HF_supply,
            NT_supply,
            price_spectrum,
            PF_holdings,
            HF_holdings,
            NT_holdings,
        )

        PF_cash = PF_cash - np.sum((PF_holdings_sec - PF_holdings) * clearing_prices, axis=1)
        HF_cash = HF_cash - np.sum((HF_holdings_sec - HF_holdings) * clearing_prices, axis=1)
        NT_cash = NT_cash - np.sum((NT_holdings_sec - NT_holdings) * clearing_prices, axis=1)

        PF_holdings = PF_holdings_sec
        NT_holdings = NT_holdings_sec
        HF_holdings = HF_holdings_sec

        PF_demand = pf_demand_all_maturities(
            price_spectrum,
            maturity_spectrum,
            pf_fair_price(base_rate, term_premium, maturity_spectrum, inflation_rate, target_inflation),
            PF_liabilities - PF_holdings,
        )
        PF_demand = cap_demand_by_cash(PF_demand, price_spectrum, PF_cash, pf_liquidity_buffer)

        HF_demand = hf_demand_all_maturities(price_spectrum, maturity_spectrum, HF_fair_prices, scale_demand, HF_holdings)
        HF_demand = cap_demand_by_cash(HF_demand, price_spectrum, HF_cash, hf_liquidity_buffer)

        gov_supply = stochastic_bond_supply(gov_auc_q, maturity_spectrum)
        clearing_prices_gov, PF_holdings_prim, HF_holdings_prim = find_clearing_price_auction(
            PF_demand,
            HF_demand,
            PF_holdings,
            HF_holdings,
            gov_supply,
            price_spectrum,
        )

        PF_cash = PF_cash - np.sum((PF_holdings_prim - PF_holdings) * clearing_prices_gov, axis=1)
        HF_cash = HF_cash - np.sum((HF_holdings_prim - HF_holdings) * clearing_prices_gov, axis=1)

        PF_holdings = PF_holdings_prim
        HF_holdings = HF_holdings_prim

        PF_cash += 100 * PF_holdings[:, 0]
        HF_cash += 100 * HF_holdings[:, 0]
        NT_cash += 100 * NT_holdings[:, 0]

        PF_cash -= 100 * PF_liabilities[:, 0]

        PF_holdings[:, :-1] = PF_holdings[:, 1:]
        PF_holdings[:, -1] = 0
        HF_holdings[:, :-1] = HF_holdings[:, 1:]
        HF_holdings[:, -1] = 0
        NT_holdings[:, :-1] = NT_holdings[:, 1:]
        NT_holdings[:, -1] = 0

        PF_liabilities[:, :-1] = PF_liabilities[:, 1:]
        PF_liabilities[:, -1] = 0

        yield_curve = get_yield_curve(clearing_prices_gov, maturity_spectrum)
        yield_curves.append(yield_curve)

    return {"yield_curves": yield_curves, "inflation_path": inf, "interest_rate_path": interest_rate_path}


def get_12m_yield_series(yield_curves, maturity_grid):
    idx = np.argmin(np.abs(maturity_grid - 12))
    return np.array([yc[idx] * 100 for yc in yield_curves])


def plot_qe_vs_counterfactual(qe_series, cf_series, intervention_period=10):
    periods = np.arange(len(qe_series))
    plt.figure(figsize=(10, 5))
    plt.plot(periods, qe_series, label="QE", color="tab:blue", marker="o")
    plt.plot(periods, cf_series, label="Counterfactual", color="tab:orange", marker="s", linestyle="--")
    plt.axvline(x=intervention_period, color="gray", linestyle=":", linewidth=1, label="QE starts")
    plt.xlabel("Period")
    plt.ylabel("12M yield (%)")
    plt.title("QE vs counterfactual: 12M yield")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    qe_result = run_qe_experiment(apply_qe=True, simulation_periods=25, seed=42, t_CB=10)
    counterfactual_result = run_qe_experiment(apply_qe=False, simulation_periods=25, seed=42, t_CB=10)

    qe_series = get_12m_yield_series(qe_result["yield_curves"], maturity_spectrum)
    cf_series = get_12m_yield_series(counterfactual_result["yield_curves"], maturity_spectrum)

    print("QE 12M yield path:", np.round(qe_series, 4))
    print("Counterfactual 12M yield path:", np.round(cf_series, 4))
    print("Final gap (QE - CF):", qe_series[-1] - cf_series[-1])
    print("Max absolute gap:", np.max(np.abs(qe_series - cf_series)))

    plot_qe_vs_counterfactual(qe_series, cf_series, intervention_period=10)
