import numpy as np

from demand_functions import hf_fund_fair_price_hetero_expect, hf_demand_all_maturities


def test_hf_fair_price_includes_upward_bias():
    base_rate = 100.5
    term_premium = 0.000008
    maturity_spectrum = np.array([3, 6])
    hf_persistence = np.array([0.95])
    hf_long_term_inflation = np.array([1.005])
    inflation_rate = 1.005

    unbiased = hf_fund_fair_price_hetero_expect(
        base_rate,
        term_premium,
        maturity_spectrum,
        1,
        hf_persistence,
        hf_long_term_inflation,
        inflation_rate,
        hf_fair_price_adjustment=0.0,
    )
    biased = hf_fund_fair_price_hetero_expect(
        base_rate,
        term_premium,
        maturity_spectrum,
        1,
        hf_persistence,
        hf_long_term_inflation,
        inflation_rate,
        hf_fair_price_adjustment=0.05,
    )

    assert np.all(biased > unbiased)


def test_hf_demand_is_stronger_than_price_gap_only():
    price_spectrum = np.array([90.0, 95.0, 100.0])
    maturity_spectrum = np.array([3, 6])
    fair_prices = np.array([[95.0, 97.0], [95.0, 97.0]])
    hf_holdings = np.array([[1000, 1000], [1000, 1000]])

    demand = hf_demand_all_maturities(price_spectrum, maturity_spectrum, fair_prices, 1.0, hf_holdings)

    assert demand.shape == (2, 2, 3)
    assert np.any(demand > 0)
