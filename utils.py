def get_yield_curve(prices, maturity_spectrum):
    """
    Converts prices to yields for each maturity
    """
    return (100/prices)**(1/maturity_spectrum) - 1