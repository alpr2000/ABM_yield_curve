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