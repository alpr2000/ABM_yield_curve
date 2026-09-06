"""Compare the example QE simulation with a no-intervention counterfactual.

The complete market model is implemented in example_QE_counterfactual.py.
This file only configures and compares two paired runs of that model.
"""

import numpy as np

from example_QE_counterfactual import run_qe_experiment
from parameters import maturity_spectrum
from utils import plot_tenor_yield_over_time_multiple


SEED = 42
SIMULATION_PERIODS = 10
T_CB = 5


if __name__ == "__main__":
    qe_result = run_qe_experiment(
        apply_qe=True,
        simulation_periods=SIMULATION_PERIODS,
        seed=SEED,
        t_CB=T_CB,
    )
    counterfactual_result = run_qe_experiment(
        apply_qe=False,
        simulation_periods=SIMULATION_PERIODS,
        seed=SEED,
        t_CB=T_CB,
    )

    qe_yields = np.array(qe_result["yield_curves"])
    counterfactual_yields = np.array(counterfactual_result["yield_curves"])
    twelve_month_index = np.argmin(np.abs(maturity_spectrum - 12))
    qe_12m = qe_yields[:, twelve_month_index] * 100
    counterfactual_12m = counterfactual_yields[:, twelve_month_index] * 100

    print("QE 12M yield path:", np.round(qe_12m, 4))
    print("Counterfactual 12M yield path:", np.round(counterfactual_12m, 4))
    print("Final gap (QE - counterfactual):", qe_12m[-1] - counterfactual_12m[-1])
    print("Maximum absolute gap:", np.max(np.abs(qe_12m - counterfactual_12m)))

    plot_tenor_yield_over_time_multiple(
        {
            "QE": qe_result["yield_curves"],
            "Counterfactual": counterfactual_result["yield_curves"],
        },
        maturity_spectrum,
        tenor_months=120,
        title="120M tenor yield: QE vs counterfactual",
    )
