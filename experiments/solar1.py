import numpy as np
from methods import SolarMethod
from subsolvers import NelderMead
from test_functions import Quadratic
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


subsolver = NelderMead()
method = SolarMethod(subsolver, base_n=1, subsolver_iterations=10)

N = 20000
runs = 5

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 16})

title_printed = False
for K in [1, 10, 100, 1000]:
    suboptimality_logs = []
    suboptimality_log_best = None
    suboptimality_log_worst = None

    for _ in range(runs):
        _, time_log, function_log = method.optimise(
            Quadratic.f, None, Quadratic.initial_point(), 
            N, is_feasible=Quadratic.is_feasible, 
            base_changes=K, verbose=True
        )

        suboptimality_log = np.array(function_log) - Quadratic.solution()[1]
        suboptimality_logs.append(suboptimality_log)

        if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
            suboptimality_log_worst = suboptimality_log
        elif suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
            suboptimality_log_best = suboptimality_log

    mean = np.mean(suboptimality_logs, axis=0)

    if not title_printed:
        N_shift = 500
        N_beginning = 10000
        suboptimality_beginning = np.log10(mean[N_shift:N_beginning])
        alpha = curve_fit(lambda x, a, b: a * x + b, xdata = np.array(list(range(N_shift, N_beginning))), ydata = suboptimality_beginning)[0]

    if title_printed:
        plt.plot(range(N+1), mean, label=f"$K = {K}$")
    else:
        plt.plot(range(N+1), mean, label=f"Solar method\n$K = {K}$")
        plt.plot(range(N+1), 10**(alpha[0] * np.array(list(range(N + 1))) + alpha[1]), "k--", label=f"$\\alpha = {-alpha[0]:.1e}$")
        title_printed = True
    
    # std = np.std(suboptimality_logs, axis=0)
    # plt.fill_between(range(N+1), mean - std, mean + std, alpha=0.2)
    plt.fill_between(range(N+1), suboptimality_log_best, suboptimality_log_worst, alpha=0.4)

plt.title(f"Quadratic problem, $\\mu/L = {Quadratic.condition_number:.1e}$")

plt.xlabel("$N$, iteration number")
plt.ylabel("$f(x^N) - f^*$ (log. scale)")
plt.yscale("log")

plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("figures/solar_quadratic_different_K.pdf")
plt.show()
