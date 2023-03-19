import numpy as np
from methods import SolarMethod
from subsolvers import NelderMead
from test_functions import Quadratic, Quadratic25, Quadratic50
import matplotlib.pyplot as plt

N = 5000
K = 25
runs = 3

fig, ax = plt.subplots(figsize=(8, 6))
fig.subplots_adjust(right=0.75)
plt.rcParams.update({'font.size': 16}, )

twin1 = ax.twinx()
twin2 = ax.twinx()
twin2.spines.right.set_position(("axes", 1.2))

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(16)

for item in ([twin1.title, twin1.xaxis.label, twin1.yaxis.label] +
             twin1.get_xticklabels() + twin1.get_yticklabels()):
    item.set_fontsize(16)

for item in ([twin2.title, twin2.xaxis.label, twin2.yaxis.label] +
             twin2.get_xticklabels() + twin2.get_yticklabels()):
    item.set_fontsize(16)

# Quadratic10 runs

Bs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
log = []
log_err_lower = []
log_err_upper = []

for B in Bs:
    subsolver = NelderMead()
    method = SolarMethod(subsolver, base_n=B, subsolver_iterations=10)

    suboptimality_logs = []
    suboptimality_log_best = None
    suboptimality_log_worst = None

    for _ in range(runs):
        _, time_log, function_log = method.optimise(
            Quadratic.f, None, Quadratic.initial_point(), 
            N, is_feasible=Quadratic.is_feasible, 
            base_changes=K, verbose=True
        )

        suboptimality_log = (np.array(function_log) - Quadratic.solution()[1]) / (function_log[0] - Quadratic.solution()[1])
        suboptimality_logs.append(suboptimality_log)

        if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
            suboptimality_log_worst = suboptimality_log
        if suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
            suboptimality_log_best = suboptimality_log

    mean = np.mean(suboptimality_logs, axis=0)
    log.append(mean[-1])
    log_err_lower.append(mean[-1]-suboptimality_log_best[-1])
    log_err_upper.append(suboptimality_log_worst[-1]-mean[-1])

p1 = ax.errorbar(np.array(Bs) / Quadratic.n, log, 
             yerr=[log_err_lower, log_err_upper], 
             label=f"$n={Quadratic.n}$, $\\mu/L = {Quadratic.condition_number:.1e}$",
             color="C0")


# Quadratic25 runs

Bs = [1, 2, 5, 7, 10, 12, 15, 17, 20, 22, 24]
log = []
log_err_lower = []
log_err_upper = []

for B in Bs:
    subsolver = NelderMead()
    method = SolarMethod(subsolver, base_n=B, subsolver_iterations=10)

    suboptimality_logs = []
    suboptimality_log_best = None
    suboptimality_log_worst = None

    for _ in range(runs):
        _, time_log, function_log = method.optimise(
            Quadratic25.f, None, Quadratic25.initial_point(), 
            N, is_feasible=Quadratic25.is_feasible, 
            base_changes=K, verbose=True
        )

        suboptimality_log = (np.array(function_log) - Quadratic25.solution()[1]) / (function_log[0] - Quadratic25.solution()[1])
        suboptimality_logs.append(suboptimality_log)

        if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
            suboptimality_log_worst = suboptimality_log
        if suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
            suboptimality_log_best = suboptimality_log

    mean = np.mean(suboptimality_logs, axis=0)
    log.append(mean[-1])
    log_err_lower.append(mean[-1]-suboptimality_log_best[-1])
    log_err_upper.append(suboptimality_log_worst[-1]-mean[-1])

p2 = twin1.errorbar(np.array(Bs) / Quadratic25.n, log, 
             yerr=[log_err_lower, log_err_upper], 
             label=f"$n={Quadratic25.n}$, $\\mu/L = {Quadratic25.condition_number:.1e}$",
             color="C1")


# Quadratic50 runs

Bs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
log = []
log_err_lower = []
log_err_upper = []

for B in Bs:
    subsolver = NelderMead()
    method = SolarMethod(subsolver, base_n=B, subsolver_iterations=10)

    suboptimality_logs = []
    suboptimality_log_best = None
    suboptimality_log_worst = None

    for _ in range(runs):
        _, time_log, function_log = method.optimise(
            Quadratic50.f, None, Quadratic50.initial_point(), 
            N, is_feasible=Quadratic50.is_feasible, 
            base_changes=K, verbose=True
        )

        suboptimality_log = (np.array(function_log) - Quadratic50.solution()[1]) / (function_log[0] - Quadratic50.solution()[1])
        suboptimality_logs.append(suboptimality_log)

        if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
            suboptimality_log_worst = suboptimality_log
        if suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
            suboptimality_log_best = suboptimality_log

    mean = np.mean(suboptimality_logs, axis=0)
    log.append(mean[-1])
    log_err_lower.append(mean[-1]-suboptimality_log_best[-1])
    log_err_upper.append(suboptimality_log_worst[-1]-mean[-1])

p3 = twin2.errorbar(np.array(Bs) / Quadratic50.n, log, 
             yerr=[log_err_lower, log_err_upper], 
             label=f"$n={Quadratic50.n}$, $\\mu/L = {Quadratic50.condition_number:.1e}$",
             color="C2")


# Plotting

ax.set_title(f"Quadratic problem, $N={N}$")

ax.set_xlabel("$B / N$, base variables proportion")
ax.set_ylabel("$(f(x^N) - f^*) / (f(x^0) - f^*)$ (log. scale)")
ax.set_yscale("log")
twin1.set_yscale("log")
twin2.set_yscale("log")

ax.yaxis.label.set_color(p1[0].get_color())
twin1.yaxis.label.set_color(p2[0].get_color())
twin2.yaxis.label.set_color(p3[0].get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1[0].get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2[0].get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3[0].get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2, p3], loc="upper left")

ax.grid(alpha=0.4)
fig.tight_layout()
fig.savefig("figures/solar_quadratic_different_B.pdf")
plt.show()
