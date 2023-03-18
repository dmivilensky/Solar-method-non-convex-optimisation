import numpy as np
from methods import SolarMethod, ZOGradientDescent, MomentumThreePoint
from subsolvers import NelderMead
from test_functions import Quadratic
import matplotlib.pyplot as plt


N = 20000
K = 100
runs = 5

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 16})

# Solar method runs

subsolver = NelderMead()
solar_method = SolarMethod(subsolver, base_n=1, subsolver_iterations=10)

time_logs = []
suboptimality_logs = []
suboptimality_log_best = None
suboptimality_log_worst = None

for _ in range(runs):
    _, time_log, function_log = solar_method.optimise(
        Quadratic.f, None, Quadratic.initial_point(), 
        N, is_feasible=Quadratic.is_feasible, 
        base_changes=K, verbose=True
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - Quadratic.solution()[1]
    suboptimality_logs.append(suboptimality_log)

    if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
        suboptimality_log_worst = suboptimality_log
    elif suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
        suboptimality_log_best = suboptimality_log

# mean_time = np.mean(time_logs, axis=0)
mean = np.mean(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Solar method")
plt.fill_between(range(N+1), suboptimality_log_best, suboptimality_log_worst, alpha=0.4)


# ZO gradient descent runs

gradient_descent = ZOGradientDescent(step_size=1/(Quadratic.n * (Quadratic.mu + Quadratic.L)))

time_logs = []
suboptimality_logs = []

for _ in range(runs):
    _, time_log, function_log = gradient_descent.optimise(
        Quadratic.f, None, Quadratic.initial_point(), N
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - Quadratic.solution()[1]
    suboptimality_logs.append(suboptimality_log)

# mean_time = np.mean(time_logs, axis=0)
mean = np.mean(suboptimality_logs, axis=0)
std = np.std(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Gradient descent")
plt.fill_between(range(N+1), mean-std, mean+std, alpha=0.4)


# ZO steepest gradient descent runs

gradient_descent = ZOGradientDescent()

time_logs = []
suboptimality_logs = []

for _ in range(runs):
    _, time_log, function_log = gradient_descent.optimise(
        Quadratic.f, None, Quadratic.initial_point(), N
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - Quadratic.solution()[1]
    suboptimality_logs.append(suboptimality_log)

mean = np.mean(suboptimality_logs, axis=0)
std = np.std(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Steepest gradient descent")
plt.fill_between(range(N+1), mean-std, mean+std, alpha=0.4)


# Momentum three point runs

mtp = MomentumThreePoint(Quadratic.L, 1 - 1e-2, 1e-3)

time_logs = []
suboptimality_logs = []

for _ in range(runs):
    _, time_log, function_log = mtp.optimise(
        Quadratic.f, None, Quadratic.initial_point(), 
        N, Quadratic.is_feasible
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - Quadratic.solution()[1]
    suboptimality_logs.append(suboptimality_log)

mean = np.mean(suboptimality_logs, axis=0)
std = np.std(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Momentum three point")
plt.fill_between(range(N+1), mean-std, mean+std, alpha=0.4)


# Plotting

plt.title(f"Quadratic problem, $\\mu/L = {Quadratic.condition_number:.1e}$")

plt.xlabel("$N$, iteration number")
plt.ylabel("$f(x^N) - f^*$ (log. scale)")
plt.yscale("log")

plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("figures/solar_quadratic_comparison.pdf")
plt.show()
