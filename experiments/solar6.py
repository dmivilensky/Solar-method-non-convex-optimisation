import numpy as np
from methods import SolarMethod, Adam, SimulatedAnnealing, MonotonicSequenceBasinHopping, ConjugateGradients
from subsolvers import NelderMead
from test_functions import Rastrigin
import matplotlib.pyplot as plt


N = 2000
K = N // 50
n = 200
B = 5
runs = 5

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 16})

# Solar method runs

subsolver = NelderMead()
method = SolarMethod(subsolver, base_n=B, ray_method="linear", subsolver_iterations=10)

time_logs = []
suboptimality_logs = []
suboptimality_log_best = None
suboptimality_log_worst = None

for _ in range(runs):
    _, time_log, function_log = method.optimise(
        Rastrigin.f, Rastrigin.df, Rastrigin.initial_point(n), 
        N, is_feasible=Rastrigin.is_feasible, 
        base_changes=K, verbose=True
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - Rastrigin.solution(n)[1]
    suboptimality_logs.append(suboptimality_log)

    if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
        suboptimality_log_worst = suboptimality_log
    if suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
        suboptimality_log_best = suboptimality_log

mean = np.mean(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Solar method, B={B}")
plt.fill_between(range(N+1), suboptimality_log_best, suboptimality_log_worst, alpha=0.2)


# Solar method runs

subsolver = NelderMead()
method = SolarMethod(subsolver, base_n=5*B, ray_method="linear", subsolver_iterations=10)

time_logs = []
suboptimality_logs = []
suboptimality_log_best = None
suboptimality_log_worst = None

for _ in range(runs):
    _, time_log, function_log = method.optimise(
        Rastrigin.f, Rastrigin.df, Rastrigin.initial_point(n), 
        N, is_feasible=Rastrigin.is_feasible, 
        base_changes=K, verbose=True
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - Rastrigin.solution(n)[1]
    suboptimality_logs.append(suboptimality_log)

    if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
        suboptimality_log_worst = suboptimality_log
    if suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
        suboptimality_log_best = suboptimality_log

mean = np.mean(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Solar method, B={5*B}")
plt.fill_between(range(N+1), suboptimality_log_best, suboptimality_log_worst, alpha=0.2)


# Simulated Annealing runs

T = 1
sa = SimulatedAnnealing(
    None, [[-5.12 for _ in range(n)], [5.12 for _ in range(n)]],
    0.1, temperature_function=lambda it: 0.99**it * T
)

time_logs = []
suboptimality_logs = []
suboptimality_log_best = None
suboptimality_log_worst = None

for _ in range(runs):
    _, time_log, function_log = sa.optimise(
        Rastrigin.f, Rastrigin.df, Rastrigin.initial_point(n), N
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - Rastrigin.solution(n)[1]
    suboptimality_logs.append(suboptimality_log)

    if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
        suboptimality_log_worst = suboptimality_log
    if suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
        suboptimality_log_best = suboptimality_log

mean = np.mean(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Simulated Annealing")
plt.fill_between(range(N+1), suboptimality_log_best, suboptimality_log_worst, alpha=0.2)

# Simulated Annealing runs

T = 1
msbh = MonotonicSequenceBasinHopping(
    None, [[-5.12 for _ in range(n)], [5.12 for _ in range(n)]], 0.1
)

time_logs = []
suboptimality_logs = []
suboptimality_log_best = None
suboptimality_log_worst = None

for _ in range(runs):
    _, time_log, function_log = msbh.optimise(
        Rastrigin.f, Rastrigin.df, Rastrigin.initial_point(n), N
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - Rastrigin.solution(n)[1]
    suboptimality_logs.append(suboptimality_log)

    if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
        suboptimality_log_worst = suboptimality_log
    if suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
        suboptimality_log_best = suboptimality_log

mean = np.mean(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Monotonic Sequence Basin Hopping")
plt.fill_between(range(N+1), suboptimality_log_best, suboptimality_log_worst, alpha=0.2)

# Plotting

plt.title(f"Rastrigin problem, $n={n}$")

plt.xlabel("$N$, iteration number")
plt.ylabel("$f(x^N) - f^*$ (log. scale)")
plt.yscale("log")

plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("figures/solar_rastrigin200_comparison.pdf")
plt.show()
