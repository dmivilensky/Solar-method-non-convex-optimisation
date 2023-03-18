import numpy as np
from methods import SolarMethod, ZOGradientDescent, GradientDescent, ConjugateGradients
from subsolvers import NelderMead
from test_functions import RosenbrockSkokov
import matplotlib.pyplot as plt


N = 20000
K = N // 50
n = 100
B = 20
runs = 3

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 16})

# Solar method runs

subsolver = NelderMead()
method = SolarMethod(subsolver, base_n=B, subsolver_iterations=10)

time_logs = []
suboptimality_logs = []
suboptimality_log_best = None
suboptimality_log_worst = None

for _ in range(runs):
    _, time_log, function_log = method.optimise(
        RosenbrockSkokov.f, None, RosenbrockSkokov.initial_point(n), 
        N, is_feasible=RosenbrockSkokov.is_feasible, 
        base_changes=K, verbose=True
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - RosenbrockSkokov.solution(n)[1]
    suboptimality_logs.append(suboptimality_log)

    if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
        suboptimality_log_worst = suboptimality_log
    if suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
        suboptimality_log_best = suboptimality_log

mean = np.mean(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Solar method, $K = {K}$, $B = {B}$")
plt.fill_between(range(N+1), suboptimality_log_best, suboptimality_log_worst, alpha=0.2)


# Solar cone method runs

subsolver = NelderMead()
method = SolarMethod(subsolver, base_n=B, ray_method="cone", subsolver_iterations=10, cone_angle=np.pi/20)

time_logs = []
suboptimality_logs = []
suboptimality_log_best = None
suboptimality_log_worst = None

for _ in range(runs):
    _, time_log, function_log = method.optimise(
        RosenbrockSkokov.f, RosenbrockSkokov.df, RosenbrockSkokov.initial_point(n), 
        N, is_feasible=RosenbrockSkokov.is_feasible, 
        base_changes=K, verbose=True
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - RosenbrockSkokov.solution(n)[1]
    suboptimality_logs.append(suboptimality_log)

    if suboptimality_log_worst is None or suboptimality_log[-1] > suboptimality_log_worst[-1]:
        suboptimality_log_worst = suboptimality_log
    if suboptimality_log_best is None or suboptimality_log[-1] < suboptimality_log_best[-1]:
        suboptimality_log_best = suboptimality_log

mean = np.mean(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Solar method (FO)")
plt.fill_between(range(N+1), suboptimality_log_best, suboptimality_log_worst, alpha=0.2)


# ZO steepest gradient descent runs

gradient_descent = ZOGradientDescent()

time_logs = []
suboptimality_logs = []

for _ in range(runs):
    _, time_log, function_log = gradient_descent.optimise(
        RosenbrockSkokov.f, None, RosenbrockSkokov.initial_point(n), N
    )

    time_logs.append(time_log)
    suboptimality_log = np.array(function_log) - RosenbrockSkokov.solution(n)[1]
    suboptimality_logs.append(suboptimality_log)

mean = np.mean(suboptimality_logs, axis=0)
std = np.std(suboptimality_logs, axis=0)
plt.plot(range(N+1), mean, label=f"Steepest gradient descent (ZO)")
plt.fill_between(range(N+1), mean-std, mean+std, alpha=0.4)


# Steepest gradient descent runs

gradient_descent = GradientDescent()

time_logs = []

_, time_log, function_log = gradient_descent.optimise(
    RosenbrockSkokov.f, RosenbrockSkokov.df, RosenbrockSkokov.initial_point(n), N
)

suboptimality_log = np.array(function_log) - RosenbrockSkokov.solution(n)[1]
plt.plot(range(N+1), suboptimality_log, label=f"Steepest gradient descent (FO)")


# PRP restarted cojugate gradients runs

cg = ConjugateGradients(method="PRP", restart_period=n)

time_logs = []

_, time_log, function_log = cg.optimise(
    RosenbrockSkokov.f, RosenbrockSkokov.df, 
    RosenbrockSkokov.initial_point(n), N,
    RosenbrockSkokov.is_feasible
)

suboptimality_log = np.array(function_log) - RosenbrockSkokov.solution(n)[1]
plt.plot(range(N+1), suboptimality_log, label=f"Restarted Polak–Ribier–Polyak CG")


# FR restarted cojugate gradients runs

cg = ConjugateGradients(method="FR", restart_period=n)

time_logs = []

_, time_log, function_log = cg.optimise(
    RosenbrockSkokov.f, RosenbrockSkokov.df, 
    RosenbrockSkokov.initial_point(n), N,
    RosenbrockSkokov.is_feasible
)

suboptimality_log = np.array(function_log) - RosenbrockSkokov.solution(n)[1]
plt.plot(range(N+1), suboptimality_log, label=f"Restarted Fletcher–Reeves CG")

# FR cojugate gradients runs

cg = ConjugateGradients(method="FR")

time_logs = []

_, time_log, function_log = cg.optimise(
    RosenbrockSkokov.f, RosenbrockSkokov.df, 
    RosenbrockSkokov.initial_point(n), N,
    RosenbrockSkokov.is_feasible
)

suboptimality_log = np.array(function_log) - RosenbrockSkokov.solution(n)[1]
plt.plot(range(N+1), suboptimality_log, label=f"Fletcher–Reeves CG")


# Plotting

plt.title(f"Rosenbrock–Skokov problem, $n={n}$")

# plt.xlabel("$t$, s")
plt.xlabel("$N$, iteration number")
plt.ylabel("$f(x^N) - f^*$ (log. scale)")
plt.yscale("log")

plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("figures/solar_rosenbrock100_comparison.pdf")
plt.show()
