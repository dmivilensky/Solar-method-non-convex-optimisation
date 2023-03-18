import time
import numpy as np
from scipy.optimize import minimize_scalar
from random_generators import generate_spherically_symmetric
from interface_optimisation_method import OptimisationMethod


class ConjugateGradients(OptimisationMethod):
    def __init__(self, method, restart_period=None):
        assert method in ["PRP", "FR"]
        self.method = method
        self.restart_period = restart_period

    def beta(self, g, g_prev):
        if self.method == "PRP":
            return np.dot(g, g - g_prev) / np.dot(g_prev, g_prev)
        elif self.method == "FR":
            return np.dot(g, g) / np.dot(g_prev, g_prev)

    def optimise(self, f, df, x, iterations, is_feasible=lambda _: True):
        time_log = [0.0]
        function_log = [f(x)]
        time_start = time.time()

        d = np.zeros_like(x)
        g = df(x)

        for it in range(iterations):
            if self.restart_period is not None and (it + 1) % self.restart_period == 0:
                d = np.zeros_like(x)

            g_prev = np.copy(g); g = df(x)
            d = -g + self.beta(g, g_prev) * d

            alpha = minimize_scalar(
                    lambda alpha: f(x + alpha * d) + (0 if is_feasible(x + alpha * d) else float("+inf")),
                    method="brent").x

            x += alpha * d
            function_log.append(f(x))

            time_log.append(time.time() - time_start)

        return x, time_log, function_log
