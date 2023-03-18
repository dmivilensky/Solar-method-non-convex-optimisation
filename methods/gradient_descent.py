import time
import numpy as np
from scipy.optimize import minimize_scalar
from interface_optimisation_method import OptimisationMethod


class GradientDescent(OptimisationMethod):
    def __init__(self, step_size=None):
        assert step_size is None or step_size > 0
        self.step_size = step_size
    
    def optimise(self, f, df, x, iterations, is_feasible=lambda _: True):
        time_log = [0.0]
        function_log = [f(x)]
        time_start = time.time()

        for _ in range(iterations):
            g = df(x)

            if self.step_size is not None:
                x -= self.step_size * g
            else:
                step_size = minimize_scalar(
                    lambda step_size: f(x - step_size * g) + (0 if is_feasible(x - step_size * g) else float("+inf")),
                    method="brent").x
                x -= step_size * g

            time_log.append(time.time() - time_start)
            function_log.append(f(x))

        return x, time_log, function_log
