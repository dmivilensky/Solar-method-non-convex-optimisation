import time
import numpy as np
from scipy.optimize import minimize_scalar
from random_generators import generate_spherically_symmetric
from interface_optimisation_method import OptimisationMethod


class ZOGradientDescent(OptimisationMethod):
    def __init__(self, step_size=None, shift=1e-1):
        assert step_size is None or step_size > 0
        assert shift > 0

        self.step_size = step_size
        self.shift = shift
    
    def optimise(self, f, _, x, iterations, is_feasible=lambda _: True):
        n = x.shape[0]

        time_log = [0.0]
        function_log = [f(x)]
        time_start = time.time()

        for _ in range(iterations):
            e = generate_spherically_symmetric(n)
            g = e * n * (f(x + self.shift * e) - f(x - self.shift * e)) / (2 * self.shift)

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
