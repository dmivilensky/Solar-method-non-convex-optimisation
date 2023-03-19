import time
import numpy as np
from random_generators import generate_vector, generate_spherically_symmetric
from interface_optimisation_method import OptimisationMethod


class SimulatedAnnealing(OptimisationMethod):
    """
    Guo-Liang Xue
    Parallel Two-level Simulated Annealing
    https://dl.acm.org/doi/pdf/10.1145/165939.166011
    """
    
    def __init__(self, subsolver, bounds, jump_length, subsolver_iterations=10, temperature_function=lambda it: 0.99**it * 1):
        assert jump_length > 0
        assert len(bounds) == 2
        assert len(bounds[0]) == len(bounds[1])
        assert temperature_function(0) > 0 and temperature_function(10000) > 0

        self.bounds = np.array(bounds)
        self.jump_length = jump_length
        self.temperature_function = temperature_function

        self.subsolver = subsolver
        self.subsolver_iterations = subsolver_iterations

    def optimise(self, f, df, x, iterations):
        n = self.bounds.shape[1]

        time_log = [0.0]
        function_log = [f(x)]
        time_start = time.time()

        x_old = self.bounds[0] + generate_vector(n) * (self.bounds[1] - self.bounds[0])
        f_old = f(x)
        x_best = np.copy(x)
        f_best = f_old

        for it in range(iterations):
            s = self.jump_length * generate_spherically_symmetric(n)

            x_new = np.clip(x_old + s, self.bounds[0], self.bounds[1])
            if self.subsolver is not None:
                x_new, _, f_new_ = self.subsolver.optimise(
                    f, df, x_new, self.subsolver_iterations, 
                    is_feasible=lambda y: np.all((y >= self.bounds[0]) & (y <= self.bounds[1])))
                f_new = f_new_[-1]
            else:
                f_new = f(x_new)

            r = np.random.random()

            if (f_new <= f_old) or (r <= np.exp((f_old - f_new) / self.temperature_function(it))):
                x_old = np.copy(x_new)
                f_old = f_new
                if f_new < f_best:
                    x_best = np.copy(x_new)
                    f_best = f_new

            time_log.append(time.time() - time_start)
            function_log.append(f_best)

        return x_best, time_log, function_log
