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
    
    def __init__(self, subsolver, bounds, jump_length, subsolver_iterations=10, temperature=None):
        assert jump_length > 0
        assert len(bounds) == 2
        assert len(bounds[0]) == len(bounds[1])
        assert temperature is None or temperature > 0

        self.bounds = np.array(bounds)
        self.jump_length = jump_length
        self.temperature = temperature

        self.subsolver = subsolver
        self.subsolver_iterations = subsolver_iterations

    def optimise(self, f, df, x, iterations):
        n = self.bounds.shape[1]

        time_log = [0.0]
        function_log = [f(x)]
        time_start = time.time()

        x = self.bounds[0] + generate_vector(n) * (self.bounds[1] - self.bounds[0])

        for _ in range(iterations):
            s = self.jump_length * generate_spherically_symmetric(n)
            
            y, _, fy_ = self.subsolver.optimise(
                f, df, x + s, self.subsolver_iterations, 
                is_feasible=lambda y: np.all((y >= self.bounds[0]) & (y <= self.bounds[1])))
            fy = fy_[-1]

            if fy < function_log[-1]:
                x = np.copy(y)

            time_log.append(time.time() - time_start)
            function_log.append(f(x))

        return x, time_log, function_log
