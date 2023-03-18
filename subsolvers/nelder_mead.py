import time
import warnings
from scipy.optimize import minimize
from interface_optimisation_method import OptimisationMethod


class NelderMead(OptimisationMethod):
    def __init__(self):
        pass

    def optimise(self, f, _, x, iterations, verbose=False):
        time_start = time.time()
        result = minimize(f, x, method="Nelder-Mead", options={"maxiter": iterations})
        if verbose and not result.success:
            warnings.warn(result.message)
        return result.x, [time.time() - time_start], [f(result.x)]
