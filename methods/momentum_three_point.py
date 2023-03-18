import time
import numpy as np
from scipy.optimize import minimize_scalar
from random_generators import generate_spherically_symmetric
from interface_optimisation_method import OptimisationMethod


class MomentumThreePoint(OptimisationMethod):
    """
    Eduard Gorbunov et al.
    A Stochastic Derivative-free Optimization Method with Momentum
    https://openreview.net/pdf?id=HylAoJSKvH
    """
    
    def __init__(self, L, beta, gamma=None):
        assert L > 0
        assert beta > 0 and beta < 1
        assert gamma is None or gamma > 0

        self.L = L
        self.beta = beta
        self.gamma = gamma
    
    def optimise(self, f, _, x, iterations, is_feasible=lambda _: True):
        n = x.shape[0]

        time_log = [0.0]
        function_log = [f(x)]
        time_start = time.time()

        v = np.zeros_like(x)
        z = np.copy(x)

        for _ in range(iterations):
            s = generate_spherically_symmetric(n)
            t = 1e-2
            if self.gamma is None:
                gamma = (1 - self.beta) * abs(f(z + t * s) - f(z)) / (t * self.L)
            else:
                gamma = self.gamma

            v_plus = self.beta * v + s
            v_minus = self.beta * v - s
            
            x_plus = x - gamma * v_plus
            x_minus = x - gamma * v_minus

            z_plus = x_plus - gamma*self.beta/(1-self.beta) * v_plus
            z_minus = x_minus - gamma*self.beta/(1-self.beta) * v_minus

            f_z = f(z) + (0 if is_feasible(z) else float("+inf"))
            f_z_plus = f(z_plus) + (0 if is_feasible(z_plus) else float("+inf"))
            f_z_minus = f(z_minus) + (0 if is_feasible(z_minus) else float("+inf"))

            if f_z_plus <= f_z and f_z_plus <= f_z_minus:
                z = z_plus
                x = x_plus
                v = v_plus
            elif f_z_minus <= f_z and f_z_minus <= f_z_plus:
                z = z_minus
                x = x_minus
                v = v_minus

            time_log.append(time.time() - time_start)
            function_log.append(f(z))

        return z, time_log, function_log
