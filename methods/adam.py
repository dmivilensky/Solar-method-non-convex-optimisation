import time
import numpy as np
from interface_optimisation_method import OptimisationMethod


class Adam(OptimisationMethod):
    def __init__(self, step_size=0.001, beta1=0.9, beta2=0.999):
        assert step_size > 0
        assert beta1 > 0 and beta2 > 0

        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
    
    def optimise(self, f, df_stoch, x, iterations):
        time_log = [0.0]
        function_log = [f(x)]
        time_start = time.time()

        m = np.zeros_like(x)
        v = 0

        x_best = np.copy(x)
        f_best = f(x)

        for it in range(iterations):
            g = df_stoch(x)

            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * np.linalg.norm(g)**2

            x -= self.step_size * m / (1 - self.beta1**(it + 1)) / (np.sqrt(v / (1 - self.beta2**(it + 1))) + 1e-8)
            f_new = f(x)

            if f_new < f_best:
                x_best = np.copy(x)
                f_best = f_new

            time_log.append(time.time() - time_start)
            function_log.append(f_best)

        return x_best, time_log, function_log
