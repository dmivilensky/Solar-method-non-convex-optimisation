import time
from treap import treap
import warnings
import numpy as np
from random_generators import generate_matrix
from interface_optimisation_method import OptimisationMethod


class SolarMethod(OptimisationMethod):
    ray_method_params = {
        "linear": {"pull_size": 1, "use_fo": False},
        "cone": {"pull_size": 1, "use_fo": True},
        "linear_secant": {"pull_size": 2, "use_fo": False}
    }

    def __init__(self, subsolver, base_n=1, ray_method="linear", subsolver_iterations=10, **ray_kwargs):
        assert ray_method in SolarMethod.ray_method_params

        self.ray_method = ray_method
        self.pull_size = SolarMethod.ray_method_params[ray_method]["pull_size"]
        self.use_fo_information = SolarMethod.ray_method_params[ray_method]["use_fo"]
        self.base_n = base_n
        self.subsolver = subsolver
        self.subsolver_iterations = subsolver_iterations
        self.ray_kwargs = ray_kwargs

    def create_ray(self, xs, gs, base, cone_angle=None, increase_angle=False, trivial=False):
        n = xs[0].shape[0]

        if self.ray_method == "linear" or trivial:
            x = xs[0]
            # random a in [-1; 1]:
            coefficients = generate_matrix(n, self.base_n) * 2 - 1
            # random tan(alpha) where alpha in [-pi/2, +pi/2]:
            coefficients = np.tan(np.pi/2 * coefficients)
            # alpha between base and the same is pi/4, alpha between base and different is 0:
            coefficients[base] = np.eye(n)[base, base]
            
            return lambda base_x: x + np.dot(coefficients, base_x - x[base])
        
        elif self.ray_method == "cone":
            assert cone_angle is not None
            assert cone_angle > 0 and cone_angle < np.pi/2
            
            x = xs[0]; g = gs[0]; g_norm = np.linalg.norm(g)
            # random a in [-1; 1]:
            coefficients = generate_matrix(n, self.base_n)
            # random tan(atan((df/dxi)/(df/dx_b)) + alpha) where alpha in [-cone_angle, +cone_angle]: 
            gradient_angles = np.arctan(g[:, None] / g[None, base])
            multiplicator = min(max(1, np.sqrt(n)/g_norm), np.pi/2/cone_angle) if increase_angle else 1
            maximum_angles = np.clip(gradient_angles + multiplicator * cone_angle, None, np.pi/2 - 1e-16)
            minimum_angles = np.clip(gradient_angles - multiplicator * cone_angle, -np.pi/2 + 1e-16, None)
            average_direction = (maximum_angles + minimum_angles) / 2
            average_angle = (maximum_angles - minimum_angles) / 2
            coefficients = np.tan(average_direction + average_angle * coefficients)
            # coefficients = np.tan(np.clip(gradient_angles + multiplicator * cone_angle * coefficients, -np.pi/2 + 1e-16, np.pi/2 - 1e-16))
            # alpha between base and the same is pi/4, alpha between base and different is 0:
            coefficients[base] = np.eye(n)[base, base]

            return lambda base_x: x + np.dot(coefficients, base_x - x[base])

        elif self.ray_method == "linear_secant":
            assert cone_angle is not None
            assert cone_angle > 0 and cone_angle < np.pi/2
            
            x1 = xs[0]; x2 = xs[1]; g = x2 - x1
            # random a in [-1; 1]:
            coefficients = generate_matrix(n, self.base_n)
            # random tan(atan((df/dxi)/(df/dx_b)) + alpha) where alpha in [-cone_angle, +cone_angle]: 
            gradient_angles = np.arctan(g[:, None] / g[None, base])
            maximum_angles = np.clip(gradient_angles + cone_angle, None, np.pi/2 - 1e-16)
            minimum_angles = np.clip(gradient_angles - cone_angle, -np.pi/2 + 1e-16, None)
            average_direction = (maximum_angles + minimum_angles) / 2
            average_angle = (maximum_angles - minimum_angles) / 2
            coefficients = np.tan(average_direction + average_angle * coefficients)
            # coefficients = np.tan(np.clip(gradient_angles + multiplicator * cone_angle * coefficients, -np.pi/2 + 1e-16, np.pi/2 - 1e-16))
            # alpha between base and the same is pi/4, alpha between base and different is 0:
            coefficients[base] = np.eye(n)[base, base]

            return lambda base_x: x1 + np.dot(coefficients, base_x - x1[base])

    def optimise(self, f, df, x, iterations, is_feasible=lambda _: True, base_changes=1, verbose=False):
        n = x.shape[0]
        assert self.base_n < n

        pull_of_points = treap()
        pull_of_points[f(x)] = x

        time_log = [0.0]
        time_start = time.time()
        function_log = [f(x)]

        for i in range(base_changes):
            base = np.random.choice(n, self.base_n, replace=False)
            
            for _ in range(iterations // base_changes):
                f_reference = pull_of_points.find_min()
                x_reference = pull_of_points[f_reference]
                del pull_of_points[f_reference]
                extracted_points = treap()
                extracted_points[f_reference] = x_reference
                
                xs = [x_reference]
                while len(xs) < self.pull_size and len(pull_of_points) > 0:
                    new_f_reference = pull_of_points.find_min()
                    extracted_points[new_f_reference] = pull_of_points[new_f_reference]
                    del pull_of_points[new_f_reference]
                    xs.append(extracted_points[new_f_reference])
                
                if len(xs) < self.pull_size:
                    ray = self.create_ray([xs[0]], [], base=base, trivial=True, **self.ray_kwargs)
                else:
                    gs = [-df(x) for x in xs] if self.use_fo_information else []
                    ray = self.create_ray(xs, gs, base=base, **self.ray_kwargs)

                base_x, _, fs = self.subsolver.optimise(
                    lambda base_x: f(ray(base_x)) + (0 if is_feasible(ray(base_x)) else float("+inf")), None,
                    x_reference[base], self.subsolver_iterations)

                x_candidate = ray(base_x); f_candidate = fs[0]
                if verbose:
                    if f_candidate >= extracted_points.find_max():
                        warnings.warn("New f is terrible")
                    elif f_candidate >= extracted_points.find_min():
                        warnings.warn("New f is bad")
                extracted_points[f_candidate] = x_candidate

                while len(pull_of_points) < self.pull_size and len(extracted_points) > 0:
                    new_f_to_pull = extracted_points.find_min()
                    if new_f_to_pull in pull_of_points:
                        break
                    pull_of_points[new_f_to_pull] = extracted_points[new_f_to_pull]
                    del extracted_points[new_f_to_pull]

                time_log.append(time.time() - time_start)
                function_log.append(pull_of_points.find_min())

        return x, time_log, function_log 
