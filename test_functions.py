from utils import *
import numpy as np


np.random.seed(42)

class Quadratic:
    n = 10
    A = (lambda B: np.dot(B, B.transpose()))(np.random.rand(n, n))
    b = np.random.random(n)
    c = np.random.random()
    mu, L = (lambda eigvals: (eigvals[0], eigvals[-1]))(np.sort(np.linalg.eig(A)[0]))
    condition_number = mu / L

    @classmethod
    def f(cls, x):
        assert x.shape == (cls.n,)
        return np.dot(np.dot(cls.A, x), x) + np.dot(cls.b, x) + cls.c

    def is_feasible(x):
        return np.all((x >= -20) & (x <= 20))
    
    @classmethod
    def initial_point(cls, _=None):
        return np.zeros(cls.n)
    
    x_solution = -np.dot(np.linalg.inv(A + A.transpose()), b)

    @classmethod
    def solution(cls, _=None):
        return cls.x_solution, cls.f(cls.x_solution)


class Quadratic50:
    n = 50
    A = (lambda B: np.dot(B, B.transpose()))(np.random.rand(n, n)) * n
    b = np.random.random(n)
    c = np.random.random()
    mu, L = (lambda eigvals: (eigvals[0], eigvals[-1]))(np.sort(np.linalg.eig(A)[0]))
    condition_number = mu / L

    @classmethod
    def f(cls, x):
        assert x.shape == (cls.n,)
        return np.dot(np.dot(cls.A, x), x) + np.dot(cls.b, x) + cls.c

    def is_feasible(x):
        return np.all((x >= -5) & (x <= 5))
    
    @classmethod
    def initial_point(cls, _=None):
        return np.zeros(cls.n)
    
    x_solution = -np.dot(np.linalg.inv(A + A.transpose()), b)

    @classmethod
    def solution(cls, _=None):
        return cls.x_solution, cls.f(cls.x_solution)


class Quadratic25:
    n = 25
    A = (lambda B: np.dot(B, B.transpose()))(np.random.rand(n, n)) * 10
    b = np.random.random(n)
    c = np.random.random()
    mu, L = (lambda eigvals: (eigvals[0], eigvals[-1]))(np.sort(np.linalg.eig(A)[0]))
    condition_number = mu / L

    @classmethod
    def f(cls, x):
        assert x.shape == (cls.n,)
        return np.dot(np.dot(cls.A, x), x) + np.dot(cls.b, x) + cls.c

    def is_feasible(x):
        return np.all((x >= -5) & (x <= 5))
    
    @classmethod
    def initial_point(cls, _=None):
        return np.zeros(cls.n)
    
    x_solution = -np.dot(np.linalg.inv(A + A.transpose()), b)

    @classmethod
    def solution(cls, _=None):
        return cls.x_solution, cls.f(cls.x_solution)

class RosenbrockSkokov:
    @staticmethod
    def f(x):
        n = x.shape[0]
        return (1 - x[0])**2 + 100 * sum((x[i] - x[i-1]**2)**2 for i in range(1, n))
    
    @staticmethod
    def df(x):
        n = x.shape[0]
        g = np.zeros(n)
        g[0] = -2 * (1 - x[0])
        g[1:n-1] = 200 * (x[1:n-1] - x[:n-2]**2) - 400 * x[1:n-1] * (x[2:] - x[1:n-1]**2)
        g[n-1] = 200 * (x[n-1] - x[n-2]**2)
        return g
    
    def is_feasible(x):
        return np.all((x >= -3) & (x <= 3))

    @staticmethod
    def initial_point(n):
        return (np.ones(n) % 2 - 0.5) / 5
    
    @staticmethod
    def solution(n):
        x = np.ones(n)
        return x, 0.0


class Rastrigin:
    @staticmethod
    def f(x):
        n = x.shape[0]
        return 10 * n * np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    
    @staticmethod
    def df(x):
        n = x.shape[0]
        return 10 * n * (2 * x + 20 * np.pi * np.sin(2 * np.pi * x))
    
    @classmethod
    def df_stoch(cls, x):
        n = x.shape[0]
        g = cls.df(x)
        indices = np.random.choice(np.arange(n), replace=False, size=int(n * 0.5))
        g[indices] = 0
        return g
    
    def is_feasible(x):
        return np.all((x >= -5.12) & (x <= 5.12))

    @staticmethod
    def initial_point(n):
        return np.ones(n) * 5
    
    @staticmethod
    def solution(n):
        x = np.zeros(n)
        return x, 0.0

class DeVilliersGlasser02:
    n = 5

    @staticmethod
    def f(x):
        t = lambda i: 0.1 * (i - 1)
        y = lambda i: 53.81 * (1.27**t(i)) * np.tanh(3.012*t(i) + np.sin(2.13*t(i))) * np.cos(np.exp(0.507)*t(i))
        return sum((x[0] * x[1]**t(i) * np.tanh(x[2]*t(i) + np.sin(x[3]*t(i))) * np.cos(t(i)*np.exp(x[4])) - y(i))**2 for i in range(1, 25))
    
    def is_feasible(x):
        return np.all((x >= 1) & (x <= 60))

    @staticmethod
    def initial_point():
        return np.ones(5) * 30
    
    @staticmethod
    def solution():
        return np.array([53.81, 1.27, 3.012, 2.13, 0.507]), 0.0