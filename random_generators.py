import numpy as np


def generate_vector(n):
    return np.random.random(n)


def generate_matrix(n, m):
    return np.random.rand(n, m)


def generate_normal_matrix(n, m):
    return np.random.normal(size=(n, m))


def generate_spherically_symmetric_matrix(n, m):
    x = generate_normal_matrix(n, m)
    return x / np.linalg.norm(x, axis=1)[:, None]


def generate_spherically_symmetric(n):
    x = np.random.normal(size=n)
    return x / np.linalg.norm(x)
