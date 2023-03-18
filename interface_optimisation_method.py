import numpy as np
from typing import Callable, Tuple, List
from abc import ABCMeta, abstractmethod


class OptimisationMethod(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'optimise') and 
                callable(subclass.optimise) or 
                NotImplemented)

    @abstractmethod
    def optimise(self, f: Callable[[np.ndarray], float], df: Callable[[np.ndarray], np.ndarray], x: np.ndarray, iterations: int, verbose: bool, **kwargs) -> Tuple[np.ndarray, List[float], List[float]]:
        raise NotImplementedError
