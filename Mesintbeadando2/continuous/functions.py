import math
import numpy as np


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin függvény n dimenzióban.
    A = 10, domain: [-5.12, 5.12]^n
    Globális minimum: f(0,...,0) = 0
    """
    A = 10
    n = x.shape[0]
    return A * n + np.sum(x**2 - A * np.cos(2 * math.pi * x))


def booth(x: np.ndarray) -> float:
    """
    Booth függvény 2D-ben.
    Domain: [-10, 10]^2
    Globális minimum: f(1, 3) = 0
    """
    assert x.shape[0] == 2, "Booth 2D-re értelmezett (x, y)."
    x1, y1 = x[0], x[1]
    return (x1 + 2*y1 - 7)**2 + (2*x1 + y1 - 5)**2


def levi(x: np.ndarray) -> float:
    """
    Lévi függvény 2D-ben.
    Domain: [-10, 10]^2
    Globális minimum: f(1, 1) = 0
    """
    assert x.shape[0] == 2, "Lévi 2D-re értelmezett (x, y)."
    x1, y1 = x[0], x[1]

    term1 = (np.sin(3 * math.pi * x1))**2
    term2 = (x1 - 1)**2 * (1 + (np.sin(3 * math.pi * y1))**2)
    term3 = (y1 - 1)**2 * (1 + (np.sin(2 * math.pi * y1))**2)
    return term1 + term2 + term3
