# tsp/tsp_instances.py

from typing import Dict, Tuple, List
import math

CityIndex = int
DistancesDict = Dict[Tuple[CityIndex, CityIndex], float]


def build_distance_dict(coords: List[Tuple[float, float]]) -> DistancesDict:
    """
    Koordinátákból szimmetrikus távolságmátrixot épít.
    distances[(i, j)] = euklideszi távolság i és j város között.
    """
    n = len(coords)
    d: DistancesDict = {}
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            xj, yj = coords[j]
            if i == j:
                dist = 0.0
            else:
                dist = math.hypot(xi - xj, yi - yj)
            d[(i, j)] = dist
    return d


# --- 10 városos kis példa ---------------------------------------------------

COORDS_10 = [
    (0.0, 0.0),
    (1.0, 5.0),
    (5.0, 2.0),
    (6.0, 6.0),
    (8.0, 3.0),
    (2.0, 1.0),
    (3.0, 7.0),
    (7.0, 8.0),
    (9.0, 5.0),
    (4.0, 4.0),
]

SMALL_10_CITY: DistancesDict = build_distance_dict(COORDS_10)


# --- 20 városos közepes példa ----------------------------------------------

COORDS_20 = [
    (0.0, 0.0),
    (2.0, 3.0),
    (4.0, 1.0),
    (6.0, 4.0),
    (8.0, 0.0),
    (1.0, 6.0),
    (3.0, 8.0),
    (5.0, 7.0),
    (7.0, 9.0),
    (9.0, 6.0),
    (2.0, -2.0),
    (4.0, -3.0),
    (6.0, -2.5),
    (8.0, -4.0),
    (1.5, 2.0),
    (3.5, 5.5),
    (5.5, 2.5),
    (7.5, 5.0),
    (9.5, 2.0),
    (4.5, 6.5),
]

MEDIUM_20_CITY: DistancesDict = build_distance_dict(COORDS_20)