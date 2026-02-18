# tsp/tsp_experiments.py

import time
from tsp.tsp_ga import TSPGeneticAlgorithm, DistancesDict
from tsp.tsp_instances import SMALL_10_CITY, MEDIUM_20_CITY


def run_single_tsp(name: str, distances: DistancesDict,
                   population_size: int, generations: int,
                   mutation_rate: float, crossover_rate: float,
                   elitism: bool, random_seed: int = 42) -> tuple[float, float]:
    """
    Egy TSP GA-futtatás adott paraméterekkel.
    Visszaadja: (best_cost, time_sec)
    """
    n_cities = len({i for (i, _) in distances.keys()})
    common = dict(
        distances=distances,
        num_cities=n_cities,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elitism=elitism,
        random_seed=random_seed,
    )

    t0 = time.perf_counter()
    ga = TSPGeneticAlgorithm(**common)
    best_route, best_cost = ga.run()
    t = time.perf_counter() - t0

    print(f"[{name}] G={generations}, K={population_size}, "
          f"pm={mutation_rate}, pc={crossover_rate}, elit={elitism}")
    print("   best route:", best_route)
    print("   best cost :", best_cost)
    print("   time [s]  :", t)
    print()
    return best_cost, t


def main():
    print("=== TSP kísérletek – 10 városos instance ===\n")

    # 10 városos példa
    dist_small = SMALL_10_CITY

    # pár paraméterkombó (teljesen szabadon bővíthető)
    configs_small = [
        dict(name="S1", G=50,  K=50,  pm=0.1, pc=0.9, elit=True),
        dict(name="S2", G=100, K=50,  pm=0.3, pc=0.9, elit=True),
        dict(name="S3", G=100, K=100, pm=0.3, pc=0.9, elit=True),
        dict(name="S4", G=200, K=100, pm=0.3, pc=0.9, elit=True),
    ]

    for cfg in configs_small:
        run_single_tsp(
            name=cfg["name"],
            distances=dist_small,
            population_size=cfg["K"],
            generations=cfg["G"],
            mutation_rate=cfg["pm"],
            crossover_rate=cfg["pc"],
            elitism=cfg["elit"],
        )

    print("\n=== TSP kísérletek – 20 városos instance ===\n")

    # 20 városos példa
    dist_med = MEDIUM_20_CITY

    configs_med = [
        dict(name="M1", G=100, K=100, pm=0.3, pc=0.9, elit=True),
        dict(name="M2", G=200, K=100, pm=0.3, pc=0.9, elit=True),
        dict(name="M3", G=200, K=200, pm=0.3, pc=0.9, elit=True),
    ]

    for cfg in configs_med:
        run_single_tsp(
            name=cfg["name"],
            distances=dist_med,
            population_size=cfg["K"],
            generations=cfg["G"],
            mutation_rate=cfg["pm"],
            crossover_rate=cfg["pc"],
            elitism=cfg["elit"],
        )


if __name__ == "__main__":
    main()