# continuous/rastrigin_nd_crossovers.py

import time
import pandas as pd

from continuous.functions import rastrigin
from continuous.ga_core import GeneticAlgorithm


def run_rastrigin_nd_crossovers() -> pd.DataFrame:
    """
    Rastrigin függvény több dimenzióban, több különböző crossover módszerrel.
    A feladatkiírás szerinti dimenziók: 3, 4, 5, 10, 100
    """

    dims = [2, 3, 4, 5, 10, 100]

    crossover_methods = [
        "one_point",
        "two_point",
        "k_point",
        "uniform",
        "path_relink",
    ]

    # közös GA-paraméterek, nagyon hasonlóak a rastrigin_nd_experiments.py-hez
    common = dict(
        cost_function=rastrigin,
        bounds=(-5.12, 5.12),
        population_size=100,
        generations=200,
        crossover_rate=0.9,
        mutation_rate=0.3,
        step_size=0.1,          # finom mutáció magasabb dimenziókhoz is
        elitism=True,
        selection_method="tournament",
        tournament_k=2,
        random_seed=42,         # reprodukálhatóság
    )

    rows = []

    print("Rastrigin több dimenzióban, többféle crossoverrel:")
    for d in dims:
        for cm in crossover_methods:
            t0 = time.perf_counter()
            ga = GeneticAlgorithm(
                dim=d,
                crossover_method=cm,
                crossover_points=3,      # k_point-hoz
                uniform_swap_prob=0.5,   # uniformhoz
                **common,
            )
            x_best, f_best = ga.run()
            elapsed = time.perf_counter() - t0

            rows.append(
                {
                    "dim": d,
                    "crossover_method": cm,
                    "best_cost": float(f_best),
                    "time_sec": elapsed,
                }
            )

            print(
                f"d={d:3d}, crossover={cm:11s} -> "
                f"best f(x)={f_best:.6e}, time={elapsed:.3f} s"
            )

    df = pd.DataFrame(rows)
    df.to_csv("rastrigin_nd_crossovers.csv", index=False)
    print("\nMentve: rastrigin_nd_crossovers.csv")
    print("Első pár sor:")
    print(df.head())

    return df


if __name__ == "__main__":
    run_rastrigin_nd_crossovers()
