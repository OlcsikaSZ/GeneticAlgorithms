# prob_experiments.py

import time
from continuous.plots import run_ga_and_get_history

def study_mutation_and_crossover():
    func_name = "rastrigin"
    dim = 2
    bounds = (-5.12, 5.12)

    base_cfg = dict(
        func_name=func_name,
        dim=dim,
        bounds=bounds,
        generations=200,
        population_size=100,
        step_size=0.1,
        selection_method="tournament",
        crossover_method="path_relink",
        elitism=True,
        random_seed=42,
        mutation_rate=0.3,
        crossover_rate=0.9,
    )

    mutation_rates = [0.1, 0.3, 0.5, 0.7]
    crossover_rates = [0.3, 0.6, 0.9]

    print("=== Mutációs valószínűség vizsgálata (pc = 0.9) ===")
    for pm in mutation_rates:
        cfg = dict(base_cfg)
        cfg["mutation_rate"] = pm
        cfg["crossover_rate"] = 0.9

        t0 = time.perf_counter()
        hist_best, hist_mean, hist_std, hist_worst, x, f = run_ga_and_get_history(**cfg)
        elapsed = time.perf_counter() - t0

        print(f"pm={pm:.2f}, pc=0.90 -> best={f:.3e}, time={elapsed:.3f}s")

    print("\n=== Crossover valószínűség vizsgálata (pm = 0.3) ===")
    for pc in crossover_rates:
        cfg = dict(base_cfg)
        cfg["mutation_rate"] = 0.3
        cfg["crossover_rate"] = pc

        t0 = time.perf_counter()
        hist_best, hist_mean, hist_std, hist_worst, x, f = run_ga_and_get_history(**cfg)
        elapsed = time.perf_counter() - t0

        print(f"pm=0.30, pc={pc:.2f} -> best={f:.3e}, time={elapsed:.3f}s")


if __name__ == "__main__":
    study_mutation_and_crossover()
