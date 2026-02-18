# continuous/experiments.py

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict, Any

import numpy as np
import pandas as pd

from continuous.functions import rastrigin, booth, levi
from continuous.ga_core import GeneticAlgorithm


@dataclass
class ExperimentConfig:
    func_name: str
    dim: int
    bounds: tuple[float, float]
    generations: int
    population_size: int
    step_size: float
    mutation_rate: float
    crossover_rate: float
    selection_method: str
    crossover_method: str
    elitism: bool


@dataclass
class ExperimentResult:
    # beállítások
    func_name: str
    dim: int
    generations: int
    population_size: int
    step_size: float
    mutation_rate: float
    crossover_rate: float
    selection_method: str
    crossover_method: str
    elitism: bool

    # eredmények
    best_cost: float
    best_x: str     # stringként tároljuk, hogy szépen menjen CSV-be
    time_sec: float


def get_function_by_name(name: str) -> Callable[[np.ndarray], float]:
    name = name.lower()
    if name == "rastrigin":
        return rastrigin
    if name == "booth":
        return booth
    if name == "levi":
        return levi
    raise ValueError(f"Ismeretlen függvénynév: {name}")


def run_single_experiment(cfg: ExperimentConfig, random_seed: int | None = None) -> ExperimentResult:
    """Egy GA-futtatás egy konkrét paraméterkészlettel."""

    cost_fn = get_function_by_name(cfg.func_name)

    start_time = time.perf_counter()

    ga = GeneticAlgorithm(
        cost_function=cost_fn,
        dim=cfg.dim,
        bounds=cfg.bounds,
        population_size=cfg.population_size,
        generations=cfg.generations,
        crossover_rate=cfg.crossover_rate,
        mutation_rate=cfg.mutation_rate,
        step_size=cfg.step_size,
        elitism=cfg.elitism,
        selection_method=cfg.selection_method,
        crossover_method=cfg.crossover_method,
        tournament_k=3,
        random_seed=random_seed,
    )

    best_x, best_f = ga.run()
    elapsed = time.perf_counter() - start_time

    return ExperimentResult(
        func_name=cfg.func_name,
        dim=cfg.dim,
        generations=cfg.generations,
        population_size=cfg.population_size,
        step_size=cfg.step_size,
        mutation_rate=cfg.mutation_rate,
        crossover_rate=cfg.crossover_rate,
        selection_method=cfg.selection_method,
        crossover_method=cfg.crossover_method,
        elitism=cfg.elitism,
        best_cost=float(best_f),
        best_x=np.array2string(best_x, precision=5),
        time_sec=elapsed,
    )


def run_param_sweep() -> pd.DataFrame:
    """
    Paraméter-söprés folytonos függvényekre.
    Itt állítod be, milyen kombinációkat akarsz kipróbálni.
    """

    # --- MIT VIZSGÁLUNK? -------------------------------------------------
    function_settings = [
        ("rastrigin", 2, (-5.12, 5.12)),
        ("booth", 2, (-10, 10)),
        ("levi", 2, (-10, 10)),
    ]

    # Generációk száma (G)
    generations_list = [5, 10, 20, 50, 100]

    # Populációméret (K)
    population_sizes = [5, 10, 20, 50, 100]

    # Lépésméret / mutáció skálája (L)
    step_sizes = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    # Szelekciós módszerek
    selection_methods = ["tournament", "roulette", "rank", "fitness_rank", "diversity"]

    # Crossover típusok
    crossover_methods = ["one_point", "two_point", "k_point", "uniform", "path_relink"]

    # Elitizmus vizsgálata
    elitism_options = [True, False]

    # Rögzített valószínűségek (nyugodtan állíthatod)
    mutation_rate = 0.3
    crossover_rate = 0.9

    results: List[ExperimentResult] = []

    # --- BRUTÁL NESTED FOR: minden kombó végigzongorázása ----------------
    for func_name, dim, bounds in function_settings:
        for G in generations_list:
            for K in population_sizes:
                for step in step_sizes:
                    for sel in selection_methods:
                        for cross in crossover_methods:
                            for elit in elitism_options:
                                cfg = ExperimentConfig(
                                    func_name=func_name,
                                    dim=dim,
                                    bounds=bounds,
                                    generations=G,
                                    population_size=K,
                                    step_size=step,
                                    mutation_rate=mutation_rate,
                                    crossover_rate=crossover_rate,
                                    selection_method=sel,
                                    crossover_method=cross,
                                    elitism=elit,
                                )

                                # Itt adhatsz neki seedet, ha összehasonlítható futásokat akarsz
                                res = run_single_experiment(cfg, random_seed=42)
                                results.append(res)

                                print(
                                    f"Kész: f={func_name}, G={G}, K={K}, "
                                    f"L={step}, sel={sel}, cross={cross}, elit={elit}, "
                                    f"best={res.best_cost:.5f}, t={res.time_sec:.4f}s"
                                )

    # --- EREDMÉNYEK TÁBLÁBA -----------------------------------------------
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv("continuous_param_sweep.csv", index=False)
    print("\nMentve: continuous_param_sweep.csv")

    # gyors ellenőrzésre dobjunk egy pár sort
    print("\nElső pár sor:")
    print(df.head())

    return df


if __name__ == "__main__":
    # Ha ezt a fájlt futtatod: python -m continuous.experiments
    run_param_sweep()

