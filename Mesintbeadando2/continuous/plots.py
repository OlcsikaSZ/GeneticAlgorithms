# continuous/plots.py

import matplotlib.pyplot as plt
import numpy as np

from continuous.functions import rastrigin, booth, levi
from continuous.ga_core import GeneticAlgorithm


def get_function_by_name(name: str):
    name = name.lower()
    if name == "rastrigin":
        return rastrigin
    if name == "booth":
        return booth
    if name == "levi":
        return levi
    raise ValueError(f"Ismeretlen függvény: {name}")


def run_ga_and_get_history(
    func_name: str,
    dim: int,
    bounds: tuple[float, float],
    generations: int,
    population_size: int,
    step_size: float,
    mutation_rate: float,
    crossover_rate: float,
    selection_method: str,
    crossover_method: str,
    elitism: bool,
    random_seed: int | None = 42,
):
    """
    Lefuttat egy GA-t a megadott paraméterekkel,
    és visszaadja a history_best / mean / std listákat.
    """
    cost_fn = get_function_by_name(func_name)

    ga = GeneticAlgorithm(
        cost_function=cost_fn,
        dim=dim,
        bounds=bounds,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        step_size=step_size,
        elitism=elitism,
        selection_method=selection_method,
        crossover_method=crossover_method,
        tournament_k=3,
        random_seed=random_seed,
    )

    best_x, best_f = ga.run()

    return ga.history_best, ga.history_mean, ga.history_std, ga.history_worst, best_x, best_f


def plot_convergence_single(
    history_best,
    history_mean,
    history_std,
    history_worst=None,
    title: str = "",
    filename: str | None = None,
):
    """
    Egy paraméterkészlet konvergenciáját rajzolja ki.
    - history_best: list[float]
    - history_mean: list[float]
    - history_std: list[float]
    """

    generations = np.arange(1, len(history_best) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(generations, history_best, label="Legjobb érték")
    plt.plot(generations, history_mean, label="Átlagos érték", linestyle="--")

    # szórás sávként (mean ± std)
    history_best = np.array(history_best)
    history_mean = np.array(history_mean)
    history_std = np.array(history_std)

    plt.fill_between(
        generations,
        history_mean - history_std,
        history_mean + history_std,
        alpha=0.2,
        label="Átlag ± szórás",
    )

    if history_worst is not None:
        history_worst = np.array(history_worst)
        plt.plot(generations, history_worst, label="Legrosszabb érték", linestyle=":")

    plt.yscale("log")  # rastrigin miatt jól mutat log skálán
    plt.xlabel("Generáció")
    plt.ylabel("Célfüggvény értéke (log skála)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Mentve: {filename}")

    plt.show()


def plot_convergence_compare_two(
    history1,
    history2,
    label1: str,
    label2: str,
    title: str,
    filename: str | None = None,
):
    """
    Két különböző paraméterkészlet legjobb értékeinek konvergenciáját hasonlítja össze.
    history1, history2: list[float] (history_best)
    """
    g1 = np.arange(1, len(history1) + 1)
    g2 = np.arange(1, len(history2) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(g1, history1, label=label1)
    plt.plot(g2, history2, label=label2)

    plt.yscale("log")
    plt.xlabel("Generáció")
    plt.ylabel("Legjobb célfüggvény érték (log skála)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Mentve: {filename}")

    plt.show()


if __name__ == "__main__":
    # Példa: két különböző paraméterkészlet összehasonlítása Rastrigin 2D-n

    func_name = "rastrigin"
    dim = 2
    bounds = (-5.12, 5.12)

    # 1. „jó” paraméterkészlet – a sweepből is láttuk, hogy ilyenek jók:
    cfg_good = dict(
        func_name=func_name,
        dim=dim,
        bounds=bounds,
        generations=100,
        population_size=100,
        step_size=0.1,
        mutation_rate=0.3,
        crossover_rate=0.9,
        selection_method="tournament",
        crossover_method="path_relink",
        elitism=True,
        random_seed=42,
    )

    # 2. „gyengébb” paraméterkészlet – kevés generáció, kicsi populáció, nincs elitizmus
    cfg_bad = dict(
        func_name=func_name,
        dim=dim,
        bounds=bounds,
        generations=20,
        population_size=10,
        step_size=2.0,
        mutation_rate=0.3,
        crossover_rate=0.9,
        selection_method="diversity",
        crossover_method="one_point",
        elitism=False,
        random_seed=42,
    )

    h1_best, h1_mean, h1_std, h1_worst, x1, f1 = run_ga_and_get_history(**cfg_good)
    h2_best, h2_mean, h2_std, h2_worst, x2, f2 = run_ga_and_get_history(**cfg_bad)

    # Egyenként is kirajzolhatod:
    plot_convergence_single(
        h1_best,
        h1_mean,
        h1_std,
        h1_worst,
        title="Konvergencia – jó paraméterkészlet (Rastrigin 2D)",
        filename="conv_rastrigin_good.png",
    )

    plot_convergence_single(
        h2_best,
        h2_mean,
        h2_std,
        h2_worst,
        title="Konvergencia – gyengébb paraméterkészlet (Rastrigin 2D)",
        filename="conv_rastrigin_bad.png",
    )

    # Összehasonlító grafikon (csak a legjobb értékek)
    plot_convergence_compare_two(
        h1_best,
        h2_best,
        label1="Jó paraméterkészlet",
        label2="Gyengébb paraméterkészlet",
        title="Konvergencia összehasonlítás – Rastrigin 2D",
        filename="conv_rastrigin_compare.png",
    )
