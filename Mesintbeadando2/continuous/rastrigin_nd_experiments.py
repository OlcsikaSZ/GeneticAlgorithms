# continuous/rastrigin_nd_experiments.py
import time
from continuous.functions import rastrigin
from continuous.ga_core import GeneticAlgorithm

# közös beállítások minden dimenzióra
common = dict(
    cost_function=rastrigin,
    bounds=(-5.12, 5.12),
    population_size=100,
    generations=200,
    crossover_rate=0.9,
    mutation_rate=0.3,
    step_size=0.1,           # finom mutáció, hogy magasabb dimenziókban se szálljon el
    elitism=True,
    selection_method="tournament",
    crossover_method="path_relink",
    tournament_k=2,
    random_seed=42,
)

dims = [2, 3, 4, 5, 10, 100]

print("Rastrigin több dimenzióban (azonos GA-paraméterekkel):")
for d in dims:
    t0 = time.perf_counter()
    ga = GeneticAlgorithm(dim=d, **common)
    x_best, f_best = ga.run()
    elapsed = time.perf_counter() - t0
    print(f"d = {d:3d} -> best f(x) = {f_best:.6e}, time = {elapsed:.3f} s")