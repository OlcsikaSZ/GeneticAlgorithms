# early_stop_experiments.py
import time
from continuous.functions import rastrigin
from continuous.ga_core import GeneticAlgorithm

common = dict(
    cost_function=rastrigin,
    dim=2,
    bounds=(-5.12, 5.12),
    population_size=100,
    crossover_rate=0.9,
    mutation_rate=0.3,
    step_size=0.5,
    elitism=True,
    selection_method="tournament",
    crossover_method="path_relink",
    tournament_k=2,
    random_seed=42,
)

# 1) Fix G = 200, nincs early stop
t0 = time.perf_counter()
ga_fix = GeneticAlgorithm(generations=200, **common)
x_fix, f_fix = ga_fix.run()
t_fix = time.perf_counter() - t0
g_fix = len(ga_fix.history_best)

print("FIX G=200 -> G_futott:", g_fix, "best:", f_fix, "time:", t_fix)

# 2) Early stop toleranciával
t0 = time.perf_counter()
ga_tol = GeneticAlgorithm(
    generations=200,
    tolerance=1e-8,   # célpontosság
    **common,
)
x_tol, f_tol = ga_tol.run()
t_tol = time.perf_counter() - t0
g_tol = len(ga_tol.history_best)

print("EARLY STOP (tol=1e-8) -> G_futott:", g_tol, "best:", f_tol, "time:", t_tol)

# 3) Early stop patience-szel
t0 = time.perf_counter()
ga_pat = GeneticAlgorithm(
    generations=200,
    patience=20,      # 20 generáció javulás nélkül
    **common,
)
x_pat, f_pat = ga_pat.run()
t_pat = time.perf_counter() - t0
g_pat = len(ga_pat.history_best)

print("EARLY STOP (pat=20) -> G_futott:", g_pat, "best:", f_pat, "time:", t_pat)