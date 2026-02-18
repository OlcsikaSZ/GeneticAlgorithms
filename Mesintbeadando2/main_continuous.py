from continuous.functions import rastrigin
from continuous.ga_core import GeneticAlgorithm


if __name__ == "__main__":
    # Közös paraméterek minden futtatáshoz
    common_params = dict(
        cost_function=rastrigin,
        dim=2,
        bounds=(-5.12, 5.12),
        population_size=100,
        generations=200,
        crossover_rate=0.9,
        mutation_rate=0.3,
        step_size=0.5,
        elitism=True,
        tournament_k=2,
        random_seed=42,
    )

    # ------------------------------------------------------------------
    # 1) SZELEKCIÓS MÓDSZEREK ÖSSZEHASONLÍTÁSA (RÉGI RÉSZ, MEGMARAD)
    # ------------------------------------------------------------------
    print("### SZELEKCIÓS MÓDSZEREK (alapértelmezett: one_point crossover) ###")

    print("\n=== TOURNAMENT SELECTION ===")
    ga_tour = GeneticAlgorithm(selection_method="tournament", **common_params)
    best_x_t, best_f_t = ga_tour.run()
    print("Legjobb (tournament):", best_x_t)
    print("Értéke:", best_f_t)

    print("\n=== ROULETTE SELECTION ===")
    ga_roul = GeneticAlgorithm(selection_method="roulette", **common_params)
    best_x_r, best_f_r = ga_roul.run()
    print("Legjobb (roulette):", best_x_r)
    print("Értéke:", best_f_r)

    print("\n=== RANK SELECTION ===")
    ga_rank = GeneticAlgorithm(selection_method="rank", **common_params)
    best_x_k, best_f_k = ga_rank.run()
    print("Legjobb (rank):", best_x_k)
    print("Értéke:", best_f_k)

    print("\n=== FITNESS-RANK SELECTION ===")
    ga_fr = GeneticAlgorithm(selection_method="fitness_rank", **common_params)
    best_x_fr, best_f_fr = ga_fr.run()
    print("Legjobb (fitness_rank):", best_x_fr)
    print("Értéke:", best_f_fr)

    print("\n=== DIVERSITY-BASED SELECTION ===")
    ga_div = GeneticAlgorithm(selection_method="diversity", **common_params)
    best_x_div, best_f_div = ga_div.run()
    print("Legjobb (diversity):", best_x_div)
    print("Értéke:", best_f_div)

    # ------------------------------------------------------------------
    # 2) CROSSOVER MÓDSZEREK ÖSSZEHASONLÍTÁSA (ÚJ RÉSZ)
    # ------------------------------------------------------------------
    print("\n\n### CROSSOVER MÓDSZEREK (fix: tournament selection) ###")

    # itt fixáljuk a szelekciót tournamentre, hogy csak a crossover változzon
    crossover_base = dict(
        selection_method="tournament",
        **common_params,
    )

    for method in ["one_point", "two_point", "k_point", "uniform", "path_relink"]:
        print(f"\n=== {method.upper()} CROSSOVER ===")
        ga = GeneticAlgorithm(
            crossover_method=method,
            crossover_points=3,      # csak k_point-nál lényeges
            uniform_swap_prob=0.5,   # csak uniformnál lényeges
            **crossover_base,
        )
        best_x, best_f = ga.run()
        print("Legjobb x:", best_x)
        print("Értéke:", best_f)
