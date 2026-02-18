import random
from typing import Callable, Tuple, List

import numpy as np

Bounds = Tuple[float, float]  # (min, max)


class GeneticAlgorithm:
    """
    Egyszerű, általános GA folytonos függvények minimalizálására.

    - kromoszóma: np.ndarray (valós számok vektora)
    - cél: MINIMALIZÁLNI a cost függvényt (fitness = cost, kisebb a jobb)
    """

    def __init__(
        self,
        cost_function: Callable[[np.ndarray], float],
        dim: int,
        bounds: Bounds,
        population_size: int = 20,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        step_size: float = 0.1,
        tournament_k: int = 3,
        elitism: bool = True,
        selection_method: str = "tournament",
        crossover_method: str = "one_point",
        crossover_points: int = 2,
        uniform_swap_prob: float = 0.5,
        random_seed: int | None = None,
        tolerance: float | None = None,
        patience: int | None = None,
    ):
        self.cost_function = cost_function
        self.dim = dim
        self.bounds = bounds
        self.pop_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.step_size = step_size
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.selection_method = selection_method

        # új attribútumok:
        self.crossover_method = crossover_method.lower()
        self.crossover_points = crossover_points
        self.uniform_swap_prob = uniform_swap_prob
        self.tolerance = tolerance
        self.patience = patience

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.population: np.ndarray = self._init_population()
        self.costs: np.ndarray = self._evaluate_population()

        self.history_best: List[float] = []
        self.history_mean: List[float] = []
        self.history_std: List[float] = []
        self.history_worst: List[float] = []

    # ----- inicializálás ------------------------------------------------------

    def _init_population(self) -> np.ndarray:
        low, high = self.bounds
        return np.random.uniform(low=low, high=high, size=(self.pop_size, self.dim))

    def _evaluate_population(self) -> np.ndarray:
        return np.array([self.cost_function(ind) for ind in self.population])

    # ----- szelekció: tournament selection -----------------------------------

    def _tournament_select(self) -> np.ndarray:
        """
        Tournament selection:
        - Véletlenül kiválaszt k darab egyedet
        - Közülük a legkisebb cost (legjobb) nyer
        """
        indices = np.random.choice(self.pop_size, size=self.tournament_k, replace=False)
        best_idx = indices[np.argmin(self.costs[indices])]
        return self.population[best_idx].copy()

    # ----- szelekció: rulettekerék minimumnál --------------------------------

    def _roulette_select(self) -> np.ndarray:
        """
        Rulettekerekes szelekció MINIMALIZÁLÁSNÁL.

        Kisebb cost = jobb. Átalakítjuk 'jósággá' (fitness),
        és aszerint sorsolunk.
        """
        # kis eps, hogy ne legyen 0 osztás
        eps = 1e-12
        max_cost = float(np.max(self.costs))

        # ha valami fura, és mindenki ugyanaz, válasszunk randomot
        if not np.isfinite(max_cost):
            idx = np.random.randint(0, self.pop_size)
            return self.population[idx].copy()

        # jóság: minél kisebb a cost, annál nagyobb legyen a fitness
        fitness = max_cost - self.costs + eps

        # ha minden fitness kb. nulla → fallback randomra
        total_fit = float(np.sum(fitness))
        if total_fit <= 0 or not np.isfinite(total_fit):
            idx = np.random.randint(0, self.pop_size)
            return self.population[idx].copy()

        probs = fitness / total_fit
        idx = np.random.choice(self.pop_size, p=probs)
        return self.population[idx].copy()
    
    # ----- szelekció: rank szelekció --------------------------------
    
    def _rank_select(self) -> np.ndarray:
        """
        Rank-based selection (rangsor szerinti kiválasztás).

        - sorbarendezi az egyedeket cost szerint (kisebb = jobb)
        - a legjobb kapja a legnagyobb rangsúlyt (N), a legrosszabb a legkisebbet (1)
        - a rangok alapján sorsol
        """
        # indexek cost szerint rendezve: [best_idx, ..., worst_idx]
        sorted_indices = np.argsort(self.costs)
        n = self.pop_size

        # fitness súlyok: best -> N, worst -> 1
        fitness = np.zeros(n, dtype=float)
        for rank, idx in enumerate(sorted_indices):
            fitness[idx] = n - rank  # rank=0 (best) → N, rank=n-1 (worst) → 1

        # valószínűségi eloszlás
        probs = fitness / np.sum(fitness)

        selected_idx = np.random.choice(n, p=probs)
        return self.population[selected_idx].copy()
    
    # ----- szelekció: fitness-rank szelekció --------------------------------

    def _fitness_rank_select(self) -> np.ndarray:
        """
        Fitness-rank selection:
        - a fitness (1/cost vagy max-cost átalakítás) és
          a rang súly (N..1) kombinációja
        """
        n = self.pop_size

        # rangsor meghatározása
        sorted_indices = np.argsort(self.costs)

        # rang súlyok: best=N, worst=1
        rank_weights = np.arange(n, 0, -1)  # pl. 100..1

        # fitness átalakítás minimálásra
        max_cost = float(np.max(self.costs))
        eps = 1e-12
        fitness_base = max_cost - self.costs + eps  # mint a roulette-nél

        # kombinált fitness: rang * (max_cost - cost)
        fitness = np.zeros(n, dtype=float)
        for pos, idx in enumerate(sorted_indices):
            fitness[idx] = rank_weights[pos] * fitness_base[idx]

        # valószínűségi eloszlás
        probs = fitness / np.sum(fitness)

        selected_idx = np.random.choice(n, p=probs)
        return self.population[selected_idx].copy()
    
    # ----- szelekció: diverzitás alapú szelekció --------------------------------
    
    def _diversity_select(self) -> np.ndarray:
        """
        Diverzitás alapú szelekció:
        - minden egyedhez kiszámoljuk az átlagos távolságot a többitől
        - nagyobb diverzitás = nagyobb esély
        """
        n = self.pop_size

        # távolságmátrix kiszámítása
        distances = np.zeros(n)
        for i in range(n):
            distances[i] = np.mean(np.linalg.norm(self.population[i] - self.population, axis=1))

        # normalizálás
        fitness = distances + 1e-12
        probs = fitness / np.sum(fitness)

        idx = np.random.choice(n, p=probs)
        return self.population[idx].copy()

    # ----- közös szülőválasztó ------------------------------------------------

    def _select_parent(self) -> np.ndarray:
        method = self.selection_method.lower()
        if method == "tournament":
            return self._tournament_select()
        elif method == "roulette":
            return self._roulette_select()
        elif method == "rank":
            return self._rank_select()
        elif method == "fitness_rank":
            return self._fitness_rank_select()
        elif method == "diversity":
            return self._diversity_select()
        else:
            return self._tournament_select()
        
    # ----- keresztezés: többféle crossover ----------------------------------

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Keresztezés a beállított crossover_method alapján.

        Lehetséges értékek:
        - "one_point"
        - "two_point"
        - "k_point" (multipoint, self.crossover_points alapján)
        - "uniform"
        - "path_relink"
        """
        # nincs keresztezés (csak másoljuk a szülőket)
        if random.random() > self.crossover_rate or self.dim == 1:
            return parent1.copy(), parent2.copy()

        method = self.crossover_method

        if method == "one_point":
            return self._one_point_crossover(parent1, parent2)
        elif method == "two_point":
            return self._two_point_crossover(parent1, parent2)
        elif method == "k_point":
            return self._k_point_crossover(parent1, parent2, self.crossover_points)
        elif method == "uniform":
            return self._uniform_crossover(parent1, parent2)
        elif method == "path_relink":
            return self._path_relink_crossover(parent1, parent2)
        else:
            # ha valami hülyeséget adunk meg, essünk vissza a one_point-ra
            return self._one_point_crossover(parent1, parent2)
        
    # ===== különböző crossover operátorok ===============================

    def _one_point_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Egypontos crossover (régi viselkedés)."""
        point = random.randint(1, self.dim - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _k_point_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-pontos crossover (k >= 1).
        A kromoszóma több szakaszra vágása, szakaszonként szülőváltással.
        """
        if self.dim <= 1:
            return parent1.copy(), parent2.copy()

        k = max(1, min(k, self.dim - 1))  # ne legyen több pont, mint dim-1
        points = sorted(random.sample(range(1, self.dim), k))
        points = [0] + points + [self.dim]

        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        for i in range(len(points) - 1):
            start, end = points[i], points[i + 1]
            if i % 2 == 0:
                # páros szegmensek: eredeti sorrend
                child1[start:end] = parent1[start:end]
                child2[start:end] = parent2[start:end]
            else:
                # páratlan szegmensek: csere
                child1[start:end] = parent2[start:end]
                child2[start:end] = parent1[start:end]

        return child1, child2

    def _two_point_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kétpontos crossover – speciális esete a K-pontosnak (k=2)."""
        return self._k_point_crossover(parent1, parent2, k=2)

    def _uniform_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform crossover:
        minden génnél függetlenül döntünk, hogy melyik szülőtől jöjjön.
        """
        mask = np.random.rand(self.dim) < self.uniform_swap_prob
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _path_relink_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Egyszerű "path relinking" folyamatos térben:
        a két szülő közötti szakaszon veszünk két pontot.
        """
        alpha = random.random()  # 0..1
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    # ----- mutáció: kis véletlen eltolás + levágás a tartományra -------------

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        low, high = self.bounds
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                individual[i] += np.random.normal(loc=0.0, scale=self.step_size)
        individual = np.clip(individual, low, high)
        return individual

    # ----- fő ciklus ----------------------------------------------------------

    def run(self) -> Tuple[np.ndarray, float]:
        # kezdő populáció alapján már vannak self.costs értékek
        best_overall = float(np.min(self.costs))
        no_improve = 0

        for gen in range(self.generations):
            new_population = []

            if self.elitism:
                best_idx = int(np.argmin(self.costs))
                elite = self.population[best_idx].copy()
                new_population.append(elite)

            while len(new_population) < self.pop_size:
                p1 = self._select_parent()
                p2 = self._select_parent()

                c1, c2 = self._crossover(p1, p2)

                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            self.population = np.array(new_population)
            self.costs = self._evaluate_population()

            best = float(np.min(self.costs))
            mean = float(np.mean(self.costs))
            std = float(np.std(self.costs))
            worst = float(np.max(self.costs))

            self.history_best.append(best)
            self.history_mean.append(mean)
            self.history_std.append(std)
            self.history_worst.append(worst)

            # --- EARLY STOP LOGIKA ---
            # ha javult a best_overall, nullázzuk a számlálót
            if best < best_overall - 1e-12:
                best_overall = best
                no_improve = 0
            else:
                no_improve += 1

            # 1) abszolút tolerancia
            if self.tolerance is not None and best_overall <= self.tolerance:
                # elértük a kívánt pontosságot
                break

            # 2) patience generáción át nincs javulás
            if self.patience is not None and no_improve >= self.patience:
                break

        best_idx = int(np.argmin(self.costs))
        best_individual = self.population[best_idx]
        best_cost = float(self.costs[best_idx])
        return best_individual, best_cost
