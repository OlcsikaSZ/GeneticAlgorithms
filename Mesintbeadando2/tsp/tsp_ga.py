# tsp_ga.py

import random
from typing import Dict, Tuple, List, Optional
import numpy as np

CityIndex = int
DistancesDict = Dict[Tuple[CityIndex, CityIndex], float]


def tour_length(distances: DistancesDict, tour: List[int]) -> float:
    """Teljes körhossz kiszámítása (zárt kör, utolsó vissza az elsőre)."""
    dist = 0.0
    prev = tour[0]
    for city in tour[1:]:
        dist += distances[(prev, city)]
        prev = city
    dist += distances[(tour[-1], tour[0])]
    return dist


class TSPGeneticAlgorithm:
    """
    Egyszerű GA az utazó ügynök problémára (TSP).

    Reprezentáció: egy permutáció [0..n-1] a városindexekből.
    Cél: a körút hossza legyen MINIMÁLIS.
    """

    def __init__(
        self,
        distances: DistancesDict,
        num_cities: int,
        population_size: int = 50,
        generations: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        elitism: bool = True,
        selection_method: str = "tournament",  # vagy "roulette"
        tournament_k: int = 3,
        crossover_method: str = "ox",          # "ox" (order) vagy "pmx"
        random_seed: Optional[int] = None,
    ):
        self.distances = distances
        self.num_cities = num_cities
        self.pop_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.selection_method = selection_method.lower()
        self.tournament_k = tournament_k
        self.crossover_method = crossover_method.lower()

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # kezdeti populáció + értékelés
        self.population = self._init_population()
        self.costs = self._evaluate_population()

        # konvergencia loggolása
        self.history_best: List[float] = []
        self.history_mean: List[float] = []
        self.history_std: List[float] = []
        self.history_worst: List[float] = []

    # ---------- inicializálás ----------

    def _init_population(self) -> np.ndarray:
        """Véletlen permutációk populációja."""
        pop = []
        base = list(range(self.num_cities))
        for _ in range(self.pop_size):
            perm = base.copy()
            random.shuffle(perm)
            pop.append(perm)
        return np.array(pop, dtype=int)

    def _evaluate_population(self) -> np.ndarray:
        return np.array(
            [tour_length(self.distances, ind.tolist()) for ind in self.population],
            dtype=float,
        )

    # ---------- szelekció ----------

    def _tournament_select(self) -> np.ndarray:
        indices = np.random.choice(self.pop_size, size=self.tournament_k, replace=False)
        best_idx = indices[np.argmin(self.costs[indices])]
        return self.population[best_idx].copy()

    def _roulette_select(self) -> np.ndarray:
        # minimalizálunk → cost-ból fitness
        max_cost = float(np.max(self.costs))
        eps = 1e-12
        fitness = max_cost - self.costs + eps
        total = float(np.sum(fitness))
        if not np.isfinite(total) or total <= 0:
            idx = np.random.randint(0, self.pop_size)
        else:
            probs = fitness / total
            idx = np.random.choice(self.pop_size, p=probs)
        return self.population[idx].copy()

    def _select_parent(self) -> np.ndarray:
        if self.selection_method == "roulette":
            return self._roulette_select()
        else:
            return self._tournament_select()

    # ---------- crossover operátorok ----------

    def _crossover(self, p1: np.ndarray, p2: np.ndarray):
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()

        if self.crossover_method == "pmx":
            return self._pmx_crossover(p1, p2)
        else:
            return self._order_crossover(p1, p2)

    def _order_crossover(self, p1: np.ndarray, p2: np.ndarray):
        """Order crossover (OX) permutációkra."""
        n = self.num_cities
        a, b = sorted(random.sample(range(n), 2))
        child1 = np.full(n, -1, dtype=int)
        child2 = np.full(n, -1, dtype=int)

        # középső szegmens másolása
        child1[a:b] = p1[a:b]
        child2[a:b] = p2[a:b]

        def fill_child(child: np.ndarray, parent: np.ndarray, a: int, b: int) -> None:
            pos = b
            parent_list = parent.tolist()
            for gene in parent_list[b:] + parent_list[:b]:
                if gene not in child:
                    if pos == n:
                        pos = 0
                    child[pos] = gene
                    pos += 1

        fill_child(child1, p2, a, b)
        fill_child(child2, p1, a, b)
        return child1, child2

    def _pmx_crossover(self, p1: np.ndarray, p2: np.ndarray):
        """PMX (Partially Mapped Crossover)."""
        n = self.num_cities
        a, b = sorted(random.sample(range(n), 2))
        child1 = np.full(n, -1, dtype=int)
        child2 = np.full(n, -1, dtype=int)

        child1[a:b] = p1[a:b]
        child2[a:b] = p2[a:b]

        def pmx_fill(child: np.ndarray, parent_seg: np.ndarray, other_parent: np.ndarray):
            # 1) az ütközéseket feloldjuk a mappinggel
            for i in range(a, b):
                gene = other_parent[i]
                if gene in child:
                    continue
                pos = i
                while a <= pos < b:
                    mapped_gene = parent_seg[pos - a]
                    pos = int(np.where(other_parent == mapped_gene)[0][0])
                child[pos] = gene

            # 2) a maradék helyekre bemásoljuk az other_parent géneit
            for i in range(n):
                if child[i] == -1:
                    child[i] = other_parent[i]

        pmx_fill(child1, p1[a:b].copy(), p2.copy())
        pmx_fill(child2, p2[a:b].copy(), p1.copy())
        return child1, child2

    # ---------- mutáció ----------

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        """Két város felcserélése (swap mutáció)."""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.num_cities), 2)
            ind[i], ind[j] = ind[j], ind[i]
        return ind

    # ---------- fő ciklus ----------

    def run(self):
        for _ in range(self.generations):
            new_pop: List[np.ndarray] = []

            # elitizmus: legjobb egyed továbbvitele
            if self.elitism:
                best_idx = int(np.argmin(self.costs))
                elite = self.population[best_idx].copy()
                new_pop.append(elite)

            while len(new_pop) < self.pop_size:
                p1 = self._select_parent()
                p2 = self._select_parent()
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            self.population = np.array(new_pop, dtype=int)
            self.costs = self._evaluate_population()

            best = float(np.min(self.costs))
            mean = float(np.mean(self.costs))
            std = float(np.std(self.costs))
            worst = float(np.max(self.costs))

            self.history_best.append(best)
            self.history_mean.append(mean)
            self.history_std.append(std)
            self.history_worst.append(worst)

        best_idx = int(np.argmin(self.costs))
        best_tour = self.population[best_idx].copy()
        best_cost = float(self.costs[best_idx])
        return best_tour, best_cost
