# continuous/table_gen.py

import pandas as pd


def load_results(csv_path: str = "continuous_param_sweep.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def make_summary_table(
    df: pd.DataFrame,
    func_name: str = "rastrigin",
    top_n: int = 10,
    outfile: str = "results_table.tex",
):
    # csak az adott célfüggvény
    df_func = df[df["func_name"] == func_name].copy()

    # legjobb N konfiguráció a best_cost alapján
    df_best = df_func.sort_values("best_cost").head(top_n)

    rows = []

    for _, row in df_best.iterrows():
        G = int(row["generations"])
        K = int(row["population_size"])
        L = row["step_size"]

        # "Val. fv." – szelekció + crossover rövid leírása
        # LaTeX-ben az '_' speciális karakter, ezért escape-eljük
        valfv_raw = f"{row['selection_method']}, {row['crossover_method']}"
        valfv = valfv_raw.replace("_", r"\_")

        elit = "Igen" if row["elitism"] else "Nem"
        celfv = func_name.capitalize()

        # Eredmény: f(x) = érték
        eredm = f"f({row['best_x']}) = {row['best_cost']:.3e}"

        t = f"{row['time_sec']:.4f} s"

        # START oszlopot itt nem külön mérjük – egyszerűség kedvéért kihagyva
        line = f"{G} & {K} & {L} & {valfv} & {elit} & {celfv} & {eredm} & {t} \\\\"
        rows.append(line)

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rrrrllll}\n")
        f.write("G & K & L & Val. fv. & Elit & Célfv. & Eredmény & t \\\\\n")
        f.write("\\hline\n")
        for line in rows:
            f.write(line + "\n")
        f.write("\\end{tabular}\n")

    print(f"Mentve: {outfile}")
    print("Első sorok a táblából:")
    for r in rows[:3]:
        print(r)

def make_rastrigin_nd_crossover_table(
    csv_path: str = "rastrigin_nd_crossovers.csv",
    outfile: str = "results_rastrigin_nd_crossovers.tex",
):
    """
    LaTeX táblázat a Rastrigin ND + különböző crossover módszerek kísérlethez.

    Bemenet: rastrigin_nd_crossovers.csv (dim, crossover_method, best_cost, time_sec)
    Kimenet: results_rastrigin_nd_crossovers.tex
    """

    df = pd.read_csv(csv_path)

    # rendezzük dimenzió és crossover szerint, hogy szép legyen
    df_sorted = df.sort_values(["dim", "crossover_method"])

    rows = []

    for _, row in df_sorted.iterrows():
        d = int(row["dim"])
        cm = str(row["crossover_method"]).replace("_", r"\_")
        best = f"{row['best_cost']:.3e}"
        t = f"{row['time_sec']:.3f} s"

        line = f"{d} & {cm} & {best} & {t} \\\\"
        rows.append(line)

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rlll}\n")
        f.write("d & Keresztezés & Legjobb $f(x)$ & Idő [s] \\\\\n")
        f.write("\\hline\n")
        for line in rows:
            f.write(line + "\n")
        f.write("\\end{tabular}\n")

    print(f"Mentve: {outfile}")
    print("Első sorok a táblából:")
    for r in rows[:5]:
        print(r)

if __name__ == "__main__":
    # Folytonos paraméter-söprés eredményei
    df = load_results("continuous_param_sweep.csv")

    # Rastrigin
    make_summary_table(
        df,
        func_name="rastrigin",
        top_n=10,
        outfile="results_rastrigin.tex",
    )

    # Booth
    make_summary_table(
        df,
        func_name="booth",
        top_n=10,
        outfile="results_booth.tex",
    )

    # Lévi
    make_summary_table(
        df,
        func_name="levi",
        top_n=10,
        outfile="results_levi.tex",
    )

    # Rastrigin ND + crossover kísérlet táblája
    make_rastrigin_nd_crossover_table(
        csv_path="rastrigin_nd_crossovers.csv",
        outfile="results_rastrigin_nd_crossovers.tex",
    )
