from typing import Optional, List, Tuple
import tkinter as tk
from tkinter import ttk, messagebox
import re
import math
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def parse_graph(text: str) -> Tuple[nx.Graph, List[str]]:
    text = text.strip()
    if not text:
        raise ValueError("Empty graph description.")
    if ":" in text and "=" in text:
        return _parse_adj_list(text)
    else:
        return _parse_adj_matrix(text)

def _parse_adj_list(text: str) -> Tuple[nx.Graph, List[str]]:
    G = nx.Graph()
    lines = [ln.strip().rstrip(";") for ln in text.splitlines() if ln.strip()]
    labels: List[str] = []
    for ln in lines:
        if ":" not in ln:
            continue
        node, rest = [p.strip() for p in ln.split(":", 1)]
        if node not in labels:
            labels.append(node)
        if rest:
            for item in re.split(r",|\s+", rest):
                item = item.strip()
                if not item:
                    continue
                m = re.match(r"([A-Za-z0-9_]+)\s*[=:]?\s*([0-9]*\.?[0-9]+)", item)
                if not m:
                    continue
                nbr, w = m.group(1), float(m.group(2))
                if nbr not in labels:
                    labels.append(nbr)
                if node != nbr:
                    G.add_edge(node, nbr, weight=w)
    for lab in labels:
        if lab not in G:
            G.add_node(lab)
    return G, labels

def _parse_adj_matrix(text: str) -> Tuple[nx.Graph, List[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    rows: List[List[str]] = []
    for ln in lines:
        if "," in ln:
            parts = [p.strip() for p in ln.split(",")]
        else:
            parts = [p for p in ln.replace("\t", " ").split()]
        rows.append(parts)

    header_mode = any(not _is_number(tok) and tok != "" for tok in rows[0])
    if header_mode:
        labels = [tok for tok in rows[0] if tok != ""]
        matrix_rows = rows[1:]
        if len(matrix_rows) != len(labels):
            raise ValueError("Header/matrix size mismatch.")
        G = nx.Graph()
        for i, row in enumerate(matrix_rows):
            if not _is_number(row[0]):
                row = row[1:]
            for j, tok in enumerate(row):
                if j <= i:
                    continue
                w = float(tok)
                if w > 0 and not math.isinf(w):
                    G.add_edge(labels[i], labels[j], weight=w)
        for lab in labels:
            if lab not in G:
                G.add_node(lab)
        return G, labels
    else:
        n = len(rows)
        G = nx.Graph()
        labels = [str(i) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                w = float(rows[i][j])
                if w > 0 and not math.isinf(w):
                    G.add_edge(labels[i], labels[j], weight=w)
        for lab in labels:
            if lab not in G:
                G.add_node(lab)
        return G, labels

class GA_TSP:
    def __init__(self, G: nx.Graph, selection: str = "ranking", crossover_rate: float = 0.9,
                 mutation_rate: float = 0.05, pop_size: int = 100, elitism_pct: float = 0.05,
                 random_seed: Optional[int] = 42):
        self.G = G
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)
        self.selection = selection
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.pop_size = max(4, pop_size)
        self.elitism_count = max(0, int(elitism_pct * self.pop_size))
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def route_length(self, route: List[str]) -> float:
        total = 0.0
        for i in range(len(route)):
            a = route[i]
            b = route[(i + 1) % len(route)]
            if self.G.has_edge(a, b):
                total += float(self.G[a][b].get("weight", 1.0))
            else:
                total += 1e6
        return total

    def fitness(self, route: List[str]) -> float:
        dist = self.route_length(route)
        return 1.0 / (dist + 1e-9)

    def random_route(self) -> List[str]:
        r = self.nodes[:]
        random.shuffle(r)
        return r

    def select_parents(self, population: List[List[str]]) -> List[List[str]]:
        if self.selection == "tournament":
            return self._tournament_selection(population)
        else:
            return self._ranking_selection(population)

    def _ranking_selection(self, population: List[List[str]]) -> List[List[str]]:
        pop_sorted = sorted(population, key=lambda r: self.fitness(r), reverse=True)
        ranks = np.arange(len(pop_sorted), 0, -1)
        probs = ranks / ranks.sum()
        parents = random.choices(pop_sorted, weights=probs, k=len(population))
        return parents

    def _tournament_selection(self, population: List[List[str]], k: int = 3) -> List[List[str]]:
        parents = []
        for _ in range(len(population)):
            contenders = random.sample(population, k=min(k, len(population)))
            best = max(contenders, key=lambda r: self.fitness(r))
            parents.append(best)
        return parents

    def crossover(self, p1: List[str], p2: List[str]) -> Tuple[List[str], List[str]]:
        if random.random() > self.crossover_rate:
            return p1[:], p2[:]
        a, b = sorted(random.sample(range(self.N), 2))

        def ox(parent_a, parent_b):
            child = [None] * self.N
            child[a:b+1] = parent_a[a:b+1]
            fill = [g for g in parent_b if g not in child]
            idx = 0
            for i in range(self.N):
                if child[i] is None:
                    child[i] = fill[idx]
                    idx += 1
            return child

        return ox(p1, p2), ox(p2, p1)

    def mutate(self, route: List[str]) -> None:
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.N), 2)
            route[i], route[j] = route[j], route[i]

    def run(self, max_generations: int = 200, saturation_generations: Optional[int] = None):
        population = [self.random_route() for _ in range(self.pop_size)]
        best_route = min(population, key=lambda r: self.route_length(r))
        best_dist = self.route_length(best_route)
        history_best = [best_dist]
        no_improve = 0
        for _ in range(1, max_generations + 1):
            scored = sorted(population, key=lambda r: self.route_length(r))
            elites = scored[: self.elitism_count] if self.elitism_count > 0 else []
            parents = self.select_parents(population)
            offspring: List[List[str]] = []
            for i in range(0, len(parents) - 1, 2):
                c1, c2 = self.crossover(parents[i], parents[i + 1])
                self.mutate(c1)
                self.mutate(c2)
                offspring.extend([c1, c2])
            while len(offspring) + len(elites) < self.pop_size:
                offspring.append(self.random_route())
            population = elites + offspring[: self.pop_size - len(elites)]
            current_best = min(population, key=lambda r: self.route_length(r))
            current_dist = self.route_length(current_best)
            if current_dist + 1e-9 < best_dist:
                best_dist = current_dist
                best_route = current_best
                no_improve = 0
            else:
                no_improve += 1
            history_best.append(best_dist)
            if saturation_generations is not None and no_improve >= saturation_generations:
                break
        return best_route, best_dist, history_best

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Project 3 – Traveling & Shipment Routing (GA)")
        self.geometry("1200x780")
        self.configure(bg="#ffdca6")
        self.graph_text: Optional[tk.Text] = None
        self.G: Optional[nx.Graph] = None
        self.labels: List[str] = []
        self.stop_mode = tk.StringVar(value="saturation")
        self.stop_value = tk.IntVar(value=50)
        self.pop_size = tk.IntVar(value=100)
        self.crossover = tk.DoubleVar(value=0.9)
        self.mutation = tk.DoubleVar(value=0.05)
        self.elitism = tk.DoubleVar(value=0.05)
        self.selection_mode = tk.StringVar(value="ranking")
        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self, bg="#000")
        header.pack(fill=tk.X)
        tk.Label(header, text="Project 3", fg="white", bg="#000", font=("Consolas", 22, "bold")).pack(side=tk.LEFT, padx=10, pady=6)
        tk.Label(header, text="Traveling & Shipment Routing Using Genetic Algorithm", fg="#f28b30", bg="#000", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=12)
        top = tk.Frame(self, bg="#ffdca6")
        top.pack(fill=tk.X, padx=16, pady=10)
        tk.Label(top, text="Graph description (Adjacency list or matrix):", bg="#ffdca6", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.graph_text = tk.Text(top, height=7, width=60)
        self.graph_text.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(0, 10))
        top.grid_columnconfigure(0, weight=1)
        ttk.Button(top, text="Create Graph", command=self.on_create_graph).grid(row=1, column=3, padx=6)
        card = tk.Frame(self, bg="#f9cf88", highlightbackground="#d5ac74", highlightthickness=2)
        card.pack(fill=tk.X, padx=16, pady=8)
        box1 = tk.LabelFrame(card, text="Stopping criteria", bg="#f9cf88")
        box1.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        ttk.Radiobutton(box1, text="Saturation (no improvement)", variable=self.stop_mode, value="saturation").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(box1, text="After Generations", variable=self.stop_mode, value="generations").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(box1, from_=1, to=10000, textvariable=self.stop_value, width=10).grid(row=0, column=1, rowspan=2, padx=6)
        box2 = tk.LabelFrame(card, text="Parameters", bg="#f9cf88")
        box2.grid(row=0, column=1, padx=10, pady=10, sticky="nw")
        _row = 0
        for label, var, frm, to, inc in [
            ("Population", self.pop_size, 10, 10000, 10),
            ("Crossover", self.crossover, 0.0, 1.0, 0.01),
            ("Mutation", self.mutation, 0.0, 1.0, 0.01),
            ("Elitism", self.elitism, 0.0, 1.0, 0.01),
        ]:
            tk.Label(box2, text=label + ":", bg="#f9cf88").grid(row=_row, column=0, sticky="w")
            ttk.Spinbox(box2, from_=frm, to=to, increment=inc, textvariable=var, width=10).grid(row=_row, column=1, padx=6)
            _row += 1
        box3 = tk.LabelFrame(card, text="Selection", bg="#f9cf88")
        box3.grid(row=0, column=2, padx=10, pady=10, sticky="nw")
        ttk.Radiobutton(box3, text="Ranking", variable=self.selection_mode, value="ranking").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(box3, text="Tournament", variable=self.selection_mode, value="tournament").grid(row=1, column=0, sticky="w")
        controls = tk.Frame(self, bg="#ffdca6")
        controls.pack(fill=tk.X, padx=16)
        ttk.Button(controls, text="Run GA", command=self.on_run).pack(side=tk.LEFT)
        ttk.Button(controls, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=6)
        charts = tk.Frame(self, bg="#ffdca6")
        charts.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)
        self.fig = Figure(figsize=(6, 3.6), dpi=100)
        self.ax_fit = self.fig.add_subplot(121)
        self.ax_graph = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=charts)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.status = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status, bg="#ffdca6").pack(fill=tk.X, padx=16, pady=(0, 10))

    def on_create_graph(self):
        text = self.graph_text.get("1.0", tk.END)
        try:
            G, labels = parse_graph(text)
        except Exception as e:
            messagebox.showerror("Parse error", str(e))
            return
        self.G, self.labels = G, labels
        self.status.set(f"Graph: {len(labels)} nodes, {G.number_of_edges()} edges")
        self.draw_graph()

    def on_run(self):
        if self.G is None:
            messagebox.showwarning("Graph", "Please create the graph first.")
            return
        sel = self.selection_mode.get()
        ga = GA_TSP(
            self.G,
            selection=sel,
            crossover_rate=float(self.crossover.get()),
            mutation_rate=float(self.mutation.get()),
            pop_size=int(self.pop_size.get()),
            elitism_pct=float(self.elitism.get()),
        )
        if self.stop_mode.get() == "generations":
            best_route, best_dist, history = ga.run(
                max_generations=int(self.stop_value.get()),
                saturation_generations=None
            )
        else:
            best_route, best_dist, history = ga.run(
                max_generations=10000,
                saturation_generations=int(self.stop_value.get())
            )
        self.plot_fitness(history)
        self.draw_graph(best_route)
        self.status.set(f"Best distance: {best_dist:.3f} | Best route: {' -> '.join(best_route)}")

    def clear(self):
        self.ax_fit.clear()
        self.ax_fit.set_title("Best Fitness")
        self.ax_fit.set_xlabel("Generation")
        self.ax_fit.set_ylabel("1 / Distance")
        self.ax_graph.clear()
        self.canvas.draw_idle()
        self.status.set("Ready")

    def plot_fitness(self, history: List[float]):
        self.ax_fit.clear()
        fitness = [1.0 / (d + 1e-9) for d in history]
        self.ax_fit.plot(range(len(fitness)), fitness)
        self.ax_fit.set_title("The Best Fitness")
        self.ax_fit.set_xlabel("Generation")
        self.ax_fit.set_ylabel("Fitness (1/Distance)")
        self.canvas.draw_idle()

    def draw_graph(self, route: Optional[List[str]] = None):
        if self.G is None:
            return
        self.ax_graph.clear()
        self.ax_graph.set_title("Graph / Best Route")
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw(self.G, pos, ax=self.ax_graph, with_labels=True, node_color="#DDDDDD")
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos,
                                     edge_labels={e: f"{w:.0f}" for e, w in labels.items()},
                                     ax=self.ax_graph)
        if route is not None:
            cycle_edges = [(route[i], route[(i + 1) % len(route)]) for i in range(len(route))]
            nx.draw_networkx_edges(self.G, pos, edgelist=cycle_edges, width=3.0,
                                   edge_color="#1f77b4", ax=self.ax_graph)
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = App()
    app.mainloop()
