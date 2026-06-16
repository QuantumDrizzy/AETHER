"""Watts–Strogatz small-world networks — short paths, high clustering.

Start from a ring lattice (N nodes, each joined to its k nearest neighbours) and
rewire each edge with probability p. At p = 0 the graph is highly clustered but has
long paths (you walk around the ring); at p = 1 it is a random graph — short paths
but barely clustered. The Watts–Strogatz result is that a *tiny* amount of rewiring
collapses the path length while clustering stays high: a **small-world** regime
(your friends know each other, yet anyone is a few hops away).

It is the network-science member of the lab's connectivity/criticality thread —
the same kind of structure that shows up in brains, power grids and the web — and
sits naturally beside percolation (connectivity) and the substrates that compute on
graphs. Pure NumPy + BFS, no external graph library.
"""

from __future__ import annotations

import numpy as np


def watts_strogatz(n: int, k: int, p: float, seed: int = 0) -> list[set]:
    """Ring lattice of degree k, each edge rewired with probability p."""
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(n)]
    half = k // 2
    for i in range(n):
        for j in range(1, half + 1):
            adj[i].add((i + j) % n)
            adj[(i + j) % n].add(i)
    for i in range(n):
        for j in range(1, half + 1):
            if rng.random() < p:
                old = (i + j) % n
                if old not in adj[i]:
                    continue
                m = int(rng.integers(n))
                if m != i and m not in adj[i]:
                    adj[i].discard(old); adj[old].discard(i)
                    adj[i].add(m); adj[m].add(i)
    return adj


def clustering_coefficient(adj: list[set]) -> float:
    """Average local clustering: fraction of a node's neighbours that are linked."""
    vals = []
    for i, nbrs in enumerate(adj):
        d = len(nbrs)
        if d < 2:
            continue
        links = sum(1 for a in nbrs for b in nbrs if a < b and b in adj[a])
        vals.append(links / (d * (d - 1) / 2))
    return float(np.mean(vals)) if vals else 0.0


def avg_path_length(adj: list[set]) -> float:
    """Mean shortest-path length over all connected node pairs (BFS)."""
    n = len(adj)
    total, pairs = 0, 0
    for src in range(n):
        dist = {src: 0}
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        for v, dd in dist.items():
            if v != src:
                total += dd; pairs += 1
    return total / pairs if pairs else 0.0


def sweep_rewiring(ps, n: int = 200, k: int = 6, seed: int = 0):
    """Return (C(p)/C(0), L(p)/L(0)) — the small-world signature."""
    base = watts_strogatz(n, k, 0.0, seed)
    C0, L0 = clustering_coefficient(base), avg_path_length(base)
    C, L = [], []
    for p in ps:
        g = watts_strogatz(n, k, float(p), seed)
        C.append(clustering_coefficient(g) / C0)
        L.append(avg_path_length(g) / L0)
    return np.asarray(C), np.asarray(L)
