"""General tight-binding solver — build H(k) for an arbitrary lattice model and
diagonalize over a batch of k-points, on CPU (NumPy) or GPU (torch).

Graphene (2 orbitals) is the validation case: this solver must reproduce the
closed-form bands in `graphene.py`. The reason this module exists is the N×N
case — nanoribbons, supercells, multi-orbital materials (TMDCs) — where there is
no closed form and the k-space sweep is the expensive step that GPU batching
accelerates. The closed form is only available for toy models; this is the path
to real ones.

Convention: each hopping carries the real-space bond displacement
d = r_j − r_i + R, and contributes H_ij(k) += t·exp(i k·d) (+ Hermitian partner).
This matches the graphene structure-factor convention exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Hopping:
    """A single hopping term between orbitals i and j.

    `d` is the real-space bond displacement r_j − r_i + R (Å). `t` is the
    amplitude (eV; may be complex)."""

    i: int
    j: int
    d: tuple[float, float]
    t: complex = 1.0


@dataclass
class TightBinding:
    """A tight-binding model: N orbitals, a list of hoppings, optional on-site
    energies. Builds and diagonalizes H(k) over a batch of k-points."""

    n_orbitals: int
    hoppings: list[Hopping]
    onsite: np.ndarray | None = None  # shape (N,), eV

    def hamiltonian_batch(self, k: np.ndarray) -> np.ndarray:
        """Bloch Hamiltonian for a batch of k. k: (M, 2) -> H: (M, N, N) complex."""
        k = np.atleast_2d(np.asarray(k, dtype=float))
        m, n = k.shape[0], self.n_orbitals
        h = np.zeros((m, n, n), dtype=np.complex128)
        if self.onsite is not None:
            idx = np.arange(n)
            h[:, idx, idx] = np.asarray(self.onsite, dtype=np.complex128)
        for hop in self.hoppings:
            d = np.asarray(hop.d, dtype=float)
            phase = hop.t * np.exp(1j * (k @ d))  # (M,)
            h[:, hop.i, hop.j] += phase
            h[:, hop.j, hop.i] += np.conj(phase)  # keep H Hermitian
        return h

    def _chunk_size(self, target_mb: int) -> int:
        """k-points per chunk so one H block stays near `target_mb` (complex128)."""
        return max(1, target_mb * 1024 * 1024 // (self.n_orbitals * self.n_orbitals * 16))

    def bands(self, k: np.ndarray, target_mb: int = 512) -> np.ndarray:
        """Eigenvalues (ascending) for each k. Returns (M, N) real array (CPU).

        Chunked so the dense (chunk, N, N) Hamiltonian never exceeds ~target_mb."""
        k = np.atleast_2d(np.asarray(k, dtype=float))
        chunk = self._chunk_size(target_mb)
        out = [np.linalg.eigvalsh(self.hamiltonian_batch(k[s : s + chunk])) for s in range(0, k.shape[0], chunk)]
        return np.concatenate(out, axis=0)

    def bands_torch(self, k: np.ndarray, device: str = "cuda", complex_dtype=None, target_mb: int = 256) -> np.ndarray:
        """Same as `bands`, diagonalized with torch (GPU by default).

        Chunked to fit VRAM; falls back to complex64 if the device rejects
        complex128. Returns a NumPy (M, N) array, directly comparable to `bands`.
        Note: H is built on CPU and transferred per chunk, so this timing includes
        host→device transfer — only the eigensolve runs on the GPU."""
        import torch

        k = np.atleast_2d(np.asarray(k, dtype=float))
        if complex_dtype is None:
            complex_dtype = torch.complex128
        chunk = self._chunk_size(target_mb)
        out = []
        for s in range(0, k.shape[0], chunk):
            h = self.hamiltonian_batch(k[s : s + chunk])
            try:
                t = torch.from_numpy(h).to(device=device, dtype=complex_dtype)
                w = torch.linalg.eigvalsh(t)
            except Exception:
                t = torch.from_numpy(h).to(device=device, dtype=torch.complex64)
                w = torch.linalg.eigvalsh(t)
            out.append(w.detach().cpu().numpy())
            del t, w
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)


def graphene_model(a_cc: float = 1.42, t: float = 2.7) -> TightBinding:
    """Graphene as a general TightBinding instance (2 orbitals, 3 NN hoppings).

    Must reproduce GrapheneTB's closed-form bands — that's the validation."""
    deltas = a_cc * np.array(
        [[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0], [-0.5, -np.sqrt(3.0) / 2.0]]
    )
    hoppings = [Hopping(0, 1, (float(d[0]), float(d[1])), t) for d in deltas]
    return TightBinding(n_orbitals=2, hoppings=hoppings)


def random_model(n_orbitals: int, n_hoppings: int, seed: int = 0) -> TightBinding:
    """A synthetic N-orbital model for solver benchmarking ONLY (not a real
    material) — random Hermitian hoppings with random 2-D bond vectors."""
    rng = np.random.default_rng(seed)
    hoppings = []
    for _ in range(n_hoppings):
        i, j = rng.integers(0, n_orbitals, size=2)
        d = (float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3)))
        t = complex(rng.normal(), rng.normal())
        hoppings.append(Hopping(int(i), int(j), d, t))
    onsite = rng.normal(size=n_orbitals)
    return TightBinding(n_orbitals=n_orbitals, hoppings=hoppings, onsite=onsite)


def _benchmark() -> None:
    """Honest CPU-vs-GPU timing of the k-space diagonalization sweep."""
    import time

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    print("k-space diagonalization benchmark (CPU NumPy vs GPU torch)")
    print("  (chunked; GPU time includes CPU-built H transfer - only eigensolve is on GPU)")
    print(f"  CUDA available: {has_cuda}")
    n_k = 8_192
    rng = np.random.default_rng(0)
    for n in (2, 16, 64, 256):
        model = random_model(n_orbitals=n, n_hoppings=4 * n, seed=1)
        k = rng.uniform(-3, 3, size=(n_k, 2))

        t0 = time.perf_counter()
        model.bands(k)
        t_cpu = time.perf_counter() - t0

        line = f"  N={n:4d}, {n_k} k-points | CPU {t_cpu:7.3f}s"
        if has_cuda:
            model.bands_torch(k)  # warm-up (JIT/alloc)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.bands_torch(k)
            torch.cuda.synchronize()
            t_gpu = time.perf_counter() - t0
            line += f" | GPU {t_gpu:7.3f}s | speedup {t_cpu / t_gpu:5.2f}x"
        print(line)


if __name__ == "__main__":
    _benchmark()
