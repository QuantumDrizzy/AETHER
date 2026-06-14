"""
Echo State Network - Pure NumPy Implementation
================================================
Reservoir computing with leaky-integrator ESN for time-series
prediction of material behaviour.

Run standalone:
    python -m research.reservoir_computing.esn
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as la


class EchoStateNetwork:
    """Pure-NumPy Echo State Network with ridge-regression readout.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input signal.
    reservoir_size : int
        Number of reservoir neurons.
    output_dim : int
        Dimensionality of output.
    spectral_radius : float
        Desired spectral radius of the reservoir weight matrix.
    leak_rate : float
        Leaky-integrator leak rate α ∈ (0, 1].
    input_scaling : float
        Scaling factor for input weights.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int = 1,
        reservoir_size: int = 200,
        output_dim: int = 1,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        input_scaling: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.seed = seed

        self.Win: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.Wfb: np.ndarray | None = None
        self.Wout: np.ndarray | None = None
        self.state: np.ndarray | None = None

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise input, reservoir, and feedback weight matrices."""
        rng = np.random.default_rng(self.seed)
        N = self.reservoir_size

        # Input weights: (N, input_dim+1) - +1 for bias
        self.Win = (rng.uniform(-1, 1, (N, self.input_dim + 1))
                    * self.input_scaling)

        # Sparse reservoir matrix: ~10% connectivity
        density = min(10.0 / N, 1.0)
        W_raw = rng.uniform(-0.5, 0.5, (N, N))
        mask = rng.random((N, N)) < density
        W_raw *= mask

        # Scale to desired spectral radius
        eigenvalues = np.abs(la.eigvals(W_raw))
        rho = np.max(eigenvalues) if len(eigenvalues) > 0 else 1.0
        if rho > 0:
            self.W = W_raw * (self.spectral_radius / rho)
        else:
            self.W = W_raw

        # Feedback weights (for teacher-forced training)
        self.Wfb = rng.uniform(-0.5, 0.5, (N, self.output_dim)) * 0.1

        # Initial reservoir state
        self.state = np.zeros(N, dtype=np.float64)

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def _update_state(self, x: np.ndarray) -> np.ndarray:
        """Leaky-integrator reservoir state update.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Current input vector.

        Returns
        -------
        np.ndarray, shape (reservoir_size,)
            Updated reservoir state.
        """
        assert self.Win is not None and self.W is not None and self.state is not None

        # Prepend bias
        x_aug = np.concatenate(([1.0], x))

        pre_activation = self.Win @ x_aug + self.W @ self.state
        new_state = (1.0 - self.leak_rate) * self.state + self.leak_rate * np.tanh(pre_activation)
        self.state = new_state
        return new_state

    # ------------------------------------------------------------------
    # Training (ridge regression)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        warmup: int = 50,
        ridge: float = 1e-6,
    ) -> "EchoStateNetwork":
        """Train the readout weights using ridge regression.

        Parameters
        ----------
        X : np.ndarray, shape (T, input_dim)
            Input time series.
        Y : np.ndarray, shape (T, output_dim)
            Target time series.
        warmup : int
            Number of initial steps to discard (let the reservoir settle).
        ridge : float
            Regularisation parameter (Tikhonov / ridge).

        Returns
        -------
        EchoStateNetwork
            Self, for chaining.
        """
        T = X.shape[0]
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Reset state
        self.state = np.zeros(self.reservoir_size, dtype=np.float64)

        # Collect reservoir states
        states = np.zeros((T, self.reservoir_size), dtype=np.float64)
        for t in range(T):
            states[t] = self._update_state(X[t])

        # Discard warmup
        states_train = states[warmup:]
        Y_train = Y[warmup:]

        # Augment with bias
        ones = np.ones((states_train.shape[0], 1), dtype=np.float64)
        S = np.hstack([ones, states_train])

        # Ridge regression: Wout = (S^T S + ridge*I)^{-1} S^T Y
        A = S.T @ S + ridge * np.eye(S.shape[1])
        B = S.T @ Y_train
        self.Wout = la.solve(A, B)  # (reservoir_size+1, output_dim)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input sequence X.

        Parameters
        ----------
        X : np.ndarray, shape (T, input_dim)
            Input time series.

        Returns
        -------
        np.ndarray, shape (T, output_dim)
            Predicted output.
        """
        if self.Wout is None:
            raise RuntimeError("ESN has not been trained yet - call .fit() first")

        T = X.shape[0]
        self.state = np.zeros(self.reservoir_size, dtype=np.float64)

        predictions = np.zeros((T, self.Wout.shape[1]), dtype=np.float64)
        for t in range(T):
            s = self._update_state(X[t])
            s_aug = np.concatenate(([1.0], s))
            predictions[t] = s_aug @ self.Wout

        return predictions

    def reset_state(self) -> None:
        """Reset the reservoir state to zeros."""
        self.state = np.zeros(self.reservoir_size, dtype=np.float64)


# ======================================================================
# __main__ demo
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 65)
    print("  AETHER - Echo State Network Demo")
    print("  Synthetic piezoelectric signal prediction")
    print("=" * 65)

    # Generate synthetic piezoelectric signal
    T_total = 2000
    t = np.linspace(0, 10 * np.pi, T_total)

    # Input: mechanical stimulus (superposition of frequencies)
    stimulus = (np.sin(t) + 0.5 * np.sin(3.7 * t) + 0.3 * np.cos(7.1 * t))
    stimulus += np.random.randn(T_total) * 0.05  # measurement noise

    # Output: piezoelectric response (nonlinear transformation)
    response = (0.8 * np.tanh(1.5 * stimulus)
                + 0.3 * stimulus ** 2
                + 0.1 * np.sin(2 * stimulus))
    response += np.random.randn(T_total) * 0.02

    X = stimulus.reshape(-1, 1)
    Y = response.reshape(-1, 1)

    # Train/test split
    split = int(0.7 * T_total)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Create and train ESN
    esn = EchoStateNetwork(
        input_dim=1,
        reservoir_size=200,
        output_dim=1,
        spectral_radius=0.95,
        leak_rate=0.3,
        input_scaling=0.5,
        seed=42,
    )

    print(f"\n  Reservoir size : {esn.reservoir_size}")
    print(f"  Spectral radius: {esn.spectral_radius}")
    print(f"  Leak rate      : {esn.leak_rate}")
    print(f"  Training points: {split}")
    print(f"  Test points    : {T_total - split}")

    esn.fit(X_train, Y_train, warmup=100, ridge=1e-6)
    print("  Training complete!")

    # Predict on test set
    esn.reset_state()
    # Run through training data to warm up state
    _ = esn.predict(X_train)
    Y_pred = esn.predict(X_test)

    # Metrics
    mse = np.mean((Y_test - Y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((Y_test - Y_pred) ** 2)
    ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    nrmse = rmse / (np.max(Y_test) - np.min(Y_test)) if (np.max(Y_test) - np.min(Y_test)) > 0 else rmse

    print(f"\n  * Test Results:")
    print(f"    MSE   = {mse:.6f}")
    print(f"    RMSE  = {rmse:.6f}")
    print(f"    NRMSE = {nrmse:.6f}")
    print(f"    R²    = {r2:.6f}")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(Y_test[:200], "b-", alpha=0.7, label="Actual")
        axes[0].plot(Y_pred[:200], "r--", alpha=0.7, label="Predicted")
        axes[0].set_ylabel("Piezo Response")
        axes[0].legend()
        axes[0].set_title("ESN Prediction - Piezoelectric Response")

        axes[1].plot(Y_test[:200] - Y_pred[:200], "g-", alpha=0.7)
        axes[1].set_ylabel("Error")
        axes[1].set_xlabel("Time step")
        axes[1].set_title("Prediction Error")

        plt.tight_layout()
        plt.savefig("esn_demo_output.png", dpi=150)
        print("\n  Plot saved to esn_demo_output.png")
    except ImportError:
        print("\n  (matplotlib not available - skipping plot)")

    print()
