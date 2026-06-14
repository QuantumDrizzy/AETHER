"""
Material Behavior Predictor via Reservoir Computing
=====================================================
Wraps an Echo State Network to provide domain-specific material
behavior prediction, degradation forecasting, and nonlinearity
measurement.

Run standalone:
    python -m research.reservoir_computing.material_rc
"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.reservoir_computing.esn import EchoStateNetwork


class MaterialPredictor:
    """Predict material responses using an Echo State Network.

    Parameters
    ----------
    esn : EchoStateNetwork
        Pre-configured (but not necessarily trained) ESN instance.
    """

    def __init__(self, esn: EchoStateNetwork) -> None:
        self.esn = esn
        self._trained = False
        self._stimulus_mean: float = 0.0
        self._stimulus_std: float = 1.0
        self._response_mean: float = 0.0
        self._response_std: float = 1.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_measurements(
        self,
        measurements: list[dict[str, Any]],
        warmup: int = 50,
        ridge: float = 1e-6,
    ) -> None:
        """Train the ESN on a list of measurement records.

        Parameters
        ----------
        measurements : list[dict]
            Each dict has ``timestamp`` (float), ``stimulus`` (float or
            array), and ``response`` (float or array).
        warmup : int
            Warmup timesteps.
        ridge : float
            Ridge regularisation.
        """
        # Sort by timestamp
        measurements = sorted(measurements, key=lambda m: m["timestamp"])

        stimuli = np.array([m["stimulus"] for m in measurements], dtype=np.float64)
        responses = np.array([m["response"] for m in measurements], dtype=np.float64)

        # Normalise for better ESN performance
        self._stimulus_mean = float(np.mean(stimuli))
        self._stimulus_std = float(np.std(stimuli)) or 1.0
        self._response_mean = float(np.mean(responses))
        self._response_std = float(np.std(responses)) or 1.0

        X = ((stimuli - self._stimulus_mean) / self._stimulus_std).reshape(-1, 1)
        Y = ((responses - self._response_mean) / self._response_std).reshape(-1, 1)

        self.esn.fit(X, Y, warmup=warmup, ridge=ridge)
        self._trained = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_response(self, stimulus: np.ndarray) -> np.ndarray:
        """Predict material response to a given stimulus.

        Parameters
        ----------
        stimulus : np.ndarray, shape (T,) or (T, 1)
            Input stimulus signal.

        Returns
        -------
        np.ndarray, shape (T,)
            Predicted response (de-normalised).
        """
        if not self._trained:
            raise RuntimeError("Predictor not trained - call train_on_measurements() first")

        X = ((stimulus.flatten() - self._stimulus_mean) / self._stimulus_std).reshape(-1, 1)
        self.esn.reset_state()
        Y_pred_norm = self.esn.predict(X)
        return (Y_pred_norm.flatten() * self._response_std) + self._response_mean

    def predict_degradation(
        self,
        time_horizon: int,
        base_stimulus: np.ndarray | None = None,
        degradation_rate: float = 0.002,
    ) -> np.ndarray:
        """Predict material degradation over a time horizon.

        Models degradation as a gradual reduction in response amplitude
        with increasing noise, using the trained ESN as the base response
        model.

        Parameters
        ----------
        time_horizon : int
            Number of future timesteps to predict.
        base_stimulus : np.ndarray, optional
            Repeating stimulus pattern. If None, uses a default sinusoidal.
        degradation_rate : float
            Exponential decay rate of response quality.

        Returns
        -------
        np.ndarray, shape (time_horizon,)
            Predicted degradation envelope (response amplitude over time).
        """
        if base_stimulus is None:
            t = np.linspace(0, 4 * np.pi, min(time_horizon, 200))
            base_stimulus = np.sin(t)

        # Tile stimulus to cover horizon
        if len(base_stimulus) < time_horizon:
            repeats = time_horizon // len(base_stimulus) + 1
            base_stimulus = np.tile(base_stimulus, repeats)[:time_horizon]
        else:
            base_stimulus = base_stimulus[:time_horizon]

        # Predict base response
        if self._trained:
            response = self.predict_response(base_stimulus)
        else:
            response = base_stimulus.copy()  # identity if untrained

        # Apply degradation model
        time_axis = np.arange(time_horizon, dtype=np.float64)
        degradation_envelope = np.exp(-degradation_rate * time_axis)
        degraded = response * degradation_envelope

        # Extract amplitude envelope via windowed RMS
        window = min(50, time_horizon // 4) or 1
        amplitude = np.zeros(time_horizon)
        for i in range(time_horizon):
            start = max(0, i - window // 2)
            end = min(time_horizon, i + window // 2 + 1)
            amplitude[i] = np.sqrt(np.mean(degraded[start:end] ** 2))

        return amplitude

    # ------------------------------------------------------------------
    # Reservoir suitability analysis
    # ------------------------------------------------------------------

    def compute_nonlinearity(self, signal: np.ndarray) -> float:
        """Measure reservoir suitability via nonlinearity index.

        Quantifies how much the reservoir transforms the input beyond
        a simple linear mapping, which indicates the material/system's
        suitability for reservoir computing.

        Parameters
        ----------
        signal : np.ndarray, shape (T,)
            Input signal.

        Returns
        -------
        float
            Nonlinearity index in [0, 1]. Higher = more nonlinear.
        """
        X = signal.reshape(-1, 1)
        T = X.shape[0]

        # Collect reservoir states
        self.esn.reset_state()
        states = np.zeros((T, self.esn.reservoir_size))
        for t in range(T):
            states[t] = self.esn._update_state(X[t])

        # Fit linear model: states = A @ X_extended + bias
        X_ext = np.hstack([np.ones((T, 1)), X])
        linear_pred, _, _, _ = np.linalg.lstsq(X_ext, states, rcond=None)
        states_linear = X_ext @ linear_pred

        # Nonlinearity = fraction of variance NOT explained by linear model
        total_var = np.var(states, axis=0).sum()
        residual_var = np.var(states - states_linear, axis=0).sum()

        if total_var < 1e-12:
            return 0.0

        nonlinearity = residual_var / total_var
        return float(np.clip(nonlinearity, 0.0, 1.0))


# ======================================================================
# __main__ demo
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 65)
    print("  AETHER - Material Behavior Predictor Demo")
    print("=" * 65)

    # Create synthetic measurements (piezoelectric crystal response)
    n_measurements = 1500
    timestamps = np.linspace(0, 10.0, n_measurements)
    stimuli = (np.sin(2 * np.pi * timestamps)
               + 0.4 * np.sin(2 * np.pi * 3.3 * timestamps)
               + 0.2 * np.cos(2 * np.pi * 7.7 * timestamps))
    stimuli += np.random.randn(n_measurements) * 0.03

    # Nonlinear piezoelectric response
    responses = (0.7 * np.tanh(2.0 * stimuli)
                 + 0.2 * stimuli ** 2
                 - 0.05 * stimuli ** 3
                 + 0.15 * np.sin(3 * stimuli))
    responses += np.random.randn(n_measurements) * 0.01

    measurements = [
        {"timestamp": float(timestamps[i]),
         "stimulus": float(stimuli[i]),
         "response": float(responses[i])}
        for i in range(n_measurements)
    ]

    # Split into train/test
    split = int(0.7 * n_measurements)

    # Create ESN and predictor
    esn = EchoStateNetwork(
        input_dim=1,
        reservoir_size=150,
        output_dim=1,
        spectral_radius=0.95,
        leak_rate=0.3,
        seed=42,
    )
    predictor = MaterialPredictor(esn)

    print(f"\n  Training on {split} measurements...")
    predictor.train_on_measurements(measurements[:split], warmup=80)
    print("  ✓ Training complete")

    # Predict on test data
    test_stimuli = np.array([m["stimulus"] for m in measurements[split:]])
    test_responses = np.array([m["response"] for m in measurements[split:]])
    predicted = predictor.predict_response(test_stimuli)

    mse = np.mean((test_responses - predicted) ** 2)
    r2 = 1.0 - np.sum((test_responses - predicted) ** 2) / np.sum(
        (test_responses - np.mean(test_responses)) ** 2
    )
    print(f"\n  * Response Prediction:")
    print(f"    MSE  = {mse:.6f}")
    print(f"    R²   = {r2:.4f}")

    # Nonlinearity analysis
    test_signal = np.sin(np.linspace(0, 6 * np.pi, 500))
    nl = predictor.compute_nonlinearity(test_signal)
    print(f"\n  * Nonlinearity Index: {nl:.4f}")
    if nl > 0.5:
        print("    -> High nonlinearity - excellent reservoir computing substrate")
    elif nl > 0.2:
        print("    -> Moderate nonlinearity - suitable for RC with tuning")
    else:
        print("    -> Low nonlinearity - may need augmentation for RC")

    # Degradation prediction
    deg = predictor.predict_degradation(time_horizon=500, degradation_rate=0.003)
    print(f"\n  * Degradation Forecast (500-step horizon):")
    print(f"    Initial amplitude : {deg[0]:.4f}")
    print(f"    Final amplitude   : {deg[-1]:.4f}")
    print(f"    Decay ratio       : {deg[-1] / deg[0]:.4f}" if deg[0] > 0 else "    Decay ratio: N/A")
    print()
