"""Exact Gaussian processes for a flattened OD delay matrix.

One training simulation produces one matrix-valued response. Each OD entry is
an independent GP output *conditional on shared kernel hyperparameters*. No
PCA, latent scores, link-output GP, or scalar-output GP is fitted. Correlations
between outputs are not modelled; shared hyperparameters pool their evidence.

Hyperparameter fitting uses G = Y_normalized @ Y_normalized.T / n_outputs.
The resulting objective and gradient are EXACTLY the average negative log
marginal likelihood of the individual outputs, including all varying OD
entries. This is an algebraic efficiency improvement, not dimension reduction.
It avoids a training_samples x training_samples x OD_entries gradient array.

Storage is also exact: retain the original targets in their existing dtype
(float32 for cached FSM matrices), instead of a float64 coefficient per target
and simulation in the saved artifact. Repeated prediction lazily caches the
float64 coefficients K_train^-1(Y - mean) in memory, then evaluates
mean + K_query @ coefficients. Per-output normalization cancels algebraically.
This trades additional RAM for speed without PCA, quantization, or a changed
posterior. The derived cache is excluded from serialization.

Only training targets determine centring, scaling, and constant-output masks.
``output_scale_floor`` is in minutes: it prevents near-zero delay variation
from dominating through division by an arbitrarily tiny standard deviation.
The default 0.01 minute floor is an editable numerical choice, not calibration.

``return_std`` reports the untruncated marginal standard deviation of each
output; clipping a negative predicted mean does not make its uncertainty a
truncated-normal distribution. ``include_noise=False`` gives latent-function
uncertainty. True additionally includes learned WhiteKernel observation noise
(here unresolved numerical variation/model discrepancy), but not jitter.
Neither is a guarantee of accuracy; held-out coverage must be checked.

Inputs must already use the *fixed configured domain* to scale all
continuous features to [0, 1]. These bounds must not be estimated from test
data. Predictions outside the domain are rejected by default.

Public API: ODDelayGP().fit(X, stages, Y),
then .predict(X, stages, return_std=True). Stage IDs come from the case study.
Models are joblib/pickle serializable. .diagnostics_ contains JSON-safe values.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def _nll_and_gradient(theta, kernel, X, gram, jitter):
    """Average exact negative log likelihood and log-parameter gradient."""
    candidate = kernel.clone_with_theta(theta)
    K, dK = candidate(X, eval_gradient=True)
    K.flat[:: K.shape[0] + 1] += jitter
    try:
        L = cholesky(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e30, np.zeros_like(theta)
    K_inv = cho_solve((L, True), np.eye(len(X)), check_finite=False)
    objective = (
        0.5 * np.einsum("ij,ji->", K_inv, gram)
        + np.log(np.diag(L)).sum()
        + 0.5 * len(X) * np.log(2 * np.pi)
    )
    weight = K_inv - K_inv @ gram @ K_inv
    gradient = 0.5 * np.einsum("ij,ijk->k", weight, dK, optimize=True)
    return float(objective), gradient


class SharedKernelGP:
    """Exact multi-output GP with one fitted covariance kernel and no PCA."""

    def __init__(
        self, *, output_scale_floor=0.01, constant_tolerance=1e-12,
        n_restarts_optimizer=1, random_state=2026, max_optimizer_iterations=150,
        jitter=1e-8, clip_nonnegative=True, standardize_outputs=True,
        column_chunk_size=20000,
    ):
        self.output_scale_floor = float(output_scale_floor)
        self.constant_tolerance = float(constant_tolerance)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.random_state = int(random_state)
        self.max_optimizer_iterations = int(max_optimizer_iterations)
        self.jitter = float(jitter)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.standardize_outputs = bool(standardize_outputs)
        self.column_chunk_size = int(column_chunk_size)
        self.kernel_family = "matern52"

    def fit(self, X, Y):
        started = time.perf_counter()
        self.__dict__.pop("_prediction_alpha_", None)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y)
        if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
            raise ValueError("X and Y must be two-dimensional with matching rows.")
        if len(X) < 2 or Y.shape[1] < 1:
            raise ValueError("At least two simulations and one target are required.")
        if not np.isfinite(X).all():
            raise ValueError("GP training inputs/targets must all be finite.")
        if self.output_scale_floor <= 0 or self.column_chunk_size < 1:
            raise ValueError("The scale floor and column chunk size must be positive.")
        self.X_train_ = X.copy()
        self.n_outputs_ = Y.shape[1]
        self.target_mean_ = np.empty(self.n_outputs_, dtype=np.float64)
        target_std = np.empty(self.n_outputs_, dtype=np.float64)
        varying = np.empty(self.n_outputs_, dtype=bool)
        # Column blocks keep NumPy's float64 variance temporary bounded. Each
        # column still uses all training rows and the same reduction formula.
        for start in range(0, self.n_outputs_, self.column_chunk_size):
            selected = slice(start, start + self.column_chunk_size)
            block = Y[:, selected]
            if not np.isfinite(block).all():
                raise ValueError("GP training inputs/targets must all be finite.")
            self.target_mean_[selected] = block.mean(axis=0, dtype=np.float64)
            target_std[selected] = block.std(axis=0, dtype=np.float64)
            varying[selected] = np.ptp(block, axis=0) > self.constant_tolerance
        self.target_scale_ = (
            np.maximum(target_std, self.output_scale_floor)
            if self.standardize_outputs else np.ones(self.n_outputs_)
        )
        self.active_indices_ = np.flatnonzero(varying)
        count = len(self.active_indices_)
        self.training_targets_ = np.empty((len(X), count), dtype=Y.dtype)
        self.diagnostics_ = {
            "n_simulations": int(len(X)), "n_features": int(X.shape[1]),
            "n_outputs": int(self.n_outputs_), "n_varying_outputs": int(count),
            "n_constant_outputs": int(self.n_outputs_ - count),
            "output_scale_floor_minutes": self.output_scale_floor,
            "standardize_outputs": self.standardize_outputs,
            "kernel_family": self.kernel_family,
            "target_reduction": "none; exact Gram likelihood identity",
            "prediction_storage": "original training targets; exact posterior identity",
            "stored_target_dtype": str(self.training_targets_.dtype),
        }
        if not count:
            self.kernel_ = None
            self.L_ = None
            self.diagnostics_.update(fit_seconds=time.perf_counter() - started)
            return self
        gram = np.zeros((len(X), len(X)), dtype=np.float64)
        for start in range(0, count, self.column_chunk_size):
            columns = self.active_indices_[start:start + self.column_chunk_size]
            normalized = (Y[:, columns] - self.target_mean_[columns]) / self.target_scale_[columns]
            gram += normalized @ normalized.T
        gram /= count
        spatial_kernel = Matern(np.full(X.shape[1], 0.5), (0.03, 10.0), nu=2.5)
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * spatial_kernel
            + WhiteKernel(1e-5, (1e-8, 0.5))
        )
        rng = np.random.default_rng(self.random_state)
        starts = [kernel.theta]
        starts.extend(
            rng.uniform(kernel.bounds[:, 0], kernel.bounds[:, 1])
            for _ in range(self.n_restarts_optimizer)
        )
        attempts = []
        solutions = []
        for initial_theta in starts:
            result = minimize(
                _nll_and_gradient, initial_theta,
                args=(kernel, X, gram, self.jitter), method="L-BFGS-B",
                jac=True, bounds=kernel.bounds,
                options={"maxiter": self.max_optimizer_iterations, "ftol": 1e-10},
            )
            attempts.append({
                "success": bool(result.success), "message": str(result.message),
                "iterations": int(result.nit), "nll_per_output": float(result.fun),
            })
            if np.isfinite(result.fun):
                solutions.append(result)
        if not solutions:
            raise RuntimeError("Every GP hyperparameter optimization failed.")
        best = min(solutions, key=lambda item: item.fun)
        self.kernel_ = kernel.clone_with_theta(best.x)
        K = self.kernel_(X)
        K.flat[::len(X) + 1] += self.jitter
        self.L_ = cholesky(K, lower=True, check_finite=False)
        for start in range(0, count, self.column_chunk_size):
            columns = self.active_indices_[start:start + self.column_chunk_size]
            # Copy ownership is intentional: later caller mutation of Y must
            # not change a fitted model. No dtype conversion is performed.
            self.training_targets_[:, start:start + len(columns)] = Y[:, columns]
        theta = self.kernel_.theta
        bounds = self.kernel_.bounds
        self.diagnostics_.update({
            "kernel": str(self.kernel_), "theta": theta.tolist(),
            "length_scales": np.asarray(self.kernel_.k1.k2.length_scale).tolist(),
            "amplitude_variance": float(self.kernel_.k1.k1.constant_value),
            "noise_variance_normalized": float(self.kernel_.k2.noise_level),
            "nll_per_varying_output": float(best.fun),
            "selected_optimizer_success": bool(best.success),
            "optimizer_attempts": attempts,
            "parameters_near_bound": np.flatnonzero(
                np.minimum(theta - bounds[:, 0], bounds[:, 1] - theta) < 0.01
            ).tolist(),
            "fit_seconds": time.perf_counter() - started,
        })
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_prediction_alpha_", None)
        return state

    def _prediction_coefficients(self):
        """Exact float64 posterior coefficients, also for older saved models."""
        cached = getattr(self, "_prediction_alpha_", None)
        if cached is None:
            cached = np.empty(self.training_targets_.shape, dtype=np.float64)
            for start in range(0, len(self.active_indices_), self.column_chunk_size):
                columns = self.active_indices_[start:start + self.column_chunk_size]
                stop = start + len(columns)
                centered = self.training_targets_[:, start:stop] - self.target_mean_[columns]
                cached[:, start:stop] = cho_solve((self.L_, True), centered, check_finite=False)
            self._prediction_alpha_ = cached
        return cached

    def predict(self, X, *, return_std=False, include_noise=False, clip_nonnegative=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train_.shape[1] or not np.isfinite(X).all():
            raise ValueError("Prediction features must be finite and match training dimensions.")
        mean = np.broadcast_to(self.target_mean_, (len(X), self.n_outputs_)).copy()
        std = np.zeros_like(mean) if return_std else None
        if self.kernel_ is not None:
            cross = self.kernel_(X, self.X_train_)
            mean[:, self.active_indices_] += cross @ self._prediction_coefficients()
            if return_std:
                solved = solve_triangular(self.L_, cross.T, lower=True, check_finite=False)
                variance = self.kernel_.k1.diag(X) - np.einsum("ij,ij->j", solved, solved)
                if include_noise:
                    variance += self.kernel_.k2.noise_level
                latent_std = np.sqrt(np.maximum(variance, 0.0))
                std[:, self.active_indices_] = (
                    latent_std[:, None] * self.target_scale_[self.active_indices_]
                )
        should_clip = self.clip_nonnegative if clip_nonnegative is None else clip_nonnegative
        if should_clip:
            np.maximum(mean, 0.0, out=mean)
        return (mean, std) if return_std else mean


class ODDelayGP:
    """One pooled GP with one-hot infrastructure configurations."""

    def __init__(self, *, stages=(0, 1, 2, 3), allow_extrapolation=False, **gp_options):
        self.mode = "pooled"
        self.stages = tuple(stages)
        self.allow_extrapolation = bool(allow_extrapolation)
        self.gp_options = gp_options

    def _inputs(self, X, stages):
        X = np.asarray(X, dtype=np.float64)
        stage = np.asarray(stages)
        if stage.ndim == 0:
            stage = np.repeat(stage, len(X))
        expected = getattr(self, "n_features_in_", 4)
        if X.ndim != 2 or X.shape[1] != expected or stage.shape != (len(X),):
            raise ValueError(f"Expected X with {expected} physical inputs and one stage per row.")
        if not np.isfinite(X).all() or not np.isin(stage, self.stages).all():
            raise ValueError("Nonfinite input or an unrecognized stage.")
        if not self.allow_extrapolation and ((X < -1e-10) | (X > 1 + 1e-10)).any():
            raise ValueError("Scaled continuous inputs lie outside the configured [0, 1] domain.")
        return X, stage

    def fit(self, X, stages, Y):
        if np.ndim(X) != 2:
            raise ValueError("Training inputs must be a matrix.")
        self.n_features_in_ = np.shape(X)[1]
        X, stages = self._inputs(X, stages)
        Y = np.asarray(Y)
        if Y.ndim != 2 or Y.shape[0] != len(X):
            raise ValueError("Y must contain one flattened OD matrix per simulation.")
        counts = {int(stage): int(np.sum(stages == stage)) for stage in self.stages}
        if min(counts.values()) < 2:
            raise ValueError("The experiment requires at least two training rows in each stage.")
        self.n_outputs_ = Y.shape[1]
        encoded = np.column_stack([X, *[(stages == value).astype(float) for value in self.stages]])
        self.models_ = {"pooled": SharedKernelGP(**self.gp_options).fit(encoded, Y)}
        self.diagnostics_ = {
            "mode": self.mode, "stage_counts": counts,
            "feature_encoding": f"{self.n_features_in_} continuous + {len(self.stages)} one-hot",
            "models": {str(key): model.diagnostics_ for key, model in self.models_.items()},
            "fit_seconds": sum(model.diagnostics_["fit_seconds"] for model in self.models_.values()),
        }
        return self

    def predict(self, X, stages, *, return_std=False, include_noise=False, clip_nonnegative=None):
        X, stages = self._inputs(X, stages)
        kwargs = dict(return_std=return_std, include_noise=include_noise, clip_nonnegative=clip_nonnegative)
        encoded = np.column_stack([X, *[(stages == value).astype(float) for value in self.stages]])
        return self.models_["pooled"].predict(encoded, **kwargs)
