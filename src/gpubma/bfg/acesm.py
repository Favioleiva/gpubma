"""Anchored Cumulative Evidence Saturation Model (ACESM) for BMA denominator reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.special import logsumexp


@dataclass
class CumulativeEvidenceCurve:
    """Empirical cumulative evidence curve C_k^obs(d) for lattice k."""
    k: int
    U_obs: float
    Z_known_log: float
    Z_known_rel: float
    d_grid: np.ndarray
    C_rel_obs: np.ndarray
    C_log_obs: np.ndarray
    d_max_obs: float
    n_elites: int
    n_samples: int
    weights_sum: float
    boundary_slope: float


class CumulativeCurveBuilder:
    """Constructs Horvitz-Thompson inclusion-weighted cumulative evidence curves."""

    @staticmethod
    def build_empirical_curve(
        k: int,
        discovered_scores: Sequence[float],
        sampled_scores: Sequence[float],
        N_k: int,
        n_grid_points: int = 80,
        d_max_override: Optional[float] = None,
    ) -> CumulativeEvidenceCurve:
        r"""Construct C_k^obs(d) from discovered elite and probability sample scores.

        Parameters
        ----------
        k : int
            Lattice index.
        discovered_scores : Sequence[float]
            Canonical log scores of known elite / genealogical models D_k.
        sampled_scores : Sequence[float]
            Canonical log scores of probability sample models S_k from M_k \ D_k.
        N_k : int
            Total number of models in lattice k = comb(p, k).
        n_grid_points : int, default=80
            Number of evaluation points on the distance grid.
        d_max_override : Optional[float], default=None
            Cutoff distance threshold override for artificial truncation backtesting.
        """
        disc_list = list(discovered_scores)
        samp_arr = np.array(sampled_scores, dtype=np.float64)
        n_elites = len(disc_list)
        n_samples = len(samp_arr)

        all_scores = disc_list + samp_arr.tolist()
        if not all_scores:
            raise ValueError(f"No models provided for lattice k={k}.")

        U_obs = float(max(all_scores))

        # Known elite mass
        if n_elites > 0:
            Z_known_log = float(logsumexp(disc_list))
            elite_rel_mass = np.exp(np.array(disc_list, dtype=np.float64) - U_obs)
            elite_distances = U_obs - np.array(disc_list, dtype=np.float64)
            Z_known_rel = float(np.sum(elite_rel_mass))
        else:
            Z_known_log = float("-inf")
            Z_known_rel = 0.0
            elite_distances = np.array([], dtype=np.float64)
            elite_rel_mass = np.array([], dtype=np.float64)

        # Sampled non-elite mass
        if n_samples > 0:
            N_non_elite = max(N_k - n_elites, 1)
            sample_weight = float(N_non_elite) / float(n_samples)
            sample_distances = U_obs - samp_arr
            sample_rel_mass = np.exp(-sample_distances) * sample_weight
        else:
            sample_weight = 0.0
            sample_distances = np.array([], dtype=np.float64)
            sample_rel_mass = np.array([], dtype=np.float64)

        # Distance span
        all_dists = np.concatenate([elite_distances, sample_distances]) if n_samples > 0 else elite_distances
        d_max_observed = float(np.max(all_dists)) if len(all_dists) > 0 else 0.0
        d_max = min(float(d_max_override), d_max_observed) if d_max_override is not None else d_max_observed
        if not np.isfinite(d_max) or d_max <= 0.0:
            d_max = 1.0
        else:
            d_max = max(d_max, 1e-4)

        # Evaluation grid in relative distance space (d=0 to d_max)
        t = np.linspace(0.0, 1.0, n_grid_points, dtype=np.float64)
        d_grid = d_max * (t ** 1.3)
        d_grid[0] = 0.0

        C_rel_obs = np.zeros(n_grid_points, dtype=np.float64)
        for j, d_val in enumerate(d_grid):
            m_elite = np.sum(elite_rel_mass[elite_distances <= d_val]) if n_elites > 0 else 0.0
            m_sample = np.sum(sample_rel_mass[sample_distances <= d_val]) if n_samples > 0 else 0.0
            C_rel_obs[j] = m_elite + m_sample

        # Enforce monotonicity and anchor lower bound
        C_rel_obs = np.maximum.accumulate(C_rel_obs)
        C_rel_obs = np.maximum(C_rel_obs, Z_known_rel)
        C_log_obs = np.where(C_rel_obs > 0, np.log(C_rel_obs) + U_obs, float("-inf"))

        # Boundary slope at d_max
        if n_grid_points >= 4 and d_grid[-1] > d_grid[-4]:
            delta_d = d_grid[-1] - d_grid[-4]
            delta_C = C_rel_obs[-1] - C_rel_obs[-4]
            boundary_slope = max(float(delta_C / delta_d), 1e-12)
        else:
            boundary_slope = 1e-6

        return CumulativeEvidenceCurve(
            k=k,
            U_obs=U_obs,
            Z_known_log=Z_known_log,
            Z_known_rel=Z_known_rel,
            d_grid=d_grid,
            C_rel_obs=C_rel_obs,
            C_log_obs=C_log_obs,
            d_max_obs=d_max,
            n_elites=n_elites,
            n_samples=n_samples,
            weights_sum=sample_weight * n_samples if n_samples > 0 else 0.0,
            boundary_slope=boundary_slope,
        )


@dataclass
class ACESMFitResult:
    """Fitted ACESM result for a single lattice."""
    k: int
    log_Z_hat: float
    rel_plateau_hat: float
    Z_known_rel: float
    Z_known_log: float
    U_obs: float
    delta_Z_hat: float
    alpha_hat: float
    beta_hat: float
    opt_success: bool
    opt_message: str
    opt_nfev: int
    final_loss: float
    is_boundary_collapsed: bool


class AnchoredWeibullModel:
    """Anchored Weibull Saturation Model with free shape parameter beta."""

    def evaluate_cdf(self, d: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Evaluate Phi(d) = 1 - exp(-(alpha * d)^beta)."""
        log_alpha, log_beta = params[1], params[2]
        alpha = math.exp(log_alpha)
        beta = math.exp(log_beta)
        ad = np.maximum(alpha * d, 0.0)
        power = np.clip(ad ** beta, 0.0, 700.0)
        return 1.0 - np.exp(-power)

    def evaluate_derivative(self, d: float, params: np.ndarray, delta_Z: float) -> float:
        """Evaluate d/dd C_hat(d)."""
        if d <= 1e-12:
            return 0.0
        log_alpha, log_beta = params[1], params[2]
        alpha = math.exp(log_alpha)
        beta = math.exp(log_beta)
        ad = alpha * d
        power = min(ad ** beta, 700.0)
        deriv_phi = math.exp(-power) * beta * (ad ** (beta - 1.0)) * alpha
        return max(float(delta_Z * deriv_phi), 0.0)

    def get_initial_guesses(
        self,
        d_grid: np.ndarray,
        C_rel_obs: np.ndarray,
        Z_known_rel: float,
    ) -> List[np.ndarray]:
        observed_delta = max(C_rel_obs[-1] - Z_known_rel, 1e-4)
        d_max = max(d_grid[-1], 1.0)
        guesses = []
        for mult in [1.05, 1.5, 3.0, 10.0, 50.0]:
            log_delta = math.log(max(observed_delta * mult, 1e-6))
            for beta_val in [0.5, 1.0, 1.5, 2.5, 3.5]:
                log_alpha = math.log(max(1.0 / (d_max * 0.5), 1e-5))
                guesses.append(np.array([log_delta, log_alpha, math.log(beta_val)], dtype=np.float64))
        return guesses

    def get_bounds(self) -> Bounds:
        return Bounds(
            lb=[-25.0, -15.0, math.log(0.1)],
            ub=[35.0, 10.0, math.log(10.0)],
        )


class RestrictedShapeWeibullModel:
    """Anchored Weibull Saturation Model with locked shape parameter beta = beta_fixed."""

    def __init__(self, beta_fixed: float = 3.5):
        self.beta_fixed = float(beta_fixed)

    def evaluate_cdf(self, d: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Evaluate Phi(d) = 1 - exp(-(alpha * d)^beta)."""
        log_alpha = params[1]
        alpha = math.exp(log_alpha)
        ad = np.maximum(alpha * d, 0.0)
        power = np.clip(ad ** self.beta_fixed, 0.0, 700.0)
        return 1.0 - np.exp(-power)

    def evaluate_derivative(self, d: float, params: np.ndarray, delta_Z: float) -> float:
        """Evaluate d/dd C_hat(d)."""
        if d <= 1e-12:
            return 0.0
        log_alpha = params[1]
        alpha = math.exp(log_alpha)
        ad = alpha * d
        power = min(ad ** self.beta_fixed, 700.0)
        deriv_phi = math.exp(-power) * self.beta_fixed * (ad ** (self.beta_fixed - 1.0)) * alpha
        return max(float(delta_Z * deriv_phi), 0.0)

    def get_initial_guesses(
        self,
        d_grid: np.ndarray,
        C_rel_obs: np.ndarray,
        Z_known_rel: float,
    ) -> List[np.ndarray]:
        observed_delta = max(C_rel_obs[-1] - Z_known_rel, 1e-4)
        d_max = max(d_grid[-1], 1.0)
        guesses = []
        for mult in [1.05, 1.5, 3.0, 10.0, 50.0]:
            log_delta = math.log(max(observed_delta * mult, 1e-6))
            log_alpha = math.log(max(1.0 / (d_max * 0.5), 1e-5))
            guesses.append(np.array([log_delta, log_alpha], dtype=np.float64))
        return guesses

    def get_bounds(self) -> Bounds:
        return Bounds(lb=[-25.0, -15.0], ub=[35.0, 10.0])


class ACESMReconstructor:
    """Master fitting engine for ACESM denominator reconstruction."""

    @staticmethod
    def fit_lattice(
        curve: CumulativeEvidenceCurve,
        model: Optional[Any] = None,
        beta: Optional[float] = 3.5,
        lambda_momentum: float = 0.0,
        multi_start_n: int = 15,
    ) -> ACESMFitResult:
        """Fit Anchored Weibull Saturation model to the cumulative curve."""
        if model is None:
            if beta is not None and beta > 0:
                model = RestrictedShapeWeibullModel(beta_fixed=beta)
            else:
                model = AnchoredWeibullModel()
        d_grid = curve.d_grid
        C_rel_obs = curve.C_rel_obs
        Z_known_rel = curve.Z_known_rel
        U_obs = curve.U_obs
        boundary_slope = curve.boundary_slope

        eval_mask = d_grid > 0
        d_eval = d_grid[eval_mask]
        C_obs_eval = np.maximum(C_rel_obs[eval_mask], 1e-12)
        d_max = float(d_grid[-1])

        def objective(params: np.ndarray) -> float:
            log_delta_Z = params[0]
            delta_Z = math.exp(log_delta_Z)
            phi_vals = model.evaluate_cdf(d_eval, params)
            C_hat = Z_known_rel + delta_Z * phi_vals

            # Scale-invariant path loss: [log(1 + C_hat) - log(1 + C_obs)]^2
            diff = np.log1p(C_hat) - np.log1p(C_obs_eval)
            loss_path = float(np.mean(diff ** 2))

            loss_mom = 0.0
            if lambda_momentum > 0.0:
                g_hat = model.evaluate_derivative(d_max, params, delta_Z)
                mom_diff = math.log1p(g_hat) - math.log1p(boundary_slope)
                loss_mom = lambda_momentum * (mom_diff ** 2)

            return loss_path + loss_mom

        guesses = model.get_initial_guesses(d_grid, C_rel_obs, Z_known_rel)
        bounds = model.get_bounds()

        best_res = None
        best_loss = float("inf")
        best_params = None

        for p0 in guesses[:multi_start_n]:
            try:
                res = minimize(
                    objective,
                    p0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-7},
                )
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_res = res
                    best_params = res.x
            except Exception:
                continue

        if best_res is None or best_params is None:
            p0 = guesses[0]
            best_params = p0
            best_loss = objective(p0)
            opt_success = False
            opt_msg = "Optimization fallback to initial guess"
            nfev = 0
        else:
            opt_success = best_res.success
            opt_msg = str(best_res.message)
            nfev = int(best_res.nfev)

        log_delta_Z_hat = best_params[0]
        delta_Z_hat = math.exp(log_delta_Z_hat)
        alpha_hat = math.exp(best_params[1])
        rel_plateau_hat = Z_known_rel + delta_Z_hat
        log_Z_hat = U_obs + math.log(max(rel_plateau_hat, 1e-300))

        is_boundary_collapsed = bool(rel_plateau_hat <= 1.0005 * max(Z_known_rel, float(C_rel_obs[-1])))

        return ACESMFitResult(
            k=curve.k,
            log_Z_hat=log_Z_hat,
            rel_plateau_hat=rel_plateau_hat,
            Z_known_rel=Z_known_rel,
            Z_known_log=curve.Z_known_log,
            U_obs=U_obs,
            delta_Z_hat=delta_Z_hat,
            alpha_hat=alpha_hat,
            beta_hat=beta,
            opt_success=opt_success,
            opt_message=opt_msg,
            opt_nfev=nfev,
            final_loss=best_loss,
            is_boundary_collapsed=is_boundary_collapsed,
        )
