import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import BaseController
from .pid import Controller as PIDController

try:
  from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH, DEL_T, STEER_RANGE
except Exception:
  CONTROL_START_IDX = 100
  CONTEXT_LENGTH = 20
  DEL_T = 0.1
  STEER_RANGE = (-2.0, 2.0)


def _sigmoid(x: float) -> float:
  z = float(np.clip(x, -60.0, 60.0))
  return float(1.0 / (1.0 + np.exp(-z)))


def _lerp(a: float, b: float, t: float) -> float:
  return float(a + (b - a) * t)


class Controller(BaseController):
  _echo_done = False

  DEFAULT_CONFIG = {
    # Core behavior
    "base_mix": 0.12,
    "ff_only_mode": 0,
    "simple_hybrid_mode": 0,
    "known_good_fixed_mix_mode": 1,
    "aggressive_horizon_mode": 0,
    "analytic_preview_mode": 0,
    "bc_preview_mode": 0,
    "bc_preview_use_mlp": 0,
    "bc_preview_pid_mix": 0.20,
    "bc_preview_err_gain": 0.08,
    "bc_preview_dy_gain": 0.03,
    "bc_preview_du_cap": 0.30,
    "bc_preview_smoothing": 0.10,
    "analytic_preview_horizon": 12,
    "analytic_preview_decay": 0.82,
    "analytic_preview_rate_weight": 2.0,
    "analytic_preview_du_cap": 0.28,
    "analytic_preview_margin": 0.0,
    "preview_pid_mode": 0,
    "preview_pid_horizon": 8,
    "preview_pid_decay": 0.72,
    "preview_pid_ff_weight": 0.95,
    "preview_pid_p": 0.10,
    "preview_pid_i": 0.015,
    "preview_pid_d": 0.040,
    "preview_pid_dy": 0.030,
    "preview_pid_integral_decay": 0.992,
    "preview_pid_integral_limit": 0.90,
    "preview_pid_integral_zone": 0.70,
    "preview_pid_smoothing": 0.12,
    "aggressive_horizon": 18,
    "aggressive_prefix_steps": 8,
    "aggressive_base_mix": 0.0,
    "aggressive_mix_gain": 8.0,
    "aggressive_mix_max": 1.0,
    "aggressive_improve_floor": 0.0,
    "aggressive_track_weight": 8.0,
    "aggressive_dy_weight": 0.8,
    "aggressive_ddy_weight": 0.06,
    "aggressive_du_weight": 0.04,
    "aggressive_delta_weight": 0.12,
    "aggressive_barrier_weight": 0.03,
    "aggressive_terminal_weight": 4.0,
    "aggressive_du_max_step": 0.22,
    "aggressive_delta_u_max_abs": 0.60,
    "aggressive_max_delta_vs_pid": 0.80,
    "aggressive_polish_amp": 0.06,
    "aggressive_clip_fallback_enable": 0,
    "aggressive_sign_align_enable": 1,
    "simple_veto_mode": 0,
    "fixed_mix_tail_veto_enable": 0,
    "fixed_mix_emergency_fallback_enable": 0,
    "fixed_mix_clip_fallback_enable": 1,
    "fixed_mix_delta_scale_enable": 1,
    "fixed_mix_delta_scale_ref": 0.04,
    "fixed_mix_delta_scale_min": 0.20,
    "fixed_mix_oracle_candidate_mix": 0.50,
    "fixed_mix_nn_mix_enable": 0,
    "fixed_mix_nn_mix_floor": 0.05,
    "fixed_mix_nn_gate_threshold": 0.0,
    "fixed_mix_post_residual_enable": 0,
    "fixed_mix_post_residual_scale": 1.0,
    "fixed_mix_post_residual_clip": 0.10,
    "fixed_mix_candidate_mode": "simple_lqr_raw",
    "fixed_mix_bc_candidate_mix": 0.35,
    "strict_safe_mode": 0,
    "accept_enable": 0,  # default off: direct adaptive blend generally performs better
    "legacy_fixed_mix_mode": 0,  # strict legacy path: raw PID + MPC candidate + linear blend only
    "legacy_repro_mode": 0,  # isolated direct-y fixed-mix branch for regression reproduction
    "legacy_pid_exact_mode": 1,  # in legacy mode, base_mix=0 must reduce to exact repo PID behavior
    "regime_enable": 0,
    "tail_mode_enable": 0,


    # Simple hybrid mode
    "simple_horizon": 10,
    "simple_mix_base": 0.18,
    "simple_mix_min": 0.18,
    "simple_mix_max": 0.18,
    "simple_mix_err_gain": 0.015,
    "simple_mix_dy_gain": 0.008,
    "simple_track_weight": 1.05,
    "simple_dy_weight": 1.35,
    "simple_ddy_weight": 0.28,
    "simple_du_weight": 0.22,
    "simple_delta_weight": 1.9,
    "simple_barrier_weight": 0.10,
    "simple_terminal_weight": 1.5,
    "simple_max_pred_delta": 0.16,
    "safe_err_gate_1": 0.12,
    "safe_err_gate_2": 0.25,
    "safe_dy_gate_1": 0.25,
    "safe_dy_gate_2": 0.50,
    "safe_mix_1": 0.03,
    "safe_mix_2": 0.01,
    "safe_pred_delta_cap": 0.015,
    "safe_calm_delta_cap": 0.015,
    "safe_mild_delta_cap": 0.005,
    "safe_err_growth_veto": 0.03,
    "safe_dy_growth_veto": 0.10,
    "safe_target_flip_slope": 0.03,
    "safe_rate_limit_margin": 0.85,
    "safe_local_adv_margin": 0.0005,
    "safe_local_err_margin": 0.002,
    "safe_recovery_trend_margin": 0.02,
    "local_adv_min_calm": 0.00000025,
    "local_adv_min_mild": 0.00000090,
    "accept_budget_window": 90,
    "accept_budget_max_fraction": 0.025,
    "marginal_accept_suppression_after_n": 1,
    "veto_err_rise_steps": 3,
    "veto_dy_rise_steps": 3,
    "veto_sign_streak": 4,
    "veto_flip_window": 6,
    "veto_flip_count_th": 3,
    "veto_min_mix_delta": 0.01,
    "veto_hold_steps": 1,
    "veto_dy_thresh": 0.75,
    "veto_err_thresh": 0.35,
    "veto_delta_thresh": 0.12,
    "veto_sign_err_thresh": 0.18,
    "fixed_mix_emergency_dy_thresh": 1.5,
    "fixed_mix_emergency_err_thresh": 0.45,
    "fixed_mix_emergency_err_rise_steps": 2,
    "fixed_mix_emergency_dy_confirm": 0.8,
    "fixed_mix_emergency_hold_steps": 8,

    # Legacy-compatible scalar defaults
    "horizon": 14,
    "tracking_weight": 1.1,
    "dy_weight": 1.4,
    "ddy_weight": 0.22,
    "du_weight": 0.20,
    "delta_weight": 2.2,
    "barrier_weight": 0.12,
    "terminal_weight": 1.6,

    # Regime-aware horizon scheduling
    "horizon_calm": 10,
    "horizon_mid": 14,
    "horizon_aggressive": 18,

    # Regime-aware weight scheduling (calm -> aggressive interpolation)
    "track_weight_calm": 0.95,
    "track_weight_mid": 1.15,
    "track_weight_aggressive": 1.60,
    "dy_weight_calm": 1.10,
    "dy_weight_mid": 1.45,
    "dy_weight_aggressive": 2.20,
    "ddy_weight_calm": 0.10,
    "ddy_weight_mid": 0.22,
    "ddy_weight_aggressive": 0.46,
    "du_weight_calm": 0.14,
    "du_weight_mid": 0.22,
    "du_weight_aggressive": 0.34,
    "delta_weight_calm": 1.45,
    "delta_weight_mid": 2.20,
    "delta_weight_aggressive": 3.20,
    "barrier_weight_calm": 0.08,
    "barrier_weight_mid": 0.12,
    "barrier_weight_aggressive": 0.30,
    "terminal_weight_aggressive": 2.10,

    # Adaptive mix scheduling
    "mix_min": 0.00,
    "mix_max": 0.20,
    "mix_err_gain": 0.10,
    "mix_dy_gain": 0.06,
    "mix_jerk_gain": 0.02,
    "mix_step_gain": 0.01,
    "mix_smoothing": 0.14,
    "mix_hysteresis": 0.01,

    # Regime thresholds
    "regime_err_thresh": 0.55,
    "regime_dy_thresh": 0.85,
    "regime_step_thresh": 2.0,
    "regime_sat_thresh": 0.72,
    "risk_alpha": 0.84,
    "risk_hysteresis": 0.06,
    "mismatch_thresh": 0.25,
    "mismatch_gain": 6.0,

    # Tail protection / anti-catastrophe
    "tail_err_thresh": 1.35,
    "tail_dy_thresh": 1.8,
    "tail_growth_thresh": 0.20,
    "tail_hold_steps": 6,
    "tail_risk_gain": 1.25,
    "recovery_mix_bonus": 0.06,
    "recovery_du_scale": 1.35,
    "recovery_track_weight_scale": 1.25,
    "integral_freeze_in_tail": 0,
    "oscillation_damping_scale": 1.35,

    # Hard constraints
    "du_max_step": 0.17,
    "delta_u_max_abs": 0.18,
    "max_mpc_delta_vs_pid": 0.24,
    "soft_pid_du_cap": 2.0,

    # Signal conditioning
    "target_lpf_alpha": 0.30,
    "target_median_enable": 1,
    "target_step_thresh": 0.90,
    "curvature_window": 3,
    "dy_filter_tau": 0.20,

    # Observer (dy channel)
    "fast_gain": 0.16,
    "fast_leak": 0.45,
    "fast_clip": 0.35,
    "slow_gain": 0.025,
    "slow_leak": 0.992,
    "slow_clip": 0.90,

    # Residual correction layer (bounded, deterministic)
    "residual_enable": 0,
    "residual_gain": 1.0,
    "residual_clip": 0.035,
    "residual_lr": 0.003,
    "residual_decay": 0.995,
    "residual_weight_clip": 0.25,
    "residual_disable_tail": 1,

    # Acceptance logic (optional)
    "accept_margin": 0.98,
    "accept_err_margin": 0.005,
    "accept_dy_slack": 0.04,
    "accept_hold_steps": 3,
    "accept_cost_ratio": 1.0005,
    "accept_min_advantage": -0.00020,
    "accept_max_delta_normal": 0.090,
    "accept_max_delta_relaxed": 0.140,
    "accept_prefix_steps": 3,
    "accept_min_mix": 0.0015,
    "accept_cost_ratio_calm": 0.996,
    "accept_cost_ratio_mid": 1.0005,
    "accept_cost_ratio_aggressive": 1.004,
    "accept_min_advantage_calm": 0.0004,
    "accept_min_advantage_mid": -0.0004,
    "accept_min_advantage_aggressive": -0.0012,
    "trust_ramp_up": 0.12,
    "trust_ramp_down": 0.18,
    "trust_ramp_max": 0.45,
    "trust_ramp_mix_bonus": 0.80,
    "trust_ramp_cost_bonus": 0.0050,
    "consistency_alpha": 0.86,
    "consistency_ratio_bonus": 0.0035,
    "consistency_adv_bonus": 0.0008,
    "consistency_calm_gate": 0.10,
    "consistency_small_delta": 0.018,
    "consistency_transition_gate": 0.14,
    "transition_mix_bonus": 0.18,
    "soft_score_thresh_calm": 0.54,
    "soft_score_thresh_mid": 0.34,
    "soft_score_thresh_aggressive": 0.24,
    "multi_candidate_enable": 1,
    "candidate_score_track_w": 1.00,
    "candidate_score_jerk_w": 0.55,
    "candidate_score_rate_w": 0.35,
    "candidate_score_tail_w": 1.40,
    "candidate_score_osc_w": 0.30,
    "candidate_score_sat_w": 0.22,
    "candidate_score_consistency_w": 0.28,
    "candidate_filter_jerk_scale": 1.80,
    "candidate_filter_err_scale": 1.60,
    "candidate_filter_dy_abs": 2.50,
    "tail_reject_err": 1.25,
    "tail_reject_dy": 1.60,
    "tail_reject_sat_margin": 0.94,
    "recovery_hold_steps": 4,
    "mpc_suppress_after_tail": 1,
    "sign_check_enable": 1,
    "magnitude_check_enable": 1,

    # Smooth confidence blending when accept_enable=0
    "blend_min_adv": 0.0002,
    "blend_adv_scale": 0.006,
    "blend_dy_guard": 16.0,
    "blend_risk_guard": 0.60,
    "blend_tail_cap": 0.12,
    "trust_quality_scale": 0.12,
    "trust_family_gain": 0.30,
    "trust_consistency_gain": 0.22,
    "trust_tail_scale": 0.90,
    "trust_floor_exploit": 0.22,
    "trust_strict_threshold": 0.60,
    "family_quality_alpha": 0.92,
    "relative_score_threshold": 0.004,
    "relative_safe_band": 0.0010,
    "tail_penalty_weight": 1.40,

    # Local deterministic polish
    "polish_amp": 0.035,

    # Optional FF shortcut path
    "ff_k0": 0.0,
    "ff_k1": 0.0,
    "ff_mix": 0.0,
    "ff_tau": 0.35,

    # Fallback model if model_params is missing
    "fallback_A": 0.62,
    "fallback_B0": 0.34,
    "fallback_Bv": 0.0007,
    "fallback_C": 0.20,
    "fallback_D": 0.0,

    # Diagnostics model (legacy)
    "diag_a": 0.78,
    "diag_b0": 0.33,
    "diag_bv": 0.0007,
    "diag_c": 0.23,
    "diag_d": -0.002,
  }

  def __init__(self):
    raw_cfg = os.getenv("TOP1_MPC_CONFIG")
    cfg = dict(self.DEFAULT_CONFIG)
    if raw_cfg:
      try:
        loaded = json.loads(raw_cfg)
        if isinstance(loaded, dict):
          cfg.update(loaded)
      except json.JSONDecodeError:
        pass

    self.cfg = cfg
    self.dt = float(DEL_T)
    self.u_min = float(STEER_RANGE[0])
    self.u_max = float(STEER_RANGE[1])
    self.control_delay_steps = max(0, int(CONTROL_START_IDX) - int(CONTEXT_LENGTH))

    self.diag_mode = os.getenv("TOP1_MPC_DIAG") == "1"
    self.diag_max_steps = 14
    self.echo_mode = os.getenv("TOP1_MPC_ECHO") == "1"
    self.sign_assert_mode = os.getenv("TOP1_MPC_ASSERT_SIGN") == "1"

    # Core control switches
    self.base_mix = float(np.clip(cfg.get("base_mix", cfg.get("mpc_mix", 0.20)), 0.0, 1.0))
    self.ff_only_mode = bool(int(cfg.get("ff_only_mode", 0)))
    self.simple_hybrid_mode = bool(int(cfg.get("simple_hybrid_mode", 1)))
    self.known_good_fixed_mix_mode = bool(int(cfg.get("known_good_fixed_mix_mode", 0)))
    self.aggressive_horizon_mode = bool(int(cfg.get("aggressive_horizon_mode", 0)))
    self.analytic_preview_mode = bool(int(cfg.get("analytic_preview_mode", 0)))
    self.bc_preview_mode = bool(int(cfg.get("bc_preview_mode", 0)))
    self.bc_preview_use_mlp = bool(int(cfg.get("bc_preview_use_mlp", 0)))
    self.bc_preview_pid_mix = float(np.clip(cfg.get("bc_preview_pid_mix", 0.20), 0.0, 1.0))
    self.bc_preview_err_gain = float(np.clip(cfg.get("bc_preview_err_gain", 0.08), -2.0, 2.0))
    self.bc_preview_dy_gain = float(np.clip(cfg.get("bc_preview_dy_gain", 0.03), -2.0, 2.0))
    self.bc_preview_du_cap = float(np.clip(cfg.get("bc_preview_du_cap", 0.30), 0.01, 1.0))
    self.bc_preview_smoothing = float(np.clip(cfg.get("bc_preview_smoothing", 0.10), 0.0, 0.99))
    self.analytic_preview_horizon = int(np.clip(cfg.get("analytic_preview_horizon", 12), 2, 20))
    self.analytic_preview_decay = float(np.clip(cfg.get("analytic_preview_decay", 0.82), 0.0, 0.99))
    self.analytic_preview_rate_weight = float(np.clip(cfg.get("analytic_preview_rate_weight", 2.0), 0.0, 100.0))
    self.analytic_preview_du_cap = float(np.clip(cfg.get("analytic_preview_du_cap", 0.28), 0.01, 1.0))
    self.analytic_preview_margin = float(np.clip(cfg.get("analytic_preview_margin", 0.0), -1.0, 1.0))
    self.preview_pid_mode = bool(int(cfg.get("preview_pid_mode", 0)))
    self.preview_pid_horizon = int(np.clip(cfg.get("preview_pid_horizon", 8), 1, 16))
    self.preview_pid_decay = float(np.clip(cfg.get("preview_pid_decay", 0.72), 0.0, 0.99))
    self.preview_pid_ff_weight = float(np.clip(cfg.get("preview_pid_ff_weight", 0.95), 0.0, 3.0))
    self.preview_pid_p = float(np.clip(cfg.get("preview_pid_p", 0.10), -2.0, 2.0))
    self.preview_pid_i = float(np.clip(cfg.get("preview_pid_i", 0.015), -1.0, 1.0))
    self.preview_pid_d = float(np.clip(cfg.get("preview_pid_d", 0.040), -2.0, 2.0))
    self.preview_pid_dy = float(np.clip(cfg.get("preview_pid_dy", 0.030), -2.0, 2.0))
    self.preview_pid_integral_decay = float(np.clip(cfg.get("preview_pid_integral_decay", 0.992), 0.0, 1.0))
    self.preview_pid_integral_limit = float(np.clip(cfg.get("preview_pid_integral_limit", 0.90), 0.0, 10.0))
    self.preview_pid_integral_zone = float(np.clip(cfg.get("preview_pid_integral_zone", 0.70), 0.0, 5.0))
    self.preview_pid_smoothing = float(np.clip(cfg.get("preview_pid_smoothing", 0.12), 0.0, 0.99))
    self.aggressive_horizon = int(np.clip(cfg.get("aggressive_horizon", 18), 6, 30))
    self.aggressive_prefix_steps = int(np.clip(cfg.get("aggressive_prefix_steps", 8), 2, self.aggressive_horizon))
    self.aggressive_base_mix = float(np.clip(cfg.get("aggressive_base_mix", 0.0), 0.0, 1.0))
    self.aggressive_mix_gain = float(np.clip(cfg.get("aggressive_mix_gain", 8.0), 0.0, 100.0))
    self.aggressive_mix_max = float(np.clip(cfg.get("aggressive_mix_max", 1.0), 0.0, 1.0))
    self.aggressive_improve_floor = float(np.clip(cfg.get("aggressive_improve_floor", 0.0), -1.0, 1.0))
    self.aggressive_track_weight = float(max(1e-6, cfg.get("aggressive_track_weight", 8.0)))
    self.aggressive_dy_weight = float(max(1e-6, cfg.get("aggressive_dy_weight", 0.8)))
    self.aggressive_ddy_weight = float(max(0.0, cfg.get("aggressive_ddy_weight", 0.06)))
    self.aggressive_du_weight = float(max(1e-6, cfg.get("aggressive_du_weight", 0.04)))
    self.aggressive_delta_weight = float(max(1e-6, cfg.get("aggressive_delta_weight", 0.12)))
    self.aggressive_barrier_weight = float(max(0.0, cfg.get("aggressive_barrier_weight", 0.03)))
    self.aggressive_terminal_weight = float(max(0.1, cfg.get("aggressive_terminal_weight", 4.0)))
    self.aggressive_du_max_step = float(np.clip(cfg.get("aggressive_du_max_step", 0.22), 0.01, 1.0))
    self.aggressive_delta_u_max_abs = float(np.clip(cfg.get("aggressive_delta_u_max_abs", 0.60), 0.01, 2.0))
    self.aggressive_max_delta_vs_pid = float(np.clip(cfg.get("aggressive_max_delta_vs_pid", 0.80), 0.0, 2.0))
    self.aggressive_polish_amp = float(np.clip(cfg.get("aggressive_polish_amp", 0.06), 0.0, 0.5))
    self.aggressive_clip_fallback_enable = bool(int(cfg.get("aggressive_clip_fallback_enable", 0)))
    self.aggressive_sign_align_enable = bool(int(cfg.get("aggressive_sign_align_enable", 1)))
    self.simple_veto_mode = bool(int(cfg.get("simple_veto_mode", 0)))
    self.fixed_mix_tail_veto_enable = bool(int(cfg.get("fixed_mix_tail_veto_enable", 0)))
    self.fixed_mix_emergency_fallback_enable = bool(int(cfg.get("fixed_mix_emergency_fallback_enable", 0)))
    self.fixed_mix_clip_fallback_enable = bool(int(cfg.get("fixed_mix_clip_fallback_enable", 1)))
    self.fixed_mix_delta_scale_enable = bool(int(cfg.get("fixed_mix_delta_scale_enable", 0)))
    self.fixed_mix_delta_scale_ref = float(max(1e-6, cfg.get("fixed_mix_delta_scale_ref", 0.06)))
    self.fixed_mix_delta_scale_min = float(np.clip(cfg.get("fixed_mix_delta_scale_min", 0.35), 0.0, 1.0))
    self.fixed_mix_candidate_mode = str(cfg.get("fixed_mix_candidate_mode", "legacy")).strip().lower()
    if self.fixed_mix_candidate_mode not in ("legacy", "simple_lqr", "simple_lqr_raw", "bc_preview", "legacy_bc", "nn_oracle", "legacy_oracle", "bc_oracle"):
      self.fixed_mix_candidate_mode = "legacy"
    self.fixed_mix_bc_candidate_mix = float(np.clip(cfg.get("fixed_mix_bc_candidate_mix", 0.35), 0.0, 1.0))
    self.fixed_mix_oracle_candidate_mix = float(np.clip(cfg.get("fixed_mix_oracle_candidate_mix", 0.50), 0.0, 1.0))
    self.fixed_mix_nn_mix_enable = bool(int(cfg.get("fixed_mix_nn_mix_enable", 0)))
    self.fixed_mix_nn_mix_floor = float(np.clip(cfg.get("fixed_mix_nn_mix_floor", 0.05), 0.0, 1.0))
    self.fixed_mix_nn_gate_threshold = float(np.clip(cfg.get("fixed_mix_nn_gate_threshold", 0.0), 0.0, 1.0))
    self.fixed_mix_post_residual_enable = bool(int(cfg.get("fixed_mix_post_residual_enable", 0)))
    self.fixed_mix_post_residual_scale = float(np.clip(cfg.get("fixed_mix_post_residual_scale", 1.0), 0.0, 3.0))
    self.fixed_mix_post_residual_clip = float(np.clip(cfg.get("fixed_mix_post_residual_clip", 0.10), 0.0, 0.5))
    self.strict_safe_mode = bool(int(cfg.get("strict_safe_mode", 0)))
    self.accept_enable = bool(int(cfg.get("accept_enable", 0)))
    self.legacy_fixed_mix_mode = bool(int(cfg.get("legacy_fixed_mix_mode", 0)))
    self.legacy_repro_mode = bool(int(cfg.get("legacy_repro_mode", 0)))
    self.legacy_pid_exact_mode = bool(int(cfg.get("legacy_pid_exact_mode", 1)))
    self.regime_enable = bool(int(cfg.get("regime_enable", 1)))
    self.tail_mode_enable = bool(int(cfg.get("tail_mode_enable", 1)))

    self.simple_horizon = int(np.clip(cfg.get("simple_horizon", 10), 6, 20))
    self.simple_mix_base = float(np.clip(cfg.get("simple_mix_base", self.base_mix), 0.0, 1.0))
    self.simple_mix_min = float(np.clip(cfg.get("simple_mix_min", self.simple_mix_base), 0.0, 1.0))
    self.simple_mix_max = float(np.clip(cfg.get("simple_mix_max", self.simple_mix_base), self.simple_mix_min, 1.0))
    self.simple_mix_err_gain = float(np.clip(cfg.get("simple_mix_err_gain", 0.015), 0.0, 1.0))
    self.simple_mix_dy_gain = float(np.clip(cfg.get("simple_mix_dy_gain", 0.008), 0.0, 1.0))
    self.simple_track_weight = float(max(1e-6, cfg.get("simple_track_weight", 1.05)))
    self.simple_dy_weight = float(max(1e-6, cfg.get("simple_dy_weight", 1.35)))
    self.simple_ddy_weight = float(max(0.0, cfg.get("simple_ddy_weight", 0.28)))
    self.simple_du_weight = float(max(1e-6, cfg.get("simple_du_weight", 0.22)))
    self.simple_delta_weight = float(max(1e-6, cfg.get("simple_delta_weight", 1.9)))
    self.simple_barrier_weight = float(max(0.0, cfg.get("simple_barrier_weight", 0.10)))
    self.simple_terminal_weight = float(max(0.1, cfg.get("simple_terminal_weight", 1.5)))
    self.simple_max_pred_delta = float(np.clip(cfg.get("simple_max_pred_delta", 0.16), 0.0, 1.0))
    self.safe_err_gate_1 = float(max(0.0, cfg.get("safe_err_gate_1", 0.12)))
    self.safe_err_gate_2 = float(max(self.safe_err_gate_1, cfg.get("safe_err_gate_2", 0.25)))
    self.safe_dy_gate_1 = float(max(0.0, cfg.get("safe_dy_gate_1", 0.25)))
    self.safe_dy_gate_2 = float(max(self.safe_dy_gate_1, cfg.get("safe_dy_gate_2", 0.50)))
    self.safe_mix_1 = float(np.clip(cfg.get("safe_mix_1", 0.03), 0.0, 1.0))
    self.safe_mix_2 = float(np.clip(cfg.get("safe_mix_2", 0.01), 0.0, 1.0))
    self.safe_pred_delta_cap = float(np.clip(cfg.get("safe_pred_delta_cap", 0.015), 0.0, 1.0))
    self.safe_calm_delta_cap = float(np.clip(cfg.get("safe_calm_delta_cap", 0.015), 0.0, 1.0))
    self.safe_mild_delta_cap = float(np.clip(cfg.get("safe_mild_delta_cap", 0.005), 0.0, 1.0))
    self.safe_err_growth_veto = float(max(0.0, cfg.get("safe_err_growth_veto", 0.03)))
    self.safe_dy_growth_veto = float(max(0.0, cfg.get("safe_dy_growth_veto", 0.10)))
    self.safe_target_flip_slope = float(max(0.0, cfg.get("safe_target_flip_slope", 0.03)))
    self.safe_rate_limit_margin = float(np.clip(cfg.get("safe_rate_limit_margin", 0.85), 0.0, 1.5))
    self.safe_local_adv_margin = float(max(0.0, cfg.get("safe_local_adv_margin", 0.0005)))
    self.safe_local_err_margin = float(max(0.0, cfg.get("safe_local_err_margin", 0.002)))
    self.safe_recovery_trend_margin = float(max(0.0, cfg.get("safe_recovery_trend_margin", 0.02)))
    self.local_adv_min_calm = float(max(0.0, cfg.get("local_adv_min_calm", 0.00000025)))
    self.local_adv_min_mild = float(max(self.local_adv_min_calm, cfg.get("local_adv_min_mild", 0.00000090)))
    self.accept_budget_window = int(np.clip(cfg.get("accept_budget_window", 90), 1, 500))
    self.accept_budget_max_fraction = float(np.clip(cfg.get("accept_budget_max_fraction", 0.025), 0.0, 1.0))
    self.marginal_accept_suppression_after_n = int(np.clip(cfg.get("marginal_accept_suppression_after_n", 1), 0, 50))
    self.veto_err_rise_steps = int(np.clip(cfg.get("veto_err_rise_steps", 3), 1, 12))
    self.veto_dy_rise_steps = int(np.clip(cfg.get("veto_dy_rise_steps", 3), 1, 12))
    self.veto_sign_streak = int(np.clip(cfg.get("veto_sign_streak", 4), 1, 12))
    self.veto_flip_window = int(np.clip(cfg.get("veto_flip_window", 6), 2, 20))
    self.veto_flip_count_th = int(np.clip(cfg.get("veto_flip_count_th", 3), 1, 12))
    self.veto_min_mix_delta = float(np.clip(cfg.get("veto_min_mix_delta", 0.01), 0.0, 1.0))
    self.veto_hold_steps = int(np.clip(cfg.get("veto_hold_steps", 1), 0, 20))
    self.veto_proxy_harm_steps = max(2, min(3, self.veto_err_rise_steps))
    self.veto_dy_thresh = float(max(0.0, cfg.get("veto_dy_thresh", 0.75)))
    self.veto_err_thresh = float(max(0.0, cfg.get("veto_err_thresh", 0.35)))
    self.veto_delta_thresh = float(max(0.0, cfg.get("veto_delta_thresh", 0.12)))
    self.veto_sign_err_thresh = float(max(0.0, cfg.get("veto_sign_err_thresh", 0.18)))
    self.fixed_mix_emergency_dy_thresh = float(max(0.0, cfg.get("fixed_mix_emergency_dy_thresh", 1.5)))
    self.fixed_mix_emergency_err_thresh = float(max(0.0, cfg.get("fixed_mix_emergency_err_thresh", 0.45)))
    self.fixed_mix_emergency_err_rise_steps = int(np.clip(cfg.get("fixed_mix_emergency_err_rise_steps", 2), 1, 10))
    self.fixed_mix_emergency_dy_confirm = float(max(0.0, cfg.get("fixed_mix_emergency_dy_confirm", 0.8)))
    self.fixed_mix_emergency_hold_steps = int(np.clip(cfg.get("fixed_mix_emergency_hold_steps", 0), 0, 50))

    # Legacy/base weights (kept for compatibility)
    self.horizon = int(np.clip(cfg.get("horizon", 14), 6, 24))
    self.tracking_weight = float(max(1e-6, cfg.get("tracking_weight", 1.1)))
    self.dy_weight = float(max(1e-6, cfg.get("dy_weight", 1.4)))
    self.ddy_weight = float(max(0.0, cfg.get("ddy_weight", 0.22)))
    self.du_weight = float(max(1e-6, cfg.get("du_weight", 0.2)))
    self.delta_weight = float(max(1e-6, cfg.get("delta_weight", 2.2)))
    self.barrier_weight = float(max(0.0, cfg.get("barrier_weight", 0.12)))
    self.terminal_weight = float(max(0.1, cfg.get("terminal_weight", 1.6)))

    # Horizon schedule
    self.horizon_calm = int(np.clip(cfg.get("horizon_calm", self.horizon), 6, 24))
    self.horizon_mid = int(np.clip(cfg.get("horizon_mid", self.horizon), self.horizon_calm, 26))
    self.horizon_aggressive = int(np.clip(cfg.get("horizon_aggressive", max(self.horizon_mid, self.horizon + 2)), self.horizon_mid, 30))

    # Weight schedule
    self.track_weight_calm = float(max(1e-6, cfg.get("track_weight_calm", self.tracking_weight)))
    self.track_weight_mid = float(max(1e-6, cfg.get("track_weight_mid", self.tracking_weight)))
    self.track_weight_aggressive = float(max(1e-6, cfg.get("track_weight_aggressive", max(self.tracking_weight, 1.6))))

    self.dy_weight_calm = float(max(1e-6, cfg.get("dy_weight_calm", self.dy_weight)))
    self.dy_weight_mid = float(max(1e-6, cfg.get("dy_weight_mid", self.dy_weight)))
    self.dy_weight_aggressive = float(max(1e-6, cfg.get("dy_weight_aggressive", max(self.dy_weight, 2.0))))

    self.ddy_weight_calm = float(max(0.0, cfg.get("ddy_weight_calm", self.ddy_weight)))
    self.ddy_weight_mid = float(max(0.0, cfg.get("ddy_weight_mid", self.ddy_weight)))
    self.ddy_weight_aggressive = float(max(0.0, cfg.get("ddy_weight_aggressive", max(self.ddy_weight, 0.35))))

    self.du_weight_calm = float(max(1e-6, cfg.get("du_weight_calm", self.du_weight)))
    self.du_weight_mid = float(max(1e-6, cfg.get("du_weight_mid", self.du_weight)))
    self.du_weight_aggressive = float(max(1e-6, cfg.get("du_weight_aggressive", max(self.du_weight, 0.30))))

    self.delta_weight_calm = float(max(1e-6, cfg.get("delta_weight_calm", self.delta_weight)))
    self.delta_weight_mid = float(max(1e-6, cfg.get("delta_weight_mid", self.delta_weight)))
    self.delta_weight_aggressive = float(max(1e-6, cfg.get("delta_weight_aggressive", max(self.delta_weight, 3.0))))

    self.barrier_weight_calm = float(max(0.0, cfg.get("barrier_weight_calm", self.barrier_weight)))
    self.barrier_weight_mid = float(max(0.0, cfg.get("barrier_weight_mid", self.barrier_weight)))
    self.barrier_weight_aggressive = float(max(0.0, cfg.get("barrier_weight_aggressive", max(self.barrier_weight, 0.3))))

    self.terminal_weight_aggressive = float(max(0.1, cfg.get("terminal_weight_aggressive", max(self.terminal_weight, 2.0))))

    # Adaptive mix
    self.mix_min = float(np.clip(cfg.get("mix_min", 0.06), 0.0, 1.0))
    self.mix_max = float(np.clip(cfg.get("mix_max", 0.34), self.mix_min, 1.0))
    self.mix_err_gain = float(np.clip(cfg.get("mix_err_gain", 0.06), 0.0, 2.0))
    self.mix_dy_gain = float(np.clip(cfg.get("mix_dy_gain", 0.04), 0.0, 2.0))
    self.mix_jerk_gain = float(np.clip(cfg.get("mix_jerk_gain", 0.02), 0.0, 2.0))
    self.mix_step_gain = float(np.clip(cfg.get("mix_step_gain", 0.02), 0.0, 2.0))
    self.mix_smoothing = float(np.clip(cfg.get("mix_smoothing", 0.18), 0.0, 1.0))
    self.mix_hysteresis = float(np.clip(cfg.get("mix_hysteresis", 0.01), 0.0, 0.2))

    # Regime/risk
    self.regime_err_thresh = float(max(1e-6, cfg.get("regime_err_thresh", 0.55)))
    self.regime_dy_thresh = float(max(1e-6, cfg.get("regime_dy_thresh", 0.85)))
    self.regime_step_thresh = float(max(1e-6, cfg.get("regime_step_thresh", 2.0)))
    self.regime_sat_thresh = float(max(1e-6, cfg.get("regime_sat_thresh", 0.72)))
    self.risk_alpha = float(np.clip(cfg.get("risk_alpha", 0.84), 0.0, 0.999))
    self.risk_hysteresis = float(np.clip(cfg.get("risk_hysteresis", 0.06), 0.0, 0.5))
    self.mismatch_thresh = float(max(1e-6, cfg.get("mismatch_thresh", 0.45)))
    self.mismatch_gain = float(np.clip(cfg.get("mismatch_gain", 2.5), 0.0, 20.0))

    # Tail mode
    self.tail_err_thresh = float(max(1e-6, cfg.get("tail_err_thresh", 1.35)))
    self.tail_dy_thresh = float(max(1e-6, cfg.get("tail_dy_thresh", 1.8)))
    self.tail_growth_thresh = float(max(0.0, cfg.get("tail_growth_thresh", 0.20)))
    self.tail_hold_steps = int(np.clip(cfg.get("tail_hold_steps", 6), 0, 50))
    self.tail_risk_gain = float(np.clip(cfg.get("tail_risk_gain", 1.25), 0.0, 5.0))
    self.recovery_mix_bonus = float(np.clip(cfg.get("recovery_mix_bonus", 0.06), 0.0, 0.5))
    self.recovery_du_scale = float(np.clip(cfg.get("recovery_du_scale", 1.35), 0.5, 3.0))
    self.recovery_track_weight_scale = float(np.clip(cfg.get("recovery_track_weight_scale", 1.25), 0.5, 4.0))
    self.integral_freeze_in_tail = bool(int(cfg.get("integral_freeze_in_tail", 0)))
    self.oscillation_damping_scale = float(np.clip(cfg.get("oscillation_damping_scale", 1.35), 0.5, 4.0))

    # Constraints
    self.du_max_step = float(np.clip(cfg.get("du_max_step", 0.17), 0.01, 0.8))
    self.delta_u_max_abs = float(np.clip(cfg.get("delta_u_max_abs", 0.18), 0.005, 1.0))
    self.max_mpc_delta_vs_pid = float(np.clip(cfg.get("max_mpc_delta_vs_pid", 0.24), 0.0, 1.0))
    self.soft_pid_du_cap = float(np.clip(cfg.get("soft_pid_du_cap", 2.0), 0.02, 3.0))

    # Signal conditioning
    self.target_lpf_alpha = float(np.clip(cfg.get("target_lpf_alpha", 0.30), 0.0, 1.0))
    self.target_median_enable = bool(int(cfg.get("target_median_enable", 1)))
    self.target_step_thresh = float(max(1e-6, cfg.get("target_step_thresh", 0.90)))
    self.curvature_window = int(np.clip(cfg.get("curvature_window", 3), 2, 6))

    # Observer and dy filter
    self.dy_filter_tau = float(np.clip(cfg.get("dy_filter_tau", 0.20), 1e-3, 3.0))
    self.fast_gain = float(np.clip(cfg.get("fast_gain", 0.16), 0.0, 1.0))
    self.fast_leak = float(np.clip(cfg.get("fast_leak", 0.45), 0.0, 1.0))
    self.fast_clip = float(np.clip(cfg.get("fast_clip", 0.35), 0.01, 10.0))
    self.slow_gain = float(np.clip(cfg.get("slow_gain", 0.025), 0.0, 0.5))
    self.slow_leak = float(np.clip(cfg.get("slow_leak", 0.992), 0.8, 1.0))
    self.slow_clip = float(np.clip(cfg.get("slow_clip", 0.90), 0.05, 10.0))

    # Residual
    self.residual_enable = bool(int(cfg.get("residual_enable", 1)))
    self.residual_gain = float(np.clip(cfg.get("residual_gain", 1.0), 0.0, 5.0))
    self.residual_clip = float(np.clip(cfg.get("residual_clip", 0.035), 0.0, 0.3))
    self.residual_lr = float(np.clip(cfg.get("residual_lr", 0.003), 0.0, 0.1))
    self.residual_decay = float(np.clip(cfg.get("residual_decay", 0.995), 0.8, 1.0))
    self.residual_weight_clip = float(np.clip(cfg.get("residual_weight_clip", 0.25), 0.01, 2.0))
    self.residual_disable_tail = bool(int(cfg.get("residual_disable_tail", 1)))

    # Acceptance
    self.accept_margin = float(np.clip(cfg.get("accept_margin", 0.98), 0.5, 1.2))
    self.accept_err_margin = float(np.clip(cfg.get("accept_err_margin", 0.005), 0.0, 0.4))
    self.accept_dy_slack = float(np.clip(cfg.get("accept_dy_slack", 0.04), 0.0, 2.0))
    self.accept_hold_steps = int(np.clip(cfg.get("accept_hold_steps", 3), 0, 20))
    self.accept_cost_ratio = float(np.clip(cfg.get("accept_cost_ratio", 1.0005), 0.70, 1.05))
    self.accept_min_advantage = float(np.clip(cfg.get("accept_min_advantage", -0.00025), -0.05, 0.05))
    self.accept_max_delta_normal = float(np.clip(cfg.get("accept_max_delta_normal", 0.090), 0.0, 1.0))
    self.accept_max_delta_relaxed = float(np.clip(cfg.get("accept_max_delta_relaxed", 0.140), self.accept_max_delta_normal, 1.0))
    self.accept_prefix_steps = int(np.clip(cfg.get("accept_prefix_steps", 3), 1, 6))
    self.accept_min_mix = float(np.clip(cfg.get("accept_min_mix", 0.0015), 0.0, 1.0))
    self.accept_cost_ratio_calm = float(np.clip(cfg.get("accept_cost_ratio_calm", 0.996), 0.70, 1.05))
    self.accept_cost_ratio_mid = float(np.clip(cfg.get("accept_cost_ratio_mid", 1.0005), 0.70, 1.05))
    self.accept_cost_ratio_aggressive = float(np.clip(cfg.get("accept_cost_ratio_aggressive", 1.004), 0.70, 1.05))
    self.accept_min_advantage_calm = float(np.clip(cfg.get("accept_min_advantage_calm", 0.0004), -0.05, 0.05))
    self.accept_min_advantage_mid = float(np.clip(cfg.get("accept_min_advantage_mid", -0.0004), -0.05, 0.05))
    self.accept_min_advantage_aggressive = float(np.clip(cfg.get("accept_min_advantage_aggressive", -0.0012), -0.05, 0.05))
    self.trust_ramp_up = float(np.clip(cfg.get("trust_ramp_up", 0.12), 0.0, 1.0))
    self.trust_ramp_down = float(np.clip(cfg.get("trust_ramp_down", 0.18), 0.0, 1.0))
    self.trust_ramp_max = float(np.clip(cfg.get("trust_ramp_max", 0.45), 0.0, 1.0))
    self.trust_ramp_mix_bonus = float(np.clip(cfg.get("trust_ramp_mix_bonus", 0.80), 0.0, 3.0))
    self.trust_ramp_cost_bonus = float(np.clip(cfg.get("trust_ramp_cost_bonus", 0.0050), 0.0, 0.05))
    self.consistency_alpha = float(np.clip(cfg.get("consistency_alpha", 0.86), 0.0, 0.99))
    self.consistency_ratio_bonus = float(np.clip(cfg.get("consistency_ratio_bonus", 0.0035), 0.0, 0.05))
    self.consistency_adv_bonus = float(np.clip(cfg.get("consistency_adv_bonus", 0.0008), 0.0, 0.05))
    self.consistency_calm_gate = float(np.clip(cfg.get("consistency_calm_gate", 0.10), 0.0, 1.0))
    self.consistency_small_delta = float(np.clip(cfg.get("consistency_small_delta", 0.018), 0.0, 1.0))
    self.consistency_transition_gate = float(np.clip(cfg.get("consistency_transition_gate", 0.14), 0.0, 1.0))
    self.transition_mix_bonus = float(np.clip(cfg.get("transition_mix_bonus", 0.18), 0.0, 1.0))
    self.soft_score_thresh_calm = float(np.clip(cfg.get("soft_score_thresh_calm", 0.54), -1.0, 2.0))
    self.soft_score_thresh_mid = float(np.clip(cfg.get("soft_score_thresh_mid", 0.34), -1.0, 2.0))
    self.soft_score_thresh_aggressive = float(np.clip(cfg.get("soft_score_thresh_aggressive", 0.24), -1.0, 2.0))
    self.multi_candidate_enable = bool(int(cfg.get("multi_candidate_enable", 1)))
    self.candidate_score_track_w = float(np.clip(cfg.get("candidate_score_track_w", 1.00), 0.0, 10.0))
    self.candidate_score_jerk_w = float(np.clip(cfg.get("candidate_score_jerk_w", 0.55), 0.0, 10.0))
    self.candidate_score_rate_w = float(np.clip(cfg.get("candidate_score_rate_w", 0.35), 0.0, 10.0))
    self.candidate_score_tail_w = float(np.clip(cfg.get("candidate_score_tail_w", 1.40), 0.0, 10.0))
    self.candidate_score_osc_w = float(np.clip(cfg.get("candidate_score_osc_w", 0.30), 0.0, 10.0))
    self.candidate_score_sat_w = float(np.clip(cfg.get("candidate_score_sat_w", 0.22), 0.0, 10.0))
    self.candidate_score_consistency_w = float(np.clip(cfg.get("candidate_score_consistency_w", 0.28), 0.0, 10.0))
    self.candidate_filter_jerk_scale = float(np.clip(cfg.get("candidate_filter_jerk_scale", 1.80), 0.5, 10.0))
    self.candidate_filter_err_scale = float(np.clip(cfg.get("candidate_filter_err_scale", 1.60), 0.5, 10.0))
    self.candidate_filter_dy_abs = float(np.clip(cfg.get("candidate_filter_dy_abs", 2.50), 0.1, 10.0))
    self.tail_reject_err = float(max(0.0, cfg.get("tail_reject_err", 1.25)))
    self.tail_reject_dy = float(max(0.0, cfg.get("tail_reject_dy", 1.60)))
    self.tail_reject_sat_margin = float(np.clip(cfg.get("tail_reject_sat_margin", 0.94), 0.0, 1.5))
    self.recovery_hold_steps = int(np.clip(cfg.get("recovery_hold_steps", 4), 0, 50))
    self.mpc_suppress_after_tail = bool(int(cfg.get("mpc_suppress_after_tail", 1)))
    self.sign_check_enable = bool(int(cfg.get("sign_check_enable", 1)))
    self.magnitude_check_enable = bool(int(cfg.get("magnitude_check_enable", 1)))
    self.blend_min_adv = float(np.clip(cfg.get("blend_min_adv", 0.0002), -0.2, 0.2))
    self.blend_adv_scale = float(np.clip(cfg.get("blend_adv_scale", 0.006), 1e-5, 0.5))
    self.blend_dy_guard = float(np.clip(cfg.get("blend_dy_guard", 10.0), 0.0, 100.0))
    self.blend_risk_guard = float(np.clip(cfg.get("blend_risk_guard", 0.40), 0.0, 1.0))
    self.blend_tail_cap = float(np.clip(cfg.get("blend_tail_cap", 0.22), 0.0, 1.0))
    self.trust_quality_scale = float(np.clip(cfg.get("trust_quality_scale", 0.12), 1e-4, 2.0))
    self.trust_family_gain = float(np.clip(cfg.get("trust_family_gain", 0.30), 0.0, 2.0))
    self.trust_consistency_gain = float(np.clip(cfg.get("trust_consistency_gain", 0.22), 0.0, 2.0))
    self.trust_tail_scale = float(np.clip(cfg.get("trust_tail_scale", 0.90), 0.0, 5.0))
    self.trust_floor_exploit = float(np.clip(cfg.get("trust_floor_exploit", 0.22), 0.0, 1.0))
    self.trust_strict_threshold = float(np.clip(cfg.get("trust_strict_threshold", 0.60), 0.0, 1.0))
    self.family_quality_alpha = float(np.clip(cfg.get("family_quality_alpha", 0.92), 0.0, 0.999))
    self.relative_score_threshold = float(np.clip(cfg.get("relative_score_threshold", 0.004), 0.0, 1.0))
    self.relative_safe_band = float(np.clip(cfg.get("relative_safe_band", 0.0010), 0.0, 1.0))
    self.tail_penalty_weight = float(np.clip(cfg.get("tail_penalty_weight", 1.40), 0.0, 10.0))

    # Optional FF-only fast path
    self.ff_k0 = float(cfg.get("ff_k0", 0.0))
    self.ff_k1 = float(cfg.get("ff_k1", 0.0))
    self.ff_mix = float(np.clip(cfg.get("ff_mix", 0.0), 0.0, 0.2))
    self.ff_tau = float(np.clip(cfg.get("ff_tau", 0.35), 1e-3, 5.0))

    # Local polish
    self.polish_amp = float(np.clip(cfg.get("polish_amp", 0.035), 0.0, 0.2))

    # Fallback model
    self.fallback_A = float(np.clip(cfg.get("fallback_A", 0.62), -0.5, 1.5))
    self.fallback_B0 = float(np.clip(cfg.get("fallback_B0", 0.34), -5.0, 5.0))
    self.fallback_Bv = float(np.clip(cfg.get("fallback_Bv", 0.0007), -0.2, 0.2))
    self.fallback_C = float(np.clip(cfg.get("fallback_C", 0.20), -5.0, 5.0))
    self.fallback_D = float(np.clip(cfg.get("fallback_D", 0.0), -5.0, 5.0))

    self.diag_a = float(cfg.get("diag_a", 0.78))
    self.diag_b0 = float(cfg.get("diag_b0", 0.33))
    self.diag_bv = float(cfg.get("diag_bv", 0.0007))
    self.diag_c = float(cfg.get("diag_c", 0.23))
    self.diag_d = float(cfg.get("diag_d", -0.002))

    self.pid_controller = PIDController()

    self.v_edges, self.roll_edges, self.y_coeffs, self.dy_coeffs, self.model_from_file = self._load_piecewise_model()
    self.bc_preview_model = self._load_bc_preview_model()
    self.fixed_mix_nn_model = self._load_fixed_mix_nn_model()
    self.oracle_nn_model = self._load_oracle_nn_model()
    self.fixed_mix_post_residual_model = self._load_fixed_mix_post_residual_model()

    # Runtime state
    self.call_idx = 0
    self.prev_target = None
    self.prev_target_cond = None
    self.prev_lat = None
    self.prev_dy = 0.0
    self.dy_est = 0.0
    self.ff_state = 0.0

    self.fast_bias = 0.0
    self.slow_bias = 0.0
    self.last_innovation = 0.0

    self.last_action = 0.0
    self.prev_action_applied = False
    self.prev_v = 0.0
    self.prev_roll = 0.0
    self.prev_model = None
    self.preview_pid_integral = 0.0
    self.preview_pid_prev_err = 0.0

    self.risk_state = 0.0
    self.mix_state = self.base_mix
    self.accept_hold = 0
    self.tail_hold_count = 0
    self.recovery_hold_count = 0
    self.trust_ramp = 0.0
    self.consistency_state = 0.0
    self.mix_used_state = 0.0

    self.target_hist: List[float] = []
    self.target_cond_hist: List[float] = []
    self.error_hist: List[float] = []
    self.action_hist: List[float] = []

    # Residual state
    self.res_w = np.zeros(5, dtype=np.float64)
    self.res_feat_prev = np.zeros(5, dtype=np.float64)
    self.res_bias = 0.0

    # Instrumentation
    self.step_count = 0
    self.accept_count = 0
    self.blend_count = 0
    self.tail_forced_count = 0
    self.delta_clip_count = 0
    self.hard_reject_count = 0
    self.cost_reject_count = 0
    self.sign_reject_count = 0
    self.magnitude_reject_count = 0
    self.tail_reject_count = 0
    self.recovery_hold_active_count = 0
    self.accepted_delta_sum = 0.0
    self.accepted_err_improve_sum = 0.0
    self.accepted_adv_sum = 0.0
    self.accepted_adv_values: List[float] = []
    self.accept_regime_counts = [0, 0, 0]
    self.gate_regime_counts = [0, 0, 0]
    self.accept_gate_regime_counts = [0, 0, 0]
    self.trust_ramp_sum = 0.0
    self.consistency_sum = 0.0
    self.candidate_quality_sum = 0.0
    self.candidate_quality_values: List[float] = []
    self.candidate_count_sum = 0.0
    self.best_candidate_score_sum = 0.0
    self.tail_risk_sum = 0.0
    self.candidate_family_names = ("pid", "pid_ff", "short_predictive", "medium_predictive", "damped_recovery", "low_jerk", "aggressive_catchup")
    self.candidate_family_counts = {name: 0.0 for name in self.candidate_family_names}
    self.family_quality_ema = {name: 0.0 for name in self.candidate_family_names}
    self.family_used_adv_ema = {name: 0.0 for name in self.candidate_family_names}
    self.family_accepted_adv_sum = {name: 0.0 for name in self.candidate_family_names}
    self.family_accepted_count = {name: 0.0 for name in self.candidate_family_names}
    self.trust_before_tail_sum = 0.0
    self.trust_after_tail_sum = 0.0
    self.trust_values: List[float] = []
    self.family_agreement_sum = 0.0
    self.candidate_spread_sum = 0.0
    self.score_pid_sum = 0.0
    self.score_best_sum = 0.0
    self.score_gap_sum = 0.0
    self.score_relative_sum = 0.0
    self.score_relative_values: List[float] = []
    self.accepted_score_relative_sum = 0.0
    self.accepted_score_relative_values: List[float] = []
    self.trust_positive_sum = 0.0
    self.trust_positive_count = 0.0
    self.trust_negative_sum = 0.0
    self.trust_negative_count = 0.0
    self.tail_penalty_sum = 0.0

    self.mix_sum = 0.0
    self.mix_sq_sum = 0.0
    self.mix_used_sum = 0.0
    self.mix_used_sq_sum = 0.0
    self.mix_used_gate_sum = [0.0, 0.0, 0.0]
    self.regime_counts = [0, 0, 0]
    self.tail_mode_count = 0
    self.recovery_count = 0
    self.residual_abs_sum = 0.0
    self.residual_updates = 0
    self.simple_mode_steps = 0
    self.aggressive_steps = 0
    self.aggressive_interventions = 0
    self.aggressive_mix_used_sum = 0.0
    self.aggressive_abs_delta_sum = 0.0
    self.aggressive_delta_clip_count = 0
    self.aggressive_improve_sum = 0.0
    self.analytic_preview_steps = 0
    self.analytic_preview_abs_delta_sum = 0.0
    self.analytic_preview_gain_sum = 0.0
    self.bc_preview_steps = 0
    self.bc_preview_abs_delta_sum = 0.0
    self.preview_pid_steps = 0
    self.preview_pid_abs_delta_sum = 0.0
    self.known_good_steps = 0
    self.known_good_interventions = 0
    self.known_good_advanced_touch_count = 0
    self.known_good_mix_used_sum = 0.0
    self.known_good_abs_delta_sum = 0.0
    self.known_good_delta_clip_count = 0
    self.known_good_veto_count = 0
    self.fixed_mix_tail_veto_count = 0
    self.fixed_mix_sign_veto_count = 0
    self.fixed_mix_err_veto_count = 0
    self.fixed_mix_dy_veto_count = 0
    self.fixed_mix_delta_veto_count = 0
    self.fixed_mix_emergency_fallback_count = 0
    self.fixed_mix_emergency_active_count = 0
    self.fixed_mix_emergency_trigger_err_count = 0
    self.fixed_mix_emergency_trigger_dy_count = 0
    self.fixed_mix_emergency_trigger_rise_count = 0
    self.known_good_sign_disagree_count = 0
    self.known_good_large_delta_count = 0
    self.known_good_recent_harm_count = 0
    self.known_good_err_veto_count = 0
    self.known_good_dy_veto_count = 0
    self.known_good_flip_veto_count = 0
    self.known_good_sign_veto_count = 0
    self.known_good_hold_veto_count = 0
    self.known_good_harm_veto_count = 0
    self.known_good_post_residual_count = 0
    self.known_good_post_residual_abs_sum = 0.0
    self.known_good_prev_intervention_applied = False
    self.known_good_prev_intervention_err_abs = 0.0
    self.known_good_prev_dy_abs = 0.0
    self.known_good_emergency_prev_err_abs = 0.0
    self.known_good_emergency_prev_dy_abs = 0.0
    self.known_good_emergency_err_rise_count = 0
    self.known_good_emergency_hold_count = 0
    self.known_good_emergency_err_hist: List[float] = []
    self.known_good_emergency_dy_hist: List[float] = []
    self.known_good_recent_harm_ema = 0.0
    self.known_good_err_rise_count = 0
    self.known_good_dy_rise_count = 0
    self.known_good_local_harm_streak = 0
    self.known_good_sign_disagree_streak = 0
    self.known_good_veto_hold_count = 0
    self.known_good_reentry_pending = False
    self.known_good_reentry_count = 0
    self.known_good_blend_recent: List[int] = []
    self.pid_exact_hard_bypass_steps = 0
    self.simple_accept_count = 0
    self.simple_calm_gate_count = 0
    self.simple_mild_gate_count = 0
    self.simple_calm_accept_count = 0
    self.simple_mild_accept_count = 0
    self.simple_predictive_considered_count = 0
    self.simple_local_adv_pass_count = 0
    self.simple_local_adv_sum = 0.0
    self.simple_worst_accepted_local_adv = 0.0
    self.simple_signed_delta_sum = 0.0
    self.simple_max_abs_delta = 0.0
    self.simple_accepted_steps: List[int] = []
    self.simple_veto_err_growth = 0
    self.simple_veto_dy_growth = 0
    self.simple_veto_oscillation = 0
    self.simple_veto_recovery_trend = 0
    self.simple_veto_target_flip = 0
    self.simple_veto_limit = 0
    self.simple_veto_must_help = 0
    self.simple_veto_budget = 0
    self.simple_marginal_accept_count = 0
    self.simple_high_conviction_accept_count = 0
    self.simple_accept_recent: List[int] = []
    self.last_known_good_trace: Dict[str, float] = {}


    if self.diag_mode:
      print(
        f"control_delay_steps={self.control_delay_steps}, "
        f"CONTROL_START_IDX={CONTROL_START_IDX}, CONTEXT_LENGTH={CONTEXT_LENGTH}"
      )

    if self.echo_mode and not Controller._echo_done:
      Controller._echo_done = True
      print(
        "TOP1_MPC_ECHO "
        f"model_from_file={int(self.model_from_file)} "
        f"h=[{self.horizon_calm},{self.horizon_mid},{self.horizon_aggressive}] "
        f"base_mix={self.base_mix:.4g} mix=[{self.mix_min:.4g},{self.mix_max:.4g}] "
        f"accept_enable={int(self.accept_enable)} ff_only_mode={int(self.ff_only_mode)} simple_hybrid_mode={int(self.simple_hybrid_mode)} aggressive_horizon_mode={int(self.aggressive_horizon_mode)} known_good_fixed_mix_mode={int(self.known_good_fixed_mix_mode)} simple_veto_mode={int(self.simple_veto_mode)} fixed_mix_tail_veto_enable={int(self.fixed_mix_tail_veto_enable)} fixed_mix_emergency_fallback_enable={int(self.fixed_mix_emergency_fallback_enable)} fixed_mix_clip_fallback_enable={int(self.fixed_mix_clip_fallback_enable)} strict_safe_mode={int(self.strict_safe_mode)} legacy_fixed_mix_mode={int(self.legacy_fixed_mix_mode)} legacy_repro_mode={int(self.legacy_repro_mode)} legacy_pid_exact_mode={int(self.legacy_pid_exact_mode)} regime_enable={int(self.regime_enable)} tail_mode_enable={int(self.tail_mode_enable)} "
        f"aggr(h={self.aggressive_horizon},prefix={self.aggressive_prefix_steps},mix=[{self.aggressive_base_mix:.4g},{self.aggressive_mix_gain:.4g},{self.aggressive_mix_max:.4g}],weights=[{self.aggressive_track_weight:.3g},{self.aggressive_dy_weight:.3g},{self.aggressive_ddy_weight:.3g},{self.aggressive_du_weight:.3g},{self.aggressive_delta_weight:.3g},{self.aggressive_terminal_weight:.3g}],du={self.aggressive_du_max_step:.4g},delta={self.aggressive_delta_u_max_abs:.4g},vs_pid={self.aggressive_max_delta_vs_pid:.4g},polish={self.aggressive_polish_amp:.4g},clip_fallback={int(self.aggressive_clip_fallback_enable)}) "
        f"known_good(base_mix={self.base_mix:.4g},candidate={self.fixed_mix_candidate_mode},minimal_clips=1,clip_fallback={int(self.fixed_mix_clip_fallback_enable)},delta_scale=[enable:{int(self.fixed_mix_delta_scale_enable)},ref:{self.fixed_mix_delta_scale_ref:.4g},min:{self.fixed_mix_delta_scale_min:.4g}],nn_mix=[enable:{int(self.fixed_mix_nn_mix_enable)},floor:{self.fixed_mix_nn_mix_floor:.4g},gate:{self.fixed_mix_nn_gate_threshold:.4g},loaded:{int(self.fixed_mix_nn_model is not None)}],veto=[err_rise:{self.veto_err_rise_steps},dy_rise:{self.veto_dy_rise_steps},sign_streak:{self.veto_sign_streak},flip_window:{self.veto_flip_window},flip_count:{self.veto_flip_count_th},min_delta:{self.veto_min_mix_delta:.4g},hold:{self.veto_hold_steps}],tail_veto=[enable:{int(self.fixed_mix_tail_veto_enable)},dy:{self.veto_dy_thresh:.4g},err:{self.veto_err_thresh:.4g},delta:{self.veto_delta_thresh:.4g},sign_err:{self.veto_sign_err_thresh:.4g}],emergency=[enable:{int(self.fixed_mix_emergency_fallback_enable)},dy:{self.fixed_mix_emergency_dy_thresh:.4g},err:{self.fixed_mix_emergency_err_thresh:.4g},rise_steps:{self.fixed_mix_emergency_err_rise_steps},dy_confirm:{self.fixed_mix_emergency_dy_confirm:.4g},hold:{self.fixed_mix_emergency_hold_steps}]) "
        f"post_residual=[enable:{int(self.fixed_mix_post_residual_enable)},scale:{self.fixed_mix_post_residual_scale:.4g},clip:{self.fixed_mix_post_residual_clip:.4g},loaded:{int(self.fixed_mix_post_residual_model is not None)}] "
        f"simple(h={self.simple_horizon},mix=[{self.simple_mix_base:.4g},{self.simple_mix_min:.4g},{self.simple_mix_max:.4g}],gains=[{self.simple_mix_err_gain:.4g},{self.simple_mix_dy_gain:.4g}],weights=[{self.simple_track_weight:.3g},{self.simple_dy_weight:.3g},{self.simple_ddy_weight:.3g},{self.simple_du_weight:.3g},{self.simple_delta_weight:.3g}],max_pred_delta={self.simple_max_pred_delta:.4g}) "
        f"safe_gate(err=[{self.safe_err_gate_1:.4g},{self.safe_err_gate_2:.4g}],dy=[{self.safe_dy_gate_1:.4g},{self.safe_dy_gate_2:.4g}],mix=[{self.safe_mix_1:.4g},{self.safe_mix_2:.4g},0],delta_cap=[{self.safe_calm_delta_cap:.4g},{self.safe_mild_delta_cap:.4g}],adv_min=[{self.local_adv_min_calm:.4g},{self.local_adv_min_mild:.4g}],budget=[{self.accept_budget_window},{self.accept_budget_max_fraction:.4g}],marginal_after={self.marginal_accept_suppression_after_n}) "
        f"du_max_step={self.du_max_step:.4g} delta_u_max_abs={self.delta_u_max_abs:.4g} "
        f"residual_enable={int(self.residual_enable)} "
        f"accept(cost_ratio=[{self.accept_cost_ratio_calm:.4g},{self.accept_cost_ratio_mid:.4g},{self.accept_cost_ratio_aggressive:.4g}],min_adv=[{self.accept_min_advantage_calm:.4g},{self.accept_min_advantage_mid:.4g},{self.accept_min_advantage_aggressive:.4g}],min_mix={self.accept_min_mix:.4g},delta=[{self.accept_max_delta_normal:.4g},{self.accept_max_delta_relaxed:.4g}]) "
        f"trust(up={self.trust_ramp_up:.4g},down={self.trust_ramp_down:.4g},max={self.trust_ramp_max:.4g},mix_bonus={self.trust_ramp_mix_bonus:.4g}) "
        f"consistency(alpha={self.consistency_alpha:.4g},ratio_bonus={self.consistency_ratio_bonus:.4g},adv_bonus={self.consistency_adv_bonus:.4g},calm_gate={self.consistency_calm_gate:.4g},small_delta={self.consistency_small_delta:.4g}) "
        f"multi_candidate(enable={int(self.multi_candidate_enable)},score=[{self.candidate_score_track_w:.3g},{self.candidate_score_jerk_w:.3g},{self.candidate_score_rate_w:.3g},{self.candidate_score_tail_w:.3g}]) "
        f"tail_reject(err={self.tail_reject_err:.4g},dy={self.tail_reject_dy:.4g},sat={self.tail_reject_sat_margin:.4g},hold={self.recovery_hold_steps}) "
        f"blend(min_adv={self.blend_min_adv:.4g},scale={self.blend_adv_scale:.4g},tail_cap={self.blend_tail_cap:.4g}) "
        f"tail_hold_steps={self.tail_hold_steps} "
        f"mismatch(th={self.mismatch_thresh:.4g},gain={self.mismatch_gain:.4g})"
      )

  def _load_piecewise_model(self):
    path = (Path(__file__).resolve().parents[1] / "model_params.json").resolve()

    v_edges = np.array([0.0, 10.0, 20.0, 30.0, 50.0], dtype=np.float64)
    roll_edges = np.array([0.0, 0.5, 10.0], dtype=np.float64)
    y_coeffs = np.zeros((4, 2, 5), dtype=np.float64)
    dy_coeffs = np.zeros((4, 2, 5), dtype=np.float64)

    y_fallback = np.array([self.diag_a, self.diag_b0, self.diag_bv, self.diag_c, self.diag_d], dtype=np.float64)
    dy_fallback = np.array([self.fallback_A, self.fallback_B0, self.fallback_Bv, self.fallback_C, self.fallback_D], dtype=np.float64)

    y_coeffs[:] = y_fallback
    dy_coeffs[:] = dy_fallback

    if not path.exists():
      return v_edges, roll_edges, y_coeffs, dy_coeffs, False

    try:
      data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
      return v_edges, roll_edges, y_coeffs, dy_coeffs, False

    try:
      if isinstance(data, dict) and data.get("schema_version", 0) >= 2 and "dy_coeffs" in data:
        ve = np.array(data["v_edges"], dtype=np.float64)
        re = np.array(data["roll_edges"], dtype=np.float64)
        yc = np.array(data["y_coeffs"], dtype=np.float64)
        dc = np.array(data["dy_coeffs"], dtype=np.float64)
        if yc.ndim == 3 and dc.ndim == 3 and yc.shape[2] == 5 and dc.shape[2] == 5:
          if ve.ndim == 1 and re.ndim == 1 and ve.size == yc.shape[0] + 1 and re.size == yc.shape[1] + 1:
            return ve, re, yc, dc, True
      if isinstance(data, dict) and all(k in data for k in ("a", "b0", "bv", "c", "d")):
        y = np.array([data["a"], data["b0"], data["bv"], data["c"], data["d"]], dtype=np.float64)
        y_coeffs[:] = y
        return v_edges, roll_edges, y_coeffs, dy_coeffs, True
    except Exception:
      pass

    return v_edges, roll_edges, y_coeffs, dy_coeffs, False

  def _load_bc_preview_model(self):
    root = Path(__file__).resolve().parents[1]
    candidates = []
    override = os.getenv("TOP1_MPC_BC_MODEL_PATH", "").strip()
    if override:
      candidates.append(("mlp", Path(override).expanduser().resolve()))
    if self.bc_preview_use_mlp:
      candidates.append(("mlp", (root / "top1_mpc_bc_mlp.json").resolve()))
    candidates.append(("linear", (root / "top1_mpc_bc_linear.json").resolve()))

    for kind, path in candidates:
      if not path.exists():
        continue
      try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mu = np.asarray(data["mu"], dtype=np.float64)
        std = np.asarray(data["std"], dtype=np.float64)
        std = np.where(std < 1e-6, 1.0, std)
        if kind == "mlp" and "layers" in data:
          layers = []
          for layer in data["layers"]:
            layers.append(
              {
                "w": np.asarray(layer["w"], dtype=np.float64),
                "b": np.asarray(layer["b"], dtype=np.float64),
                "act": str(layer["act"]),
              }
            )
          return {"kind": "mlp", "mu": mu, "std": std, "layers": layers}
        if "w" in data:
          w = np.asarray(data["w"], dtype=np.float64)
          if w.ndim == 1 and mu.shape == w.shape and std.shape == w.shape:
            return {"kind": "linear", "w": w, "mu": mu, "std": std}
      except Exception:
        continue
    return None

  def _load_fixed_mix_nn_model(self):
    override = os.getenv("TOP1_MPC_MIX_MODEL_PATH", "").strip()
    if override:
      path = Path(override).expanduser().resolve()
    else:
      path = (Path(__file__).resolve().parents[1] / "top1_mpc_mix_mlp.json").resolve()
    if not path.exists():
      return None
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
      mu = np.asarray(data["mu"], dtype=np.float64)
      std = np.asarray(data["std"], dtype=np.float64)
      std = np.where(std < 1e-6, 1.0, std)
      layers = []
      for layer in data["layers"]:
        w = np.asarray(layer["w"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        act = str(layer["act"])
        layers.append({"w": w, "b": b, "act": act})
      return {"mu": mu, "std": std, "layers": layers}
    except Exception:
      return None

  def _load_oracle_nn_model(self):
    override = os.getenv("TOP1_MPC_ORACLE_MODEL_PATH", "").strip()
    if override:
      path = Path(override).expanduser().resolve()
    else:
      path = (Path(__file__).resolve().parents[1] / "top1_mpc_oracle_mlp.json").resolve()
    if not path.exists():
      return None
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
      mu = np.asarray(data["mu"], dtype=np.float64)
      std = np.asarray(data["std"], dtype=np.float64)
      std = np.where(std < 1e-6, 1.0, std)
      layers = []
      for layer in data["layers"]:
        layers.append(
          {
            "w": np.asarray(layer["w"], dtype=np.float64),
            "b": np.asarray(layer["b"], dtype=np.float64),
            "act": str(layer["act"]),
          }
        )
      return {"mu": mu, "std": std, "layers": layers}
    except Exception:
      return None

  def _load_fixed_mix_post_residual_model(self):
    override = os.getenv("TOP1_MPC_POST_RESIDUAL_MODEL_PATH", "").strip()
    if override:
      path = Path(override).expanduser().resolve()
    else:
      path = (Path(__file__).resolve().parents[1] / "top1_mpc_post_residual_mlp.json").resolve()
    if not path.exists():
      return None
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
      mu = np.asarray(data["mu"], dtype=np.float64)
      std = np.asarray(data["std"], dtype=np.float64)
      std = np.where(std < 1e-6, 1.0, std)
      layers = []
      for layer in data["layers"]:
        layers.append(
          {
            "w": np.asarray(layer["w"], dtype=np.float64),
            "b": np.asarray(layer["b"], dtype=np.float64),
            "act": str(layer["act"]),
          }
        )
      output_scale = float(np.clip(data.get("output_scale", self.fixed_mix_post_residual_clip), 0.0, 1.0))
      return {"mu": mu, "std": std, "layers": layers, "output_scale": output_scale}
    except Exception:
      return None

  def _oracle_nn_action(self, target_raw: float, lat: float, state, future_plan, u_pid: float, u_legacy: float):
    if self.oracle_nn_model is None:
      return float(u_legacy)
    x = self._fixed_mix_nn_features(target_raw, lat, state, future_plan, u_pid, u_legacy)
    x = (x - self.oracle_nn_model["mu"]) / self.oracle_nn_model["std"]
    for layer in self.oracle_nn_model["layers"]:
      x = layer["w"] @ x + layer["b"]
      if layer["act"] == "relu":
        x = np.maximum(x, 0.0)
      elif layer["act"] == "tanh":
        x = np.tanh(x)
    u = 2.0 * float(np.asarray(x).reshape(-1)[0])
    return float(np.clip(u, self.u_min, self.u_max))

  def _fixed_mix_nn_features(self, target_raw: float, lat: float, state, future_plan, u_pid: float, u_pred_raw: float):
    future = list(getattr(future_plan, "lataccel", []))

    def tg(k: int):
      if not future:
        return float(target_raw)
      idx = min(k, len(future) - 1)
      return float(future[idx])

    err = float(target_raw - lat)
    dy = 0.0 if self.prev_lat is None else float((lat - self.prev_lat) / self.dt)
    delta = float(u_pred_raw - u_pid)
    last_action = float(self.last_action if self.prev_action_applied else 0.0)
    return np.array(
      [
        err,
        abs(err),
        dy,
        abs(dy),
        float(lat),
        float(target_raw),
        float(state.v_ego),
        float(state.roll_lataccel),
        float(state.a_ego),
        last_action,
        float(u_pid),
        float(u_pred_raw),
        delta,
        abs(delta),
        tg(0),
        tg(1),
        tg(3),
        tg(7),
      ],
      dtype=np.float64,
    )

  def _fixed_mix_nn_scale(self, target_raw: float, lat: float, state, future_plan, u_pid: float, u_pred_raw: float):
    if self.fixed_mix_nn_model is None:
      return 1.0
    x = self._fixed_mix_nn_features(target_raw, lat, state, future_plan, u_pid, u_pred_raw)
    x = (x - self.fixed_mix_nn_model["mu"]) / self.fixed_mix_nn_model["std"]
    for layer in self.fixed_mix_nn_model["layers"]:
      x = layer["w"] @ x + layer["b"]
      if layer["act"] == "relu":
        x = np.maximum(x, 0.0)
      elif layer["act"] == "sigmoid":
        x = 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
    scale = float(np.asarray(x).reshape(-1)[0])
    return float(np.clip(scale, self.fixed_mix_nn_mix_floor, 1.0))

  def _fixed_mix_post_residual_features(
    self,
    target_raw: float,
    lat: float,
    state,
    future_plan,
    u_pid: float,
    u_pred_raw: float,
    base_action: float,
    mix_eff: float,
  ):
    future = list(getattr(future_plan, "lataccel", []))

    def tg(k: int):
      if not future:
        return float(target_raw)
      idx = min(k, len(future) - 1)
      return float(future[idx])

    err = float(target_raw - lat)
    dy = 0.0 if self.prev_lat is None else float((lat - self.prev_lat) / self.dt)
    last_action = float(self.last_action if self.prev_action_applied else 0.0)
    return np.array(
      [
        err,
        abs(err),
        dy,
        abs(dy),
        float(lat),
        float(target_raw),
        float(state.v_ego),
        float(state.roll_lataccel),
        float(state.a_ego),
        last_action,
        float(u_pid),
        float(u_pred_raw),
        float(base_action),
        float(u_pred_raw - u_pid),
        float(base_action - u_pid),
        float(mix_eff),
        tg(0),
        tg(1),
        tg(3),
        tg(7),
      ],
      dtype=np.float64,
    )

  def _fixed_mix_post_residual_delta(
    self,
    target_raw: float,
    lat: float,
    state,
    future_plan,
    u_pid: float,
    u_pred_raw: float,
    base_action: float,
    mix_eff: float,
  ):
    if self.fixed_mix_post_residual_model is None:
      return 0.0
    x = self._fixed_mix_post_residual_features(target_raw, lat, state, future_plan, u_pid, u_pred_raw, base_action, mix_eff)
    x = (x - self.fixed_mix_post_residual_model["mu"]) / self.fixed_mix_post_residual_model["std"]
    for layer in self.fixed_mix_post_residual_model["layers"]:
      x = layer["w"] @ x + layer["b"]
      if layer["act"] == "relu":
        x = np.maximum(x, 0.0)
      elif layer["act"] == "tanh":
        x = np.tanh(x)
    delta = float(self.fixed_mix_post_residual_model["output_scale"] * np.asarray(x).reshape(-1)[0])
    return float(np.clip(delta, -self.fixed_mix_post_residual_clip, self.fixed_mix_post_residual_clip))

  def _bc_preview_features(self, target_raw: float, state, future_plan):
    if self.bc_preview_model is None:
      return None
    future = list(getattr(future_plan, "lataccel", []))

    def tg(k: int):
      if not future:
        return float(target_raw)
      idx = min(k, len(future) - 1)
      return float(future[idx])

    t0 = float(target_raw)
    t1, t2, t3, t5, t8 = tg(0), tg(1), tg(2), tg(4), tg(7)
    v0 = float(state.v_ego)
    r0 = float(state.roll_lataccel)
    a0 = float(state.a_ego)
    vs = float(np.clip(v0 / 30.0, 0.0, 2.0))
    x = np.array([
      1.0,
      t0, t1, t2, t3, t5, t8,
      t1 - t0, t2 - t1, t3 - t2, t5 - t3,
      (t1 + t2 + t3) / 3.0,
      r0, a0, v0,
      vs * t0, vs * t1, r0 * vs,
      abs(t0), abs(t1 - t0),
    ], dtype=np.float64)
    return (x - self.bc_preview_model["mu"]) / self.bc_preview_model["std"]

  def _bc_preview_ff(self, feat: np.ndarray):
    if feat is None or self.bc_preview_model is None:
      return 0.0
    if self.bc_preview_model.get("kind") == "mlp":
      x = feat
      for layer in self.bc_preview_model["layers"]:
        x = layer["w"] @ x + layer["b"]
        if layer["act"] == "relu":
          x = np.maximum(x, 0.0)
        elif layer["act"] == "tanh":
          x = np.tanh(x)
      return float(2.0 * np.asarray(x).reshape(-1)[0])
    return float(feat @ self.bc_preview_model["w"])

  def _bc_preview_action(self, target_raw: float, lat: float, state, future_plan, u_pid: float):
    feat = self._bc_preview_features(target_raw, state, future_plan)
    if feat is None:
      return float(u_pid), {"ff": float(u_pid), "ff_blend": float(u_pid)}
    ff = self._bc_preview_ff(feat)
    dy_now = 0.0 if self.prev_lat is None else float((lat - self.prev_lat) / self.dt)
    err = float(target_raw - lat)
    ff_blend = float((1.0 - self.bc_preview_pid_mix) * ff + self.bc_preview_pid_mix * u_pid)
    action = float(ff_blend + self.bc_preview_err_gain * err - self.bc_preview_dy_gain * dy_now)
    action = float(np.clip(action, self.u_min, self.u_max))
    return action, {"ff": ff, "ff_blend": ff_blend, "dy": dy_now, "err": err}

  def _axis_interp(self, x: float, edges: np.ndarray) -> Tuple[int, int, float]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    if x <= centers[0]:
      return 0, 0, 0.0
    if x >= centers[-1]:
      last = centers.size - 1
      return last, last, 0.0
    hi = int(np.searchsorted(centers, x, side="right"))
    lo = hi - 1
    span = max(1e-9, float(centers[hi] - centers[lo]))
    w = float(np.clip((x - centers[lo]) / span, 0.0, 1.0))
    return lo, hi, w

  def _select_coeffs(self, v: float, roll: float):
    av = float(np.clip(v, self.v_edges[0], self.v_edges[-1]))
    ar = float(np.clip(abs(roll), self.roll_edges[0], self.roll_edges[-1]))

    vi0, vi1, wv = self._axis_interp(av, self.v_edges)
    ri0, ri1, wr = self._axis_interp(ar, self.roll_edges)

    def bilinear(grid: np.ndarray):
      c00 = grid[vi0, ri0]
      c01 = grid[vi0, ri1]
      c10 = grid[vi1, ri0]
      c11 = grid[vi1, ri1]
      return (1.0 - wv) * ((1.0 - wr) * c00 + wr * c01) + wv * ((1.0 - wr) * c10 + wr * c11)

    yv = bilinear(self.y_coeffs)
    dv = bilinear(self.dy_coeffs)
    y_model = {"a": float(yv[0]), "b0": float(yv[1]), "bv": float(yv[2]), "c": float(yv[3]), "d": float(yv[4])}
    dy_model = {"A": float(dv[0]), "B0": float(dv[1]), "Bv": float(dv[2]), "C": float(dv[3]), "D": float(dv[4])}
    bins = {"vi0": vi0, "vi1": vi1, "ri0": ri0, "ri1": ri1, "wv": wv, "wr": wr}
    return y_model, dy_model, bins

  def _effective_control_gain(self, y_model: Dict[str, float], dy_model: Dict[str, float], v: float):
    v_clip = float(np.clip(v, 0.0, 50.0))
    b_y = float(y_model["b0"] + y_model["bv"] * v_clip)
    b_dy_raw = float(dy_model["B0"] + dy_model["Bv"] * v_clip)
    sign_flip = False

    b_dy = b_dy_raw
    if abs(b_y) > 1e-5 and abs(b_dy_raw) > 1e-6 and np.sign(b_y) != np.sign(b_dy_raw):
      b_dy = float(abs(b_dy_raw) * np.sign(b_y))
      sign_flip = True
    elif abs(b_dy_raw) <= 1e-6 and abs(b_y) > 1e-5:
      b_dy = float(0.01 * np.sign(b_y))
      sign_flip = True

    return b_y, b_dy_raw, b_dy, sign_flip

  def _build_horizon_seq(self, now: float, future: List[float], H: int) -> np.ndarray:
    out = np.empty(H, dtype=np.float64)
    out[0] = float(now)
    if H == 1:
      return out
    n = min(len(future), H - 1)
    if n > 0:
      out[1:1 + n] = np.asarray(future[:n], dtype=np.float64)
    fill = out[n] if n > 0 else out[0]
    if 1 + n < H:
      out[1 + n:] = fill
    return out

  def _condition_target(self, target_raw: float):
    self.target_hist.append(float(target_raw))
    if len(self.target_hist) > 8:
      self.target_hist = self.target_hist[-8:]

    if self.target_median_enable and len(self.target_hist) >= 3:
      target_med = float(np.median(np.asarray(self.target_hist[-3:], dtype=np.float64)))
    else:
      target_med = float(target_raw)

    if self.prev_target_cond is None:
      target_cond = target_med
      dtarget = 0.0
      curvature = 0.0
    else:
      alpha = self.target_lpf_alpha
      target_cond = float((1.0 - alpha) * self.prev_target_cond + alpha * target_med)
      dtarget = (target_cond - self.prev_target_cond) / self.dt

      self.target_cond_hist.append(target_cond)
      if len(self.target_cond_hist) > max(8, self.curvature_window + 2):
        self.target_cond_hist = self.target_cond_hist[-max(8, self.curvature_window + 2):]

      if len(self.target_cond_hist) >= self.curvature_window + 1:
        vals = np.asarray(self.target_cond_hist[-(self.curvature_window + 1):], dtype=np.float64)
        dv = np.diff(vals) / self.dt
        curvature = float((dv[-1] - dv[0]) / max(self.dt * len(dv), 1e-6))
      else:
        curvature = 0.0

      if abs(target_med - self.prev_target_cond) > self.target_step_thresh:
        # step detector: suppress derivative spike to avoid jerk bursts
        dtarget *= 0.55

    if len(self.target_cond_hist) == 0:
      self.target_cond_hist.append(target_cond)

    self.prev_target_cond = target_cond
    return float(target_cond), float(dtarget), float(curvature)

  def _update_dy_estimate(self, lat: float):
    if self.prev_lat is None:
      self.dy_est = 0.0
      return 0.0

    dy_meas = (lat - self.prev_lat) / self.dt
    beta = float(self.dt / (self.dy_filter_tau + self.dt))
    self.dy_est = float(self.dy_est + beta * (dy_meas - self.dy_est))
    return float(dy_meas)

  def _update_observer(self, dy_now: float):
    if self.prev_model is None or not self.prev_action_applied:
      self.fast_bias = float(np.clip(self.fast_leak * self.fast_bias, -self.fast_clip, self.fast_clip))
      self.slow_bias = float(np.clip(self.slow_leak * self.slow_bias, -self.slow_clip, self.slow_clip))
      self.last_innovation = 0.0
      return

    pred = (
      self.prev_model["A"] * self.prev_dy
      + self.prev_model["B"] * self.last_action
      + self.prev_model["C"] * self.prev_roll
      + self.prev_model["D"]
      + self.fast_bias
      + self.slow_bias
    )
    innovation = float(dy_now - pred)
    self.last_innovation = innovation

    self.fast_bias = float(np.clip(self.fast_leak * self.fast_bias + self.fast_gain * innovation, -self.fast_clip, self.fast_clip))
    self.slow_bias = float(np.clip(self.slow_leak * self.slow_bias + self.slow_gain * innovation, -self.slow_clip, self.slow_clip))

    if self.residual_enable:
      self.res_bias = float(np.clip(self.residual_decay * self.res_bias + self.residual_lr * innovation, -self.residual_clip, self.residual_clip))
      self.res_w = np.clip(
        self.residual_decay * self.res_w + self.residual_lr * innovation * self.res_feat_prev,
        -self.residual_weight_clip,
        self.residual_weight_clip,
      )
      self.residual_updates += 1

  def _predict_dy_y_next(self, lat: float, dy: float, roll: float, u: float, model_step: Dict[str, float]):
    dy_next = (
      model_step["A"] * dy
      + model_step["B"] * u
      + model_step["C"] * roll
      + model_step["D"]
      + self.fast_bias
      + self.slow_bias
    )
    y_next = float(lat + self.dt * dy_next)
    return float(dy_next), float(y_next)

  def _desired_correction_direction(self, lat: float, dy: float, roll: float, target: float, u_pid: float, model_step: Dict[str, float]):
    eps = float(max(1e-3, min(self.max_mpc_delta_vs_pid, self.delta_u_max_abs, self.du_max_step, 0.05)))
    u_minus = float(np.clip(u_pid - eps, self.u_min, self.u_max))
    u_plus = float(np.clip(u_pid + eps, self.u_min, self.u_max))

    _, y_pid = self._predict_dy_y_next(lat, dy, roll, u_pid, model_step)
    _, y_minus = self._predict_dy_y_next(lat, dy, roll, u_minus, model_step)
    _, y_plus = self._predict_dy_y_next(lat, dy, roll, u_plus, model_step)

    err_pid = float(target - y_pid)
    err_minus = float(target - y_minus)
    err_plus = float(target - y_plus)

    best_u = u_pid
    best_abs_err = abs(err_pid)
    if abs(err_minus) < best_abs_err:
      best_u = u_minus
      best_abs_err = abs(err_minus)
    if abs(err_plus) < best_abs_err:
      best_u = u_plus

    desired_sign = 0
    if best_u > u_pid + 1e-12:
      desired_sign = 1
    elif best_u < u_pid - 1e-12:
      desired_sign = -1

    return {
      "desired_correction_sign": desired_sign,
      "one_step_err_if_pid": err_pid,
      "one_step_err_if_minus": err_minus,
      "one_step_err_if_plus": err_plus,
    }

  def _align_candidate_sign(self, lat: float, dy: float, roll: float, target: float, u_pid: float, u_candidate_raw: float, model_step: Dict[str, float]):
    local_dir = self._desired_correction_direction(lat, dy, roll, target, u_pid, model_step)
    desired_sign = int(local_dir["desired_correction_sign"])

    cand_sign = 0
    if u_candidate_raw > u_pid + 1e-12:
      cand_sign = 1
    elif u_candidate_raw < u_pid - 1e-12:
      cand_sign = -1

    sign_corrected = 0
    if desired_sign != 0 and cand_sign != 0 and cand_sign != desired_sign:
      mirrored = float(u_pid - (u_candidate_raw - u_pid))
      _, y_raw = self._predict_dy_y_next(lat, dy, roll, u_candidate_raw, model_step)
      _, y_mirror = self._predict_dy_y_next(lat, dy, roll, mirrored, model_step)
      if abs(target - y_mirror) <= abs(target - y_raw):
        u_candidate_raw = mirrored
        cand_sign = desired_sign
        sign_corrected = 1

    sign_agree = int(desired_sign == 0 or cand_sign == 0 or desired_sign == cand_sign)
    return float(u_candidate_raw), local_dir, sign_agree, sign_corrected

  def _tail_signal(self, err: float, dy: float, jerk_proxy: float, sat_ratio: float, risk: float):
    abs_err_hist = [abs(v) for v in self.error_hist[-3:]] + [abs(err)]
    err_growth_sustained = False
    if len(abs_err_hist) >= 3:
      err_growth_sustained = bool(
        abs_err_hist[-1] > abs_err_hist[-2] + 0.04
        and abs_err_hist[-2] > abs_err_hist[-3] + 0.02
        and abs_err_hist[-1] > 0.72 * self.tail_err_thresh
      )
    dy_growth_sustained = bool(
      abs(dy) > 0.78 * self.tail_dy_thresh
      and abs(jerk_proxy) > 0.55 * self.tail_dy_thresh
      and abs(dy) > abs(self.prev_dy) + 0.10
    )
    sign_flips = 0
    recent_err = self.error_hist[-3:] + [err]
    for i in range(1, len(recent_err)):
      if recent_err[i - 1] * recent_err[i] < 0.0:
        sign_flips += 1
    oscillation_near_sat = bool(sign_flips >= 2 and sat_ratio > self.tail_reject_sat_margin)
    severe_abs = bool(abs(err) > self.tail_err_thresh or abs(dy) > self.tail_dy_thresh)
    severe_risk = bool(risk > 0.92 and sat_ratio > 0.80 * self.tail_reject_sat_margin)
    return bool(severe_abs or err_growth_sustained or dy_growth_sustained or oscillation_near_sat or severe_risk)

  def _classify_regime(self, err: float, dy: float, dtarget: float, sat_ratio: float, jerk_proxy: float, err_growth: float, mismatch: float):
    r_err = _sigmoid((abs(err) - self.regime_err_thresh) / (0.35 * self.regime_err_thresh + 1e-6))
    r_dy = _sigmoid((abs(dy) - self.regime_dy_thresh) / (0.35 * self.regime_dy_thresh + 1e-6))
    r_step = _sigmoid((abs(dtarget) - self.regime_step_thresh) / (0.35 * self.regime_step_thresh + 1e-6))
    r_sat = _sigmoid((sat_ratio - self.regime_sat_thresh) / (0.25 * self.regime_sat_thresh + 1e-6))
    r_jerk = _sigmoid((abs(jerk_proxy) - self.regime_dy_thresh) / (0.35 * self.regime_dy_thresh + 1e-6))
    r_growth = _sigmoid((err_growth - self.tail_growth_thresh) / (0.35 * (self.tail_growth_thresh + 1e-6)))
    r_mismatch = _sigmoid((abs(mismatch) - self.mismatch_thresh) / (0.35 * self.mismatch_thresh + 1e-6))

    raw = float(0.26 * r_err + 0.20 * r_dy + 0.15 * r_step + 0.11 * r_sat + 0.10 * r_jerk + 0.08 * r_growth + 0.10 * r_mismatch)
    self.risk_state = float(self.risk_alpha * self.risk_state + (1.0 - self.risk_alpha) * raw)

    risk = float(np.clip(self.risk_state, 0.0, 1.0))

    if risk < 0.33 - self.risk_hysteresis:
      regime = 0
    elif risk < 0.66 + self.risk_hysteresis:
      regime = 1
    else:
      regime = 2

    return risk, regime

  def _schedule_mix(self, base_mix: float, err: float, dy: float, jerk_proxy: float, dtarget: float, risk: float, mismatch: float, tail_mode: bool):
    mix_raw = (
      base_mix
      + self.mix_err_gain * abs(err)
      + self.mix_dy_gain * abs(dy)
      + self.mix_jerk_gain * abs(jerk_proxy)
      + self.mix_step_gain * abs(dtarget)
    )
    if tail_mode:
      mix_raw += self.recovery_mix_bonus * (1.0 + self.tail_risk_gain * risk)

    mismatch_excess = max(0.0, abs(mismatch) - self.mismatch_thresh)
    mismatch_guard = 1.0 / (1.0 + self.mismatch_gain * mismatch_excess)
    mix_raw *= mismatch_guard

    mix_raw = float(np.clip(mix_raw, self.mix_min, self.mix_max))
    mix_next = float(self.mix_state + self.mix_smoothing * (mix_raw - self.mix_state))
    if abs(mix_next - self.mix_state) < self.mix_hysteresis:
      mix_next = self.mix_state
    self.mix_state = float(np.clip(mix_next, self.mix_min, self.mix_max))
    return self.mix_state

  def _schedule_horizon(self, regime: int, risk: float):
    if regime == 0:
      return int(self.horizon_calm)
    if regime == 1:
      mix = float(np.clip((risk - 0.33) / 0.33, 0.0, 1.0))
      return int(np.clip(round(_lerp(self.horizon_mid, self.horizon_aggressive, 0.4 * mix)), self.horizon_calm, self.horizon_aggressive))
    return int(self.horizon_aggressive)

  def _schedule_weights(self, regime: int, risk: float, tail_mode: bool, oscillatory: bool):
    if regime == 0:
      t = 0.0
      track_w = self.track_weight_calm
      dy_w = self.dy_weight_calm
      ddy_w = self.ddy_weight_calm
      du_w = self.du_weight_calm
      delta_w = self.delta_weight_calm
      barrier_w = self.barrier_weight_calm
      term_w = self.terminal_weight
    elif regime == 1:
      t = float(np.clip((risk - 0.33) / 0.33, 0.0, 1.0))
      track_w = _lerp(self.track_weight_mid, self.track_weight_aggressive, 0.35 * t)
      dy_w = _lerp(self.dy_weight_mid, self.dy_weight_aggressive, 0.35 * t)
      ddy_w = _lerp(self.ddy_weight_mid, self.ddy_weight_aggressive, 0.35 * t)
      du_w = _lerp(self.du_weight_mid, self.du_weight_aggressive, 0.35 * t)
      delta_w = _lerp(self.delta_weight_mid, self.delta_weight_aggressive, 0.35 * t)
      barrier_w = _lerp(self.barrier_weight_mid, self.barrier_weight_aggressive, 0.35 * t)
      term_w = _lerp(self.terminal_weight, self.terminal_weight_aggressive, 0.35 * t)
    else:
      t = 1.0
      track_w = self.track_weight_aggressive
      dy_w = self.dy_weight_aggressive
      ddy_w = self.ddy_weight_aggressive
      du_w = self.du_weight_aggressive
      delta_w = self.delta_weight_aggressive
      barrier_w = self.barrier_weight_aggressive
      term_w = self.terminal_weight_aggressive

    if tail_mode:
      track_w *= self.recovery_track_weight_scale
      dy_w *= 1.15
      term_w *= 1.25

    if oscillatory:
      ddy_w *= self.oscillation_damping_scale
      du_w *= self.oscillation_damping_scale
      delta_w *= self.oscillation_damping_scale

    return {
      "track_w": float(track_w),
      "dy_w": float(dy_w),
      "ddy_w": float(ddy_w),
      "du_w": float(du_w),
      "delta_w": float(delta_w),
      "barrier_w": float(barrier_w),
      "term_w": float(term_w),
    }

  def _simple_schedule_mix(self, err: float, dy: float):
    err_abs = abs(err)
    dy_abs = abs(dy)
    if err_abs < self.safe_err_gate_1 and dy_abs < self.safe_dy_gate_1:
      mix = self.safe_mix_1
    elif err_abs < self.safe_err_gate_2 and dy_abs < self.safe_dy_gate_2:
      mix = self.safe_mix_2
    else:
      mix = 0.0
    return float(np.clip(mix, 0.0, 1.0))

  def _simple_predictive_action(self, target_raw: float, lat: float, dy: float, state, future_plan, u_pid: float):
    H = int(np.clip(self.simple_horizon, 6, 20))
    tgt_seq = self._build_horizon_seq(target_raw, list(getattr(future_plan, "lataccel", [])), H)
    v_seq = self._build_horizon_seq(float(state.v_ego), list(getattr(future_plan, "v_ego", [])), H)
    roll_seq = self._build_horizon_seq(float(state.roll_lataccel), list(getattr(future_plan, "roll_lataccel", [])), H)

    ref_dy = np.zeros(H, dtype=np.float64)
    for k in range(1, H):
      ref_dy[k] = (tgt_seq[k] - tgt_seq[k - 1]) / self.dt

    x0 = np.array([lat - tgt_seq[0], dy - ref_dy[0]], dtype=np.float64)
    weights = {
      "track_w": float(self.simple_track_weight),
      "dy_w": float(self.simple_dy_weight),
      "ddy_w": float(self.simple_ddy_weight),
      "du_w": float(self.simple_du_weight),
      "delta_w": float(self.simple_delta_weight),
      "barrier_w": float(self.simple_barrier_weight),
      "term_w": float(self.simple_terminal_weight),
    }

    A_mats, B_vecs, c_vecs, stage_bins = self._build_stage_dynamics(H, tgt_seq, ref_dy, v_seq, roll_seq, u_pid)
    K, kff = self._solve_lqr(A_mats, B_vecs, c_vecs, weights)
    _, u_seq, _ = self._forward_plan(
      x0=x0,
      u_pid=u_pid,
      du_max_eff=self.du_max_step,
      A_mats=A_mats,
      B_vecs=B_vecs,
      c_vecs=c_vecs,
      ref_dy=ref_dy,
      K=K,
      kff=kff,
      weights=weights,
      delta_overrides={},
    )

    u_pred_raw = float(u_seq[0])
    bins0, A0, B0, C0, D0, _, _, _ = stage_bins[0]
    model_step0 = {"A": A0, "B": B0, "C": C0, "D": D0}
    u_pred_aligned, local_dir, sign_agree, sign_corrected = self._align_candidate_sign(
      lat,
      dy,
      float(state.roll_lataccel),
      target_raw,
      u_pid,
      u_pred_raw,
      model_step0,
    )
    u_pred = float(np.clip(u_pred_aligned, u_pid - self.simple_max_pred_delta, u_pid + self.simple_max_pred_delta))
    u_pred = float(np.clip(u_pred, self.u_min, self.u_max))
    return u_pred, {
      "H": H,
      "A": A0,
      "B": B0,
      "C": C0,
      "D": D0,
      "bins": bins0,
      "u_pred_raw": u_pred_raw,
      "u_pred_aligned": u_pred_aligned,
      "local_dir": local_dir,
      "sign_agree": int(sign_agree),
      "sign_corrected": int(sign_corrected),
    }

  def _legacy_repro_candidate(self, target_raw: float, lat: float, state, future_plan, u_pid: float):
    H = int(np.clip(self.horizon, 6, 24))
    tgt_seq = self._build_horizon_seq(target_raw, list(getattr(future_plan, "lataccel", [])), H)
    v_seq = self._build_horizon_seq(float(state.v_ego), list(getattr(future_plan, "v_ego", [])), H)
    roll_seq = self._build_horizon_seq(float(state.roll_lataccel), list(getattr(future_plan, "roll_lataccel", [])), H)

    A_arr = np.zeros(H, dtype=np.float64)
    B_arr = np.zeros(H, dtype=np.float64)
    c_arr = np.zeros(H, dtype=np.float64)
    stage_bins = []

    for k in range(H):
      y_model, _, bins = self._select_coeffs(float(v_seq[k]), float(roll_seq[k]))
      a = float(np.clip(y_model["a"], 0.0, 1.2))
      b = float(np.clip(y_model["b0"] + y_model["bv"] * float(np.clip(v_seq[k], 0.0, 50.0)), -5.0, 5.0))
      c_roll = float(np.clip(y_model["c"], -5.0, 5.0))
      d = float(np.clip(y_model["d"], -5.0, 5.0))
      k1 = min(k + 1, H - 1)

      A_arr[k] = a
      B_arr[k] = b
      c_arr[k] = float(a * tgt_seq[k] + b * u_pid + c_roll * roll_seq[k] + d - tgt_seq[k1])
      stage_bins.append((bins, a, b, c_roll, d))

    if float(np.max(np.abs(B_arr))) < 1e-6:
      bins0, a0, b0, c0, d0 = stage_bins[0]
      return float(u_pid), {
        "H": H,
        "a": a0,
        "b": b0,
        "c": c0,
        "d": d0,
        "bins": bins0,
        "u_pred_raw": float(u_pid),
      }

    Q = float(self.tracking_weight)
    R = float(self.du_weight + self.delta_weight + 1e-6)
    P = float(self.terminal_weight * self.tracking_weight)
    p = 0.0
    K = np.zeros(H, dtype=np.float64)
    kff = np.zeros(H, dtype=np.float64)

    for k in range(H - 1, -1, -1):
      a = A_arr[k]
      b = B_arr[k]
      c = c_arr[k]
      Pn = P
      pn = p
      Hs = float(R + b * b * Pn)
      G = float(b * a * Pn)
      g = float(b * (Pn * c + pn))
      K[k] = G / Hs
      kff[k] = g / Hs
      P = float(Q + a * a * Pn - (G * G) / Hs)
      p = float(a * (Pn * c + pn) - G * (g / Hs))

    e = float(lat - tgt_seq[0])
    u_prev = float(self.last_action if self.prev_action_applied else np.clip(u_pid, self.u_min, self.u_max))
    u_seq = np.zeros(H, dtype=np.float64)

    for k in range(H):
      delta = float(-(K[k] * e + kff[k]))
      delta = float(np.clip(delta, -self.delta_u_max_abs, self.delta_u_max_abs))
      u_nom = float(np.clip(u_pid + delta, self.u_min, self.u_max))
      du = float(np.clip(u_nom - u_prev, -self.du_max_step, self.du_max_step))
      u = float(np.clip(u_prev + du, self.u_min, self.u_max))
      delta_real = float(u - u_pid)
      e = float(A_arr[k] * e + B_arr[k] * delta_real + c_arr[k])
      u_prev = u
      u_seq[k] = u

    bins0, a0, b0, c0, d0 = stage_bins[0]
    return float(u_seq[0]), {
      "H": H,
      "a": a0,
      "b": b0,
      "c": c0,
      "d": d0,
      "bins": bins0,
      "u_pred_raw": float(u_seq[0]),
    }

  def _forward_plan(
    self,
    x0: np.ndarray,
    u_pid: float,
    du_max_eff: float,
    A_mats: np.ndarray,
    B_vecs: np.ndarray,
    c_vecs: np.ndarray,
    ref_dy: np.ndarray,
    K: np.ndarray,
    kff: np.ndarray,
    weights: Dict[str, float],
    delta_overrides: Dict[int, float],
  ):
    H = A_mats.shape[0]
    x = x0.copy()

    u_prev = float(self.last_action if self.prev_action_applied else np.clip(u_pid, self.u_min, self.u_max))
    u_seq = np.zeros(H, dtype=np.float64)
    d_seq = np.zeros(H, dtype=np.float64)

    track_w = weights["track_w"]
    dy_w = weights["dy_w"]
    ddy_w = weights["ddy_w"]
    du_w = weights["du_w"]
    delta_w = weights["delta_w"]
    barrier_w = weights["barrier_w"]

    cost = 0.0
    for k in range(H):
      delta = float(-(K[k] @ x + kff[k]))
      if k in delta_overrides:
        delta += float(delta_overrides[k])
      delta = float(np.clip(delta, -self.delta_u_max_abs, self.delta_u_max_abs))

      u = float(np.clip(u_pid + delta, self.u_min, self.u_max))
      du = float(np.clip(u - u_prev, -du_max_eff, du_max_eff))
      u = float(np.clip(u_prev + du, self.u_min, self.u_max))
      delta_real = float(u - u_pid)

      x_next = A_mats[k] @ x + B_vecs[k] * delta_real + c_vecs[k]
      dy_abs = float(x[1] + ref_dy[k])
      dy_abs_next = float(x_next[1] + ref_dy[min(k + 1, H - 1)])
      ddy = (dy_abs_next - dy_abs) / self.dt

      ratio = float(min(abs(du) / (du_max_eff + 1e-9), 0.999))
      sat = max(0.0, abs(u) - max(abs(self.u_min), abs(self.u_max)))
      barrier = (ratio ** 6) / (1.0 - ratio + 1e-3) + 10.0 * sat * sat

      cost += (
        track_w * float(x_next[0] * x_next[0])
        + dy_w * float(x_next[1] * x_next[1])
        + ddy_w * float(ddy * ddy)
        + du_w * float(du * du)
        + delta_w * float(delta_real * delta_real)
        + barrier_w * float(barrier)
      )

      x = x_next
      u_prev = u
      u_seq[k] = u
      d_seq[k] = delta_real

    cost += weights["term_w"] * (
      track_w * float(x[0] * x[0]) + dy_w * float(x[1] * x[1])
    )
    return float(cost), u_seq, d_seq

  def _solve_lqr(self, A_mats: np.ndarray, B_vecs: np.ndarray, c_vecs: np.ndarray, weights: Dict[str, float]):
    H = A_mats.shape[0]
    K = np.zeros((H, 2), dtype=np.float64)
    kff = np.zeros(H, dtype=np.float64)

    Q = np.diag([weights["track_w"], weights["dy_w"]])
    R = float(weights["du_w"] + weights["delta_w"] + 1e-6)

    P = float(weights["term_w"]) * np.diag([weights["track_w"], weights["dy_w"]])
    p = np.zeros(2, dtype=np.float64)

    for k in range(H - 1, -1, -1):
      A = A_mats[k]
      B = B_vecs[k].reshape(2, 1)
      c = c_vecs[k]
      Pn = P
      pn = p

      Hs = float(R + (B.T @ Pn @ B)[0, 0])
      G = (B.T @ Pn @ A).reshape(2)
      g = float(np.asarray(B.T @ (Pn @ c + pn)).reshape(-1)[0])

      K[k] = G / Hs
      kff[k] = g / Hs

      P = Q + A.T @ Pn @ A - np.outer(G, G) / Hs
      p = A.T @ (Pn @ c + pn) - G * (g / Hs)

    return K, kff

  def _build_stage_dynamics(self, H: int, tgt_seq: np.ndarray, ref_dy: np.ndarray, v_seq: np.ndarray, roll_seq: np.ndarray, u_pid: float):
    A_mats = np.zeros((H, 2, 2), dtype=np.float64)
    B_vecs = np.zeros((H, 2), dtype=np.float64)
    c_vecs = np.zeros((H, 2), dtype=np.float64)
    stage_bins = []

    for k in range(H):
      y_model, dy_model, bins = self._select_coeffs(float(v_seq[k]), float(roll_seq[k]))
      A = float(np.clip(dy_model["A"], 0.0, 0.97))
      By, B_raw, B_signed, sign_flip = self._effective_control_gain(y_model, dy_model, float(v_seq[k]))
      B = float(np.clip(B_signed, -3.0, 3.0))
      if abs(B) < 0.01:
        if abs(By) > 1e-5:
          B = 0.01 * float(np.sign(By))
        else:
          B = 0.01 if B >= 0.0 else -0.01
      C = float(np.clip(dy_model["C"], -5.0, 5.0))
      D = float(np.clip(dy_model["D"], -5.0, 5.0))

      k1 = min(k + 1, H - 1)
      c1 = float(tgt_seq[k] + self.dt * ref_dy[k] - tgt_seq[k1])
      c2 = float(A * ref_dy[k] + B * u_pid + C * roll_seq[k] + D + self.fast_bias + self.slow_bias - ref_dy[k1])

      A_mats[k] = np.array([[1.0, self.dt], [0.0, A]], dtype=np.float64)
      B_vecs[k] = np.array([0.0, B], dtype=np.float64)
      c_vecs[k] = np.array([c1, c2], dtype=np.float64)
      stage_bins.append((bins, A, B, C, D, By, B_raw, sign_flip))

    return A_mats, B_vecs, c_vecs, stage_bins

  def _one_step_cost_metrics(self, lat: float, dy: float, roll: float, target_next: float, u_pid: float, u_mpc: float, model_step: Dict[str, float], prev_u: float):
    dy_next_pid, y_next_pid = self._predict_dy_y_next(lat, dy, roll, u_pid, model_step)
    dy_next_mpc, y_next_mpc = self._predict_dy_y_next(lat, dy, roll, u_mpc, model_step)

    err_pid = float(abs(y_next_pid - target_next))
    err_mpc = float(abs(y_next_mpc - target_next))

    c_pid = float((y_next_pid - target_next) ** 2 + 0.10 * (u_pid - prev_u) ** 2)
    c_mpc = float((y_next_mpc - target_next) ** 2 + 0.10 * (u_mpc - prev_u) ** 2)

    return {
      "dy_next_pid": dy_next_pid,
      "dy_next_mpc": dy_next_mpc,
      "y_next_pid": y_next_pid,
      "y_next_mpc": y_next_mpc,
      "err_pid": err_pid,
      "err_mpc": err_mpc,
      "c_pid": c_pid,
      "c_mpc": c_mpc,
    }

  def _prefix_sequence_cost(
    self,
    x0: np.ndarray,
    u_seq: np.ndarray,
    u_pid: float,
    prev_u: float,
    du_max_eff: float,
    A_mats: np.ndarray,
    B_vecs: np.ndarray,
    c_vecs: np.ndarray,
    ref_dy: np.ndarray,
    weights: Dict[str, float],
    steps: int,
  ) -> float:
    x = x0.copy()
    start_prev_u = float(prev_u)
    u_prev = float(prev_u)
    cost = 0.0
    H = min(int(steps), int(len(u_seq)), int(A_mats.shape[0]))

    for k in range(H):
      u = float(np.clip(float(u_seq[k]), self.u_min, self.u_max))
      du = float(np.clip(u - u_prev, -du_max_eff, du_max_eff))
      u = float(np.clip(u_prev + du, self.u_min, self.u_max))
      delta_real = float(u - u_pid)

      x_next = A_mats[k] @ x + B_vecs[k] * delta_real + c_vecs[k]
      dy_abs = float(x[1] + ref_dy[k])
      dy_abs_next = float(x_next[1] + ref_dy[min(k + 1, len(ref_dy) - 1)])
      ddy = (dy_abs_next - dy_abs) / self.dt

      ratio = float(min(abs(du) / (du_max_eff + 1e-9), 0.999))
      sat = max(0.0, abs(u) - max(abs(self.u_min), abs(self.u_max)))
      barrier = (ratio ** 6) / (1.0 - ratio + 1e-3) + 10.0 * sat * sat

      cost += (
        weights["track_w"] * float(x_next[0] * x_next[0])
        + weights["dy_w"] * float(x_next[1] * x_next[1])
        + weights["ddy_w"] * float(ddy * ddy)
        + weights["du_w"] * float(du * du)
        + weights["delta_w"] * float(delta_real * delta_real)
        + weights["barrier_w"] * float(barrier)
      )

      x = x_next
      u_prev = u

    cost += 0.35 * weights["term_w"] * (
      weights["track_w"] * float(x[0] * x[0]) + weights["dy_w"] * float(x[1] * x[1])
    )
    return float(cost)

  def _score_candidate_plan(
    self,
    x0: np.ndarray,
    u_seq: np.ndarray,
    u_pid: float,
    prev_u: float,
    du_max_eff: float,
    A_mats: np.ndarray,
    B_vecs: np.ndarray,
    c_vecs: np.ndarray,
    ref_dy: np.ndarray,
    weights: Dict[str, float],
  ):
    x = x0.copy()
    u_prev = float(prev_u)
    err_sq_sum = 0.0
    jerk_sum = 0.0
    steer_rate_sum = 0.0
    oscillation_penalty = 0.0
    saturation_penalty = 0.0
    max_err = 0.0
    max_dy_abs = 0.0
    max_du = 0.0
    err_sq_hist = []
    du_hist = []
    sat_ref = max(abs(self.u_min), abs(self.u_max)) + 1e-9

    H = min(int(len(u_seq)), int(A_mats.shape[0]))
    for k in range(H):
      u = float(np.clip(float(u_seq[k]), self.u_min, self.u_max))
      du = float(np.clip(u - u_prev, -du_max_eff, du_max_eff))
      u = float(np.clip(u_prev + du, self.u_min, self.u_max))
      delta_real = float(u - u_pid)

      x_next = A_mats[k] @ x + B_vecs[k] * delta_real + c_vecs[k]
      dy_abs = float(x[1] + ref_dy[k])
      dy_abs_next = float(x_next[1] + ref_dy[min(k + 1, len(ref_dy) - 1)])
      ddy = float((dy_abs_next - dy_abs) / self.dt)
      err_sq = float(x_next[0] * x_next[0])

      err_sq_sum += err_sq
      jerk_sum += float(ddy * ddy)
      steer_rate_sum += float(du * du)
      err_sq_hist.append(err_sq)
      du_hist.append(du)
      max_err = max(max_err, abs(float(x_next[0])))
      max_dy_abs = max(max_dy_abs, abs(dy_abs_next))
      max_du = max(max_du, abs(du))

      sat_ratio = max(0.0, (abs(u) - 0.82 * sat_ref) / (0.18 * sat_ref))
      saturation_penalty += float(sat_ratio * sat_ratio)

      x = x_next
      u_prev = u

    for i in range(1, len(du_hist)):
      if abs(du_hist[i - 1]) > 1e-6 and abs(du_hist[i]) > 1e-6 and du_hist[i - 1] * du_hist[i] < 0.0:
        oscillation_penalty += 1.0
    oscillation_penalty = float(oscillation_penalty / max(1, len(du_hist) - 1))

    tail_n = max(1, min(4, len(err_sq_hist)))
    tail_risk = float(max(err_sq_hist[-tail_n:])) if err_sq_hist else 0.0
    predicted_lateral_error = float(err_sq_sum / max(1, H))
    jerk = float(jerk_sum / max(1, H))
    steering_rate = float(steer_rate_sum / max(1, H))
    saturation_penalty = float(saturation_penalty / max(1, H))
    if err_sq_hist:
      head_n = min(3, len(err_sq_hist))
      head_mean = float(np.mean(np.asarray(err_sq_hist[:head_n], dtype=np.float64)))
      tail_mean = float(np.mean(np.asarray(err_sq_hist[-head_n:], dtype=np.float64)))
      consistency_penalty = float(max(0.0, head_mean - tail_mean) + abs(head_mean - predicted_lateral_error))
    else:
      consistency_penalty = 0.0

    score = float(
      self.candidate_score_track_w * predicted_lateral_error
      + self.candidate_score_jerk_w * jerk
      + self.candidate_score_rate_w * steering_rate
      + self.candidate_score_tail_w * tail_risk
      + self.candidate_score_osc_w * oscillation_penalty
      + self.candidate_score_sat_w * saturation_penalty
      + self.candidate_score_consistency_w * consistency_penalty
    )
    return {
      "score": score,
      "predicted_lateral_error": predicted_lateral_error,
      "jerk": jerk,
      "steering_rate": steering_rate,
      "tail_risk": tail_risk,
      "oscillation_penalty": float(oscillation_penalty),
      "saturation_penalty": float(saturation_penalty),
      "consistency_penalty": float(consistency_penalty),
      "max_err": float(max_err),
      "max_dy_abs": float(max_dy_abs),
      "max_du": float(max_du),
    }

  def _generate_mpc_candidates(
    self,
    x0: np.ndarray,
    u_pid: float,
    prev_u: float,
    du_max_eff: float,
    A_mats: np.ndarray,
    B_vecs: np.ndarray,
    c_vecs: np.ndarray,
    ref_dy: np.ndarray,
    weights: Dict[str, float],
    H: int,
    gate_regime: int,
  ):
    def scaled(base: Dict[str, float], track: float, dy: float, ddy: float, du: float, delta: float, barrier: float, term: float):
      return {
        "track_w": float(base["track_w"] * track),
        "dy_w": float(base["dy_w"] * dy),
        "ddy_w": float(base["ddy_w"] * ddy),
        "du_w": float(base["du_w"] * du),
        "delta_w": float(base["delta_w"] * delta),
        "barrier_w": float(base["barrier_w"] * barrier),
        "term_w": float(base["term_w"] * term),
      }

    def evaluate_sequence(family_name: str, cand_weights: Dict[str, float], u_seq: np.ndarray):
      summary = self._score_candidate_plan(
        x0=x0,
        u_seq=u_seq,
        u_pid=u_pid,
        prev_u=prev_u,
        du_max_eff=du_max_eff,
        A_mats=A_mats,
        B_vecs=B_vecs,
        c_vecs=c_vecs,
        ref_dy=ref_dy,
        weights=cand_weights,
      )
      summary["family"] = family_name
      summary["u_seq"] = np.asarray(u_seq, dtype=np.float64)
      summary["weights"] = cand_weights
      return summary

    candidates = []
    pid_seq = np.full(H, u_pid, dtype=np.float64)
    pid_summary = evaluate_sequence("pid", weights, pid_seq)
    candidates.append(pid_summary)

    ff_scale = 0.0
    if abs(B_vecs[0, 1]) > 1e-6:
      ff_scale = float(np.clip(-0.65 * x0[0] / B_vecs[0, 1], -self.delta_u_max_abs, self.delta_u_max_abs))
    for alpha in (0.15, 0.30, 0.45):
      pid_ff_seq = np.full(H, u_pid, dtype=np.float64)
      for k in range(H):
        pid_ff_seq[k] = float(np.clip(u_pid + alpha * ff_scale * np.exp(-0.65 * k), self.u_min, self.u_max))
      summary = evaluate_sequence("pid_ff", scaled(weights, 1.00, 1.00, 1.05, 0.95, 0.95, 1.00, 1.00), pid_ff_seq)
      summary["alpha"] = alpha
      candidates.append(summary)

    family_specs = [
      ("short_predictive", scaled(weights, 1.05, 1.05, 0.90, 0.90, 0.90, 0.95, 0.95), max(6, min(H, 8)), (0.15, 0.30, 0.45)),
      ("medium_predictive", weights, H, (0.20, 0.40, 0.62)),
      ("damped_recovery", scaled(weights, 0.95, 1.30, 1.85, 1.35, 1.35, 1.20, 1.10), H, (0.10, 0.20, 0.32)),
      ("low_jerk", scaled(weights, 0.88, 1.18, 2.35, 1.70, 1.70, 1.10, 1.08), H, (0.08, 0.16, 0.24)),
      ("aggressive_catchup", scaled(weights, 1.40, 0.92, 0.60, 0.62, 0.75, 0.90, 0.95), H, (0.30, 0.55, 0.78)),
    ]

    for family_name, cand_weights, local_H, alpha_list in family_specs:
      K, kff = self._solve_lqr(A_mats[:local_H], B_vecs[:local_H], c_vecs[:local_H], cand_weights)
      _, cand_u_seq, _ = self._forward_plan(
        x0=x0,
        u_pid=u_pid,
        du_max_eff=du_max_eff,
        A_mats=A_mats[:local_H],
        B_vecs=B_vecs[:local_H],
        c_vecs=c_vecs[:local_H],
        ref_dy=ref_dy[:local_H],
        K=K,
        kff=kff,
        weights=cand_weights,
        delta_overrides={},
      )
      base_full_seq = np.full(H, float(cand_u_seq[-1] if len(cand_u_seq) else u_pid), dtype=np.float64)
      base_full_seq[:local_H] = cand_u_seq[:local_H]

      seq_variants = []
      if family_name in ("medium_predictive", "aggressive_catchup") and self.polish_amp > 0.0 and local_H >= 2:
        p0 = float(min(self.polish_amp, 0.90 * du_max_eff))
        p1 = 0.55 * p0
        for d0, d1 in [
          (0.0, 0.0), (-p0, 0.0), (p0, 0.0),
          (0.0, -p0), (0.0, p0),
          (-p1, 0.0), (p1, 0.0), (0.0, -p1), (0.0, p1),
        ]:
          _, alt_seq, _ = self._forward_plan(
            x0=x0,
            u_pid=u_pid,
            du_max_eff=du_max_eff,
            A_mats=A_mats[:local_H],
            B_vecs=B_vecs[:local_H],
            c_vecs=c_vecs[:local_H],
            ref_dy=ref_dy[:local_H],
            K=K,
            kff=kff,
            weights=cand_weights,
            delta_overrides={0: d0, 1: d1},
          )
          cand_full = np.full(H, float(alt_seq[-1] if len(alt_seq) else u_pid), dtype=np.float64)
          cand_full[:local_H] = alt_seq[:local_H]
          seq_variants.append(cand_full)
      else:
        seq_variants.append(base_full_seq)

      for alpha in alpha_list:
        for seq_variant in seq_variants:
          cand_full = np.asarray(u_pid + alpha * (seq_variant - u_pid), dtype=np.float64)
          cand_summary = evaluate_sequence(family_name, cand_weights, cand_full)
          cand_summary["alpha"] = alpha
          candidates.append(cand_summary)

    jerk_limit = max(1e-6, pid_summary["jerk"] * self.candidate_filter_jerk_scale)
    err_limit = max(0.30, pid_summary["max_err"] * self.candidate_filter_err_scale + 0.05)
    du_limit = float(1.08 * du_max_eff)
    tail_limit = max(1e-6, pid_summary["tail_risk"] * (1.0 + 0.65 * (self.candidate_filter_err_scale - 1.0)) + 0.02)

    filtered = []
    for summary in candidates:
      if summary["family"] == "pid":
        filtered.append(summary)
        continue
      reject = False
      if summary["jerk"] > jerk_limit and gate_regime <= 1:
        reject = True
      if summary["max_err"] > err_limit and gate_regime <= 1:
        reject = True
      if summary["max_dy_abs"] > self.candidate_filter_dy_abs and gate_regime <= 1:
        reject = True
      if summary["max_du"] > du_limit and gate_regime <= 1:
        reject = True
      if summary["tail_risk"] > tail_limit and gate_regime <= 1:
        reject = True
      if not reject:
        filtered.append(summary)

    if not filtered:
      filtered = [pid_summary]

    filtered.sort(key=lambda item: item["score"])
    non_pid = [item for item in filtered if item.get("family") != "pid"]
    best = non_pid[0] if non_pid else filtered[0]
    return best, filtered

  def _candidate_consensus(self, candidate_pool, u_pid: float):
    deltas = []
    for cand in candidate_pool:
      family = str(cand.get("family", ""))
      if family == "pid":
        continue
      u_seq = np.asarray(cand.get("u_seq", []), dtype=np.float64)
      if u_seq.size == 0:
        continue
      deltas.append(float(u_seq[0] - u_pid))

    if not deltas:
      return 1.0, 0.0

    arr = np.asarray(deltas, dtype=np.float64)
    spread = float(np.std(arr))
    nonzero = arr[np.abs(arr) > 1e-6]
    sign_consensus = 1.0 if nonzero.size == 0 else float(abs(np.mean(np.sign(nonzero))))
    median = float(np.median(arr))
    close = float(np.mean(np.exp(-np.abs(arr - median) / max(1e-3, 0.20 * self.max_mpc_delta_vs_pid + 1e-3))))
    spread_guard = float(np.exp(-spread / max(1e-3, 0.35 * self.max_mpc_delta_vs_pid + 1e-3)))
    agreement = float(np.clip(0.50 * sign_consensus + 0.30 * close + 0.20 * spread_guard, 0.0, 1.0))
    return agreement, spread

  def _compute_trust(
    self,
    score_relative: float,
    score_pid: float,
    pred_advantage: float,
    cost_pid_pred: float,
    one_step: Dict[str, float],
    family_agreement: float,
    family_quality: float,
    tail_penalty: float,
    cand_delta: float,
    delta_limit: float,
    risk: float,
    tail_mode: bool,
    recovering: bool,
    hard_reject: bool,
  ):
    score_ref = float(max(0.05, abs(score_pid)))
    relative_norm = float(score_relative / score_ref)
    pred_norm = float(pred_advantage / max(1e-6, cost_pid_pred))
    err_improve = float((one_step["err_pid"] - one_step["err_mpc"]) / max(0.03, one_step["err_pid"] + 0.02))
    dy_penalty = float(max(0.0, abs(one_step["dy_next_mpc"]) - abs(one_step["dy_next_pid"])))
    magnitude_term = float(np.exp(-abs(cand_delta) / max(1e-6, 1.10 * delta_limit + 1e-4)))
    dy_term = float(np.exp(-6.5 * dy_penalty))
    tail_term = float(np.exp(-self.tail_penalty_weight * max(0.0, tail_penalty)))
    family_term = float(np.clip(0.35 + 0.65 * max(0.0, family_quality), 0.0, 1.0))
    agreement_term = float(np.clip(0.30 + 0.70 * family_agreement, 0.0, 1.0))

    if score_relative <= 0.0:
      if (
        not hard_reject
        and not tail_mode
        and not recovering
        and score_relative > -self.relative_safe_band * score_ref
        and tail_penalty < 0.01
        and abs(cand_delta) < 0.30 * delta_limit
        and risk < 0.25
        and family_quality >= 0.0
      ):
        tiny_trust = 0.04 * magnitude_term * dy_term * agreement_term
        return float(np.clip(tiny_trust, 0.0, 0.06)), float(np.clip(tiny_trust, 0.0, 0.06)), relative_norm
      return 0.0, 0.0, relative_norm

    relative_term = _sigmoid((relative_norm - self.relative_score_threshold) / self.trust_quality_scale)
    pred_term = _sigmoid(pred_norm / max(0.02, 1.4 * self.trust_quality_scale))
    err_term = _sigmoid(err_improve / 0.10)
    trust_before_tail = float(np.clip(
      (0.48 * relative_term + 0.24 * pred_term + 0.18 * err_term + 0.10 * max(0.0, self.consistency_state))
      * agreement_term
      * family_term
      * magnitude_term
      * dy_term
      * tail_term,
      0.0,
      1.0,
    ))

    tail_guard = 1.0
    if tail_mode:
      tail_guard *= max(0.05, 1.0 - self.trust_tail_scale * (0.45 + 0.55 * risk))
    else:
      tail_guard *= max(0.30, 1.0 - 0.55 * max(0.0, risk - 0.55) / 0.45)
    if recovering:
      tail_guard *= 0.55

    trust_after_tail = float(np.clip(trust_before_tail * tail_guard, 0.0, 1.0))
    if self.strict_safe_mode:
      trust_after_tail = 1.0 if (not hard_reject and trust_after_tail >= self.trust_strict_threshold) else 0.0
      trust_before_tail = trust_after_tail

    return trust_before_tail, trust_after_tail, relative_norm

  def _apply_final_limits(self, u_selected: float, du_cap: float, control_active: bool):
    action = float(np.clip(u_selected, self.u_min, self.u_max))
    if control_active and self.prev_action_applied:
      action = float(np.clip(action, self.last_action - du_cap, self.last_action + du_cap))
      action = float(np.clip(action, self.u_min, self.u_max))
    return action

  def get_stats(self):
    if self.bc_preview_mode and self.bc_preview_steps > 0:
      n = max(1, self.bc_preview_steps)
      return {
        "bc_preview_mode": 1.0,
        "mean_abs_delta": float(self.bc_preview_abs_delta_sum / n),
        "blend_rate": 1.0,
      }
    if self.analytic_preview_mode and self.analytic_preview_steps > 0:
      n = max(1, self.analytic_preview_steps)
      return {
        "analytic_preview_mode": 1.0,
        "mean_abs_delta": float(self.analytic_preview_abs_delta_sum / n),
        "analytic_preview_gain_mean": float(self.analytic_preview_gain_sum / n),
        "blend_rate": 1.0,
      }
    if self.preview_pid_mode and self.preview_pid_steps > 0:
      n = max(1, self.preview_pid_steps)
      return {
        "preview_pid_mode": 1.0,
        "mean_abs_delta": float(self.preview_pid_abs_delta_sum / n),
        "blend_rate": 1.0,
      }
    if self.aggressive_horizon_mode and self.aggressive_steps > 0:
      n = max(1, self.aggressive_steps)
      return {
        "aggressive_horizon_mode": 1.0,
        "mean_mix_used": float(self.aggressive_mix_used_sum / n),
        "mean_abs_delta": float(self.aggressive_abs_delta_sum / n),
        "delta_clip_rate": float(self.aggressive_delta_clip_count / n),
        "blend_rate": float(self.aggressive_interventions / n),
        "aggressive_improve_mean": float(self.aggressive_improve_sum / n),
      }
    if self.known_good_fixed_mix_mode and self.known_good_steps > 0:
      n = max(1, self.known_good_steps)
      return {
        "simple_mode_active": 1.0,
        "known_good_fixed_mix_mode": 1.0,
        "simple_veto_mode": float(self.simple_veto_mode),
        "fixed_mix_tail_veto_enable": float(self.fixed_mix_tail_veto_enable),
        "fixed_mix_emergency_fallback_enable": float(self.fixed_mix_emergency_fallback_enable),
        "mean_mix_used": float(self.known_good_mix_used_sum / n),
        "mean_abs_delta": float(self.known_good_abs_delta_sum / n),
        "delta_clip_rate": float(self.known_good_delta_clip_count / n),
        "accepted_interventions_per_rollout": float(self.known_good_interventions),
        "blend_rate": float(self.known_good_interventions / n),
        "post_residual_rate": float(self.known_good_post_residual_count / n),
        "post_residual_mean_abs": float(self.known_good_post_residual_abs_sum / n),
        "fixed_mix_emergency_fallback_rate": float(self.fixed_mix_emergency_fallback_count / n),
        "fixed_mix_emergency_active_rate": float(self.fixed_mix_emergency_active_count / n),
        "fixed_mix_emergency_trigger_err_rate": float(self.fixed_mix_emergency_trigger_err_count / n),
        "fixed_mix_emergency_trigger_dy_rate": float(self.fixed_mix_emergency_trigger_dy_count / n),
        "fixed_mix_emergency_trigger_rise_rate": float(self.fixed_mix_emergency_trigger_rise_count / n),
        "fixed_mix_tail_veto_rate": float(self.fixed_mix_tail_veto_count / n),
        "fixed_mix_sign_veto_rate": float(self.fixed_mix_sign_veto_count / n),
        "fixed_mix_err_veto_rate": float(self.fixed_mix_err_veto_count / n),
        "fixed_mix_dy_veto_rate": float(self.fixed_mix_dy_veto_count / n),
        "fixed_mix_delta_veto_rate": float(self.fixed_mix_delta_veto_count / n),
        "veto_rate": float(self.known_good_veto_count / n),
        "harm_veto_rate": float(self.known_good_harm_veto_count / n),
        "sign_veto_rate": float(self.known_good_sign_veto_count / n),
        "flip_veto_rate": float(self.known_good_flip_veto_count / n),
        "hold_veto_rate": float(self.known_good_hold_veto_count / n),
        "reentry_rate": float(self.known_good_reentry_count / n),
        "sign_disagree_rate": float(self.known_good_sign_disagree_count / n),
        "large_delta_rate": float(self.known_good_large_delta_count / n),
        "recent_harm_rate": float(self.known_good_recent_harm_count / n),
        "veto_err_rate": float(self.known_good_err_veto_count / n),
        "veto_dy_rate": float(self.known_good_dy_veto_count / n),
        "veto_flip_rate": float(self.known_good_flip_veto_count / n),
        "veto_sign_disagree_rate": float(self.known_good_sign_veto_count / n),
        "veto_large_delta_rate": float(self.known_good_hold_veto_count / n),
        "veto_recent_harm_rate": float(self.known_good_harm_veto_count / n),
        "advanced_block_touched_rate": float(self.known_good_advanced_touch_count / n),
        "pid_exact_hard_bypass_active": float(self.pid_exact_hard_bypass_steps > 0),
      }

    if self.simple_hybrid_mode and self.simple_mode_steps > 0:
      n = max(1, self.simple_mode_steps)
      return {
        "simple_mode_active": 1.0,
        "mean_mix_used": float(self.mix_used_sum / n),
        "mean_abs_delta": float(self.accepted_delta_sum / n),
        "delta_clip_rate": float(self.delta_clip_count / n),
        "accepted_interventions_per_rollout": float(self.simple_accept_count),
        "calm_accept_rate": float(self.simple_calm_accept_count / max(1, self.simple_calm_gate_count)),
        "mild_accept_rate": float(self.simple_mild_accept_count / max(1, self.simple_mild_gate_count)),
        "local_adv_pass_rate": float(self.simple_local_adv_pass_count / max(1, self.simple_predictive_considered_count)),
        "marginal_accept_rate": float(self.simple_marginal_accept_count / max(1, self.simple_accept_count)),
        "high_conviction_accept_rate": float(self.simple_high_conviction_accept_count / max(1, self.simple_accept_count)),
        "mean_local_advantage_for_accepted_steps": float(self.simple_local_adv_sum / max(1, self.simple_accept_count)),
        "worst_accepted_step_local_disadvantage": float(self.simple_worst_accepted_local_adv if self.simple_accept_count > 0 else 0.0),
        "veto_err_growth_rate": float(self.simple_veto_err_growth / n),
        "veto_dy_growth_rate": float(self.simple_veto_dy_growth / n),
        "veto_oscillation_rate": float(self.simple_veto_oscillation / n),
        "veto_recovery_trend_rate": float(self.simple_veto_recovery_trend / n),
        "veto_target_flip_rate": float(self.simple_veto_target_flip / n),
        "veto_limit_rate": float(self.simple_veto_limit / n),
        "veto_must_help_rate": float(self.simple_veto_must_help / n),
        "veto_budget_rate": float(self.simple_veto_budget / n),
        "cumulative_signed_delta": float(self.simple_signed_delta_sum),
        "max_abs_delta": float(self.simple_max_abs_delta),
        "accepted_step_indices": list(self.simple_accepted_steps),
      }

    n = max(1, self.step_count)
    acc_n = max(1, self.accept_count)
    mean_mix = float(self.mix_sum / n)
    mix_std = float(np.sqrt(max(0.0, self.mix_sq_sum / n - mean_mix * mean_mix)))
    mean_mix_used = float(self.mix_used_sum / n)
    mix_used_std = float(np.sqrt(max(0.0, self.mix_used_sq_sum / n - mean_mix_used * mean_mix_used)))
    if self.accepted_adv_values:
      adv_arr = np.asarray(self.accepted_adv_values, dtype=np.float64)
      adv_p10 = float(np.percentile(adv_arr, 10))
      adv_p50 = float(np.percentile(adv_arr, 50))
      adv_p90 = float(np.percentile(adv_arr, 90))
    else:
      adv_p10 = 0.0
      adv_p50 = 0.0
      adv_p90 = 0.0
    if self.candidate_quality_values:
      cq_arr = np.asarray(self.candidate_quality_values, dtype=np.float64)
      cq_p50 = float(np.percentile(cq_arr, 50))
      cq_p90 = float(np.percentile(cq_arr, 90))
    else:
      cq_p50 = 0.0
      cq_p90 = 0.0
    if self.trust_values:
      trust_arr = np.asarray(self.trust_values, dtype=np.float64)
      trust_p50 = float(np.percentile(trust_arr, 50))
      trust_p90 = float(np.percentile(trust_arr, 90))
    else:
      trust_p50 = 0.0
      trust_p90 = 0.0
    if self.score_relative_values:
      sr_arr = np.asarray(self.score_relative_values, dtype=np.float64)
      sr_p50 = float(np.percentile(sr_arr, 50))
      sr_p90 = float(np.percentile(sr_arr, 90))
    else:
      sr_p50 = 0.0
      sr_p90 = 0.0
    if self.accepted_score_relative_values:
      asr_arr = np.asarray(self.accepted_score_relative_values, dtype=np.float64)
      asr_p50 = float(np.percentile(asr_arr, 50))
    else:
      asr_p50 = 0.0
    return {
      "accept_rate": float(self.accept_count / n),
      "hard_reject_rate": float(self.hard_reject_count / n),
      "cost_reject_rate": float(self.cost_reject_count / n),
      "soft_cost_reject_rate": float(self.cost_reject_count / n),
      "sign_reject_rate": float(self.sign_reject_count / n),
      "magnitude_reject_rate": float(self.magnitude_reject_count / n),
      "tail_reject_rate": float(self.tail_reject_count / n),
      "recovery_hold_rate": float(self.recovery_hold_active_count / n),
      "mean_abs_delta": float(self.accepted_delta_sum / acc_n),
      "mean_err_improve": float(self.accepted_err_improve_sum / acc_n),
      "accepted_advantage_mean": float(self.accepted_adv_sum / acc_n),
      "accepted_advantage_p10": adv_p10,
      "accepted_advantage_p50": adv_p50,
      "accepted_advantage_p90": adv_p90,
      "tail_rate": float(self.tail_forced_count / n),
      "delta_clip_rate": float(self.delta_clip_count / n),
      "blend_rate": float(self.blend_count / n),
      "mean_mix": mean_mix,
      "mix_std": mix_std,
      "mean_mix_used": mean_mix_used,
      "mix_used_std": mix_used_std,
      "trust_ramp_mean": float(self.trust_ramp_sum / n),
      "consistency_mean": float(self.consistency_sum / n),
      "candidate_quality_mean": float(self.candidate_quality_sum / n),
      "candidate_quality_p50": cq_p50,
      "candidate_quality_p90": cq_p90,
      "candidate_count": float(self.candidate_count_sum / n),
      "best_candidate_score": float(self.best_candidate_score_sum / n),
      "tail_risk_mean": float(self.tail_risk_sum / n),
      "trust_mean": float(self.trust_after_tail_sum / n),
      "trust_p50": trust_p50,
      "trust_p90": trust_p90,
      "trust_before_tail_mean": float(self.trust_before_tail_sum / n),
      "trust_after_tail_mean": float(self.trust_after_tail_sum / n),
      "family_agreement_mean": float(self.family_agreement_sum / n),
      "candidate_spread_mean": float(self.candidate_spread_sum / n),
      "score_pid_mean": float(self.score_pid_sum / n),
      "score_best_mean": float(self.score_best_sum / n),
      "score_gap_mean": float(self.score_gap_sum / n),
      "score_relative_mean": float(self.score_relative_sum / n),
      "score_relative_p50": sr_p50,
      "score_relative_p90": sr_p90,
      "accepted_score_relative_mean": float(self.accepted_score_relative_sum / acc_n),
      "accepted_score_relative_p50": asr_p50,
      "trust_when_relative_positive": float(self.trust_positive_sum / max(1.0, self.trust_positive_count)),
      "trust_when_relative_negative": float(self.trust_negative_sum / max(1.0, self.trust_negative_count)),
      "tail_penalty_mean": float(self.tail_penalty_sum / n),
      "acceptance_rate_after_multicandidate": float(self.accept_count / n),
      "candidate_family_pid_rate": float(self.candidate_family_counts["pid"] / n),
      "candidate_family_pid_ff_rate": float(self.candidate_family_counts["pid_ff"] / n),
      "candidate_family_short_predictive_rate": float(self.candidate_family_counts["short_predictive"] / n),
      "candidate_family_medium_predictive_rate": float(self.candidate_family_counts["medium_predictive"] / n),
      "candidate_family_damped_recovery_rate": float(self.candidate_family_counts["damped_recovery"] / n),
      "candidate_family_low_jerk_rate": float(self.candidate_family_counts["low_jerk"] / n),
      "candidate_family_aggressive_catchup_rate": float(self.candidate_family_counts["aggressive_catchup"] / n),
      "mpc_rate": float(self.blend_count / n),
      "regime0_rate": float(self.regime_counts[0] / n),
      "regime1_rate": float(self.regime_counts[1] / n),
      "regime2_rate": float(self.regime_counts[2] / n),
      "gate_regime0_rate": float(self.gate_regime_counts[0] / n),
      "gate_regime1_rate": float(self.gate_regime_counts[1] / n),
      "gate_regime2_rate": float(self.gate_regime_counts[2] / n),
      "accept_rate_regime0": float(self.accept_regime_counts[0] / max(1, self.regime_counts[0])),
      "accept_rate_regime1": float(self.accept_regime_counts[1] / max(1, self.regime_counts[1])),
      "accept_rate_regime2": float(self.accept_regime_counts[2] / max(1, self.regime_counts[2])),
      "accept_rate_gate_regime0": float(self.accept_gate_regime_counts[0] / max(1, self.gate_regime_counts[0])),
      "accept_rate_gate_regime1": float(self.accept_gate_regime_counts[1] / max(1, self.gate_regime_counts[1])),
      "accept_rate_gate_regime2": float(self.accept_gate_regime_counts[2] / max(1, self.gate_regime_counts[2])),
      "mix_used_gate_regime0": float(self.mix_used_gate_sum[0] / max(1, self.gate_regime_counts[0])),
      "mix_used_gate_regime1": float(self.mix_used_gate_sum[1] / max(1, self.gate_regime_counts[1])),
      "mix_used_gate_regime2": float(self.mix_used_gate_sum[2] / max(1, self.gate_regime_counts[2])),
      "calm_accept_rate": float(self.accept_gate_regime_counts[0] / max(1, self.gate_regime_counts[0])),
      "transition_accept_rate": float(self.accept_gate_regime_counts[1] / max(1, self.gate_regime_counts[1])),
      "tail_mode_rate": float(self.tail_mode_count / n),
      "recovery_rate": float(self.recovery_count / n),
      "residual_mean_abs": float(self.residual_abs_sum / n),
      "family_quality_ema_pid": float(self.family_quality_ema["pid"]),
      "family_quality_ema_pid_ff": float(self.family_quality_ema["pid_ff"]),
      "family_quality_ema_short_predictive": float(self.family_quality_ema["short_predictive"]),
      "family_quality_ema_medium_predictive": float(self.family_quality_ema["medium_predictive"]),
      "family_quality_ema_damped_recovery": float(self.family_quality_ema["damped_recovery"]),
      "family_quality_ema_low_jerk": float(self.family_quality_ema["low_jerk"]),
      "family_quality_ema_aggressive_catchup": float(self.family_quality_ema["aggressive_catchup"]),
      "family_used_adv_ema_pid": float(self.family_used_adv_ema["pid"]),
      "family_used_adv_ema_pid_ff": float(self.family_used_adv_ema["pid_ff"]),
      "family_used_adv_ema_short_predictive": float(self.family_used_adv_ema["short_predictive"]),
      "family_used_adv_ema_medium_predictive": float(self.family_used_adv_ema["medium_predictive"]),
      "family_used_adv_ema_damped_recovery": float(self.family_used_adv_ema["damped_recovery"]),
      "family_used_adv_ema_low_jerk": float(self.family_used_adv_ema["low_jerk"]),
      "family_used_adv_ema_aggressive_catchup": float(self.family_used_adv_ema["aggressive_catchup"]),
      "accepted_adv_mean_pid": float(self.family_accepted_adv_sum["pid"] / max(1.0, self.family_accepted_count["pid"])),
      "accepted_adv_mean_pid_ff": float(self.family_accepted_adv_sum["pid_ff"] / max(1.0, self.family_accepted_count["pid_ff"])),
      "accepted_adv_mean_short_predictive": float(self.family_accepted_adv_sum["short_predictive"] / max(1.0, self.family_accepted_count["short_predictive"])),
      "accepted_adv_mean_medium_predictive": float(self.family_accepted_adv_sum["medium_predictive"] / max(1.0, self.family_accepted_count["medium_predictive"])),
      "accepted_adv_mean_damped_recovery": float(self.family_accepted_adv_sum["damped_recovery"] / max(1.0, self.family_accepted_count["damped_recovery"])),
      "accepted_adv_mean_low_jerk": float(self.family_accepted_adv_sum["low_jerk"] / max(1.0, self.family_accepted_count["low_jerk"])),
      "accepted_adv_mean_aggressive_catchup": float(self.family_accepted_adv_sum["aggressive_catchup"] / max(1.0, self.family_accepted_count["aggressive_catchup"])),
      "pid_exact_hard_bypass_active": float(self.pid_exact_hard_bypass_steps > 0),
    }

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.call_idx += 1
    control_active = self.call_idx > self.control_delay_steps
    active_step = self.call_idx - self.control_delay_steps

    target_raw = float(target_lataccel)
    lat = float(current_lataccel)

    u_pid = float(self.pid_controller.update(target_raw, lat, state, future_plan))

    if self.bc_preview_mode and self.bc_preview_model is not None:
      feat = self._bc_preview_features(target_raw, state, future_plan)
      ff = self._bc_preview_ff(feat)
      dy_now = 0.0 if self.prev_lat is None else float((lat - self.prev_lat) / self.dt)
      err = float(target_raw - lat)
      ff_blend = float((1.0 - self.bc_preview_pid_mix) * ff + self.bc_preview_pid_mix * u_pid)
      raw_action = float(ff_blend + self.bc_preview_err_gain * err - self.bc_preview_dy_gain * dy_now)
      if self.prev_action_applied:
        raw_action = float(self.bc_preview_smoothing * self.last_action + (1.0 - self.bc_preview_smoothing) * raw_action)
      final_action = self._apply_final_limits(raw_action, self.bc_preview_du_cap, control_active)

      if control_active:
        self.step_count += 1
        self.bc_preview_steps += 1
        self.bc_preview_abs_delta_sum += abs(float(final_action - u_pid))

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source=bc_preview bc_preview_mode=1 "
          f"u_pid_raw={u_pid:.10g} ff={ff:.10g} ff_blend={ff_blend:.10g} err={err:.10g} dy={dy_now:.10g} "
          f"raw_action={raw_action:.10g} final_action={final_action:.10g}"
        )

      self.prev_target = target_raw
      self.prev_target_cond = target_raw
      self.prev_lat = lat
      self.prev_dy = dy_now
      self.dy_est = dy_now
      self.prev_v = float(state.v_ego)
      self.prev_roll = float(state.roll_lataccel)
      self.prev_model = None
      self.error_hist.append(float(err))
      if len(self.error_hist) > 12:
        self.error_hist = self.error_hist[-12:]
      self.action_hist.append(float(final_action))
      if len(self.action_hist) > 12:
        self.action_hist = self.action_hist[-12:]
      if control_active:
        self.last_action = float(final_action)
        self.prev_action_applied = True
      return float(final_action)

    if self.analytic_preview_mode:
      H = int(self.analytic_preview_horizon)
      tgt_seq = self._build_horizon_seq(target_raw, list(getattr(future_plan, "lataccel", [])), H + 1)
      v_seq = self._build_horizon_seq(float(state.v_ego), list(getattr(future_plan, "v_ego", [])), H)
      roll_seq = self._build_horizon_seq(float(state.roll_lataccel), list(getattr(future_plan, "roll_lataccel", [])), H)
      u_anchor = float(self.last_action if self.prev_action_applied else u_pid)

      alpha = 1.0
      beta = 0.0
      gamma = 0.0
      numer = self.analytic_preview_rate_weight * u_anchor
      denom = self.analytic_preview_rate_weight
      decay_w = 1.0
      pred_pid_cost = self.analytic_preview_rate_weight * (u_pid - u_anchor) * (u_pid - u_anchor)
      pred_star_terms = []

      for k in range(H):
        y_model, _, _ = self._select_coeffs(float(v_seq[k]), float(roll_seq[k]))
        a = float(np.clip(y_model["a"], 0.0, 1.2))
        b = float(np.clip(y_model["b0"] + y_model["bv"] * float(np.clip(v_seq[k], 0.0, 50.0)), -5.0, 5.0))
        c_term = float(np.clip(y_model["c"], -5.0, 5.0) * float(roll_seq[k]) + np.clip(y_model["d"], -5.0, 5.0))

        alpha = a * alpha
        beta = a * beta + b
        gamma = a * gamma + c_term
        target_k = float(tgt_seq[k + 1])

        numer += decay_w * beta * (target_k - alpha * lat - gamma)
        denom += decay_w * beta * beta
        pred_pid = alpha * lat + beta * u_pid + gamma
        pred_pid_cost += decay_w * (pred_pid - target_k) * (pred_pid - target_k)
        pred_star_terms.append((decay_w, alpha, beta, gamma, target_k))
        decay_w *= self.analytic_preview_decay

      if denom <= 1e-9:
        u_star = float(u_pid)
        pred_star_cost = pred_pid_cost
      else:
        u_star = float(numer / denom)
        pred_star_cost = self.analytic_preview_rate_weight * (u_star - u_anchor) * (u_star - u_anchor)
        for wk, ak, bk, gk, tk in pred_star_terms:
          pred = ak * lat + bk * u_star + gk
          pred_star_cost += wk * (pred - tk) * (pred - tk)

      raw_action = float(u_star if pred_star_cost <= pred_pid_cost + self.analytic_preview_margin else u_pid)
      final_action = self._apply_final_limits(raw_action, self.analytic_preview_du_cap, control_active)

      if control_active:
        self.step_count += 1
        self.analytic_preview_steps += 1
        self.analytic_preview_abs_delta_sum += abs(float(final_action - u_pid))
        self.analytic_preview_gain_sum += float(pred_pid_cost - pred_star_cost)

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source=analytic_preview analytic_preview_mode=1 "
          f"u_pid_raw={u_pid:.10g} u_anchor={u_anchor:.10g} u_star={u_star:.10g} raw_action={raw_action:.10g} final_action={final_action:.10g} "
          f"pred_pid_cost={pred_pid_cost:.10g} pred_star_cost={pred_star_cost:.10g} "
          f"gain={pred_pid_cost - pred_star_cost:.10g}"
        )

      dy_now = 0.0 if self.prev_lat is None else float((lat - self.prev_lat) / self.dt)
      self.prev_target = target_raw
      self.prev_target_cond = target_raw
      self.prev_lat = lat
      self.prev_dy = dy_now
      self.dy_est = dy_now
      self.prev_v = float(state.v_ego)
      self.prev_roll = float(state.roll_lataccel)
      self.prev_model = None
      self.error_hist.append(float(target_raw - lat))
      if len(self.error_hist) > 12:
        self.error_hist = self.error_hist[-12:]
      self.action_hist.append(float(final_action))
      if len(self.action_hist) > 12:
        self.action_hist = self.action_hist[-12:]
      if control_active:
        self.last_action = float(final_action)
        self.prev_action_applied = True
      return float(final_action)

    if self.preview_pid_mode:
      err = float(target_raw - lat)
      dy_raw = 0.0 if self.prev_lat is None else float((lat - self.prev_lat) / self.dt)

      targets = [target_raw] + list(getattr(future_plan, "lataccel", []))[:max(0, self.preview_pid_horizon - 1)]
      if targets:
        weights = []
        w = 1.0
        for _ in targets:
          weights.append(w)
          w *= self.preview_pid_decay
        preview_target = float(np.dot(np.asarray(targets, dtype=np.float64), np.asarray(weights, dtype=np.float64)) / max(1e-9, np.sum(weights)))
      else:
        preview_target = target_raw

      y_model, _, _ = self._select_coeffs(float(state.v_ego), float(state.roll_lataccel))
      a = float(np.clip(y_model["a"], 0.0, 1.2))
      b = float(np.clip(y_model["b0"] + y_model["bv"] * float(np.clip(state.v_ego, 0.0, 50.0)), -5.0, 5.0))
      c_roll = float(np.clip(y_model["c"], -5.0, 5.0))
      d = float(np.clip(y_model["d"], -5.0, 5.0))

      if abs(b) < 1e-5:
        u_ff = float(u_pid)
      else:
        u_ff = float((preview_target - a * lat - c_roll * float(state.roll_lataccel) - d) / b)

      if abs(err) < self.preview_pid_integral_zone and err * self.preview_pid_prev_err >= -1e-6:
        self.preview_pid_integral = float(
          self.preview_pid_integral_decay * self.preview_pid_integral + err * self.dt
        )
      else:
        self.preview_pid_integral = float(self.preview_pid_integral_decay * self.preview_pid_integral)
      self.preview_pid_integral = float(np.clip(
        self.preview_pid_integral,
        -self.preview_pid_integral_limit,
        self.preview_pid_integral_limit,
      ))

      derr = float((err - self.preview_pid_prev_err) / self.dt) if self.call_idx > 1 else 0.0
      preview_err = float(preview_target - lat)
      action_raw = float(
        self.preview_pid_ff_weight * u_ff
        + self.preview_pid_p * preview_err
        + self.preview_pid_i * self.preview_pid_integral
        + self.preview_pid_d * derr
        - self.preview_pid_dy * dy_raw
      )
      if self.prev_action_applied:
        action_raw = float(self.preview_pid_smoothing * self.last_action + (1.0 - self.preview_pid_smoothing) * action_raw)
      final_action = float(np.clip(action_raw, self.u_min, self.u_max))

      if control_active:
        self.step_count += 1
        self.preview_pid_steps += 1
        self.preview_pid_abs_delta_sum += abs(float(final_action - u_pid))

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source=preview_pid preview_pid_mode=1 "
          f"u_pid_raw={u_pid:.10g} u_ff={u_ff:.10g} preview_target={preview_target:.10g} "
          f"err={err:.10g} dy={dy_raw:.10g} int={self.preview_pid_integral:.10g} derr={derr:.10g} "
          f"action_raw={action_raw:.10g} final_action={final_action:.10g} "
          f"model_a={a:.10g} model_b={b:.10g} model_c={c_roll:.10g} model_d={d:.10g}"
        )

      self.preview_pid_prev_err = float(err)
      self.prev_target = target_raw
      self.prev_target_cond = target_raw
      self.prev_lat = lat
      self.prev_dy = dy_raw
      self.dy_est = dy_raw
      self.prev_v = float(state.v_ego)
      self.prev_roll = float(state.roll_lataccel)
      self.prev_model = None
      self.error_hist.append(float(err))
      if len(self.error_hist) > 12:
        self.error_hist = self.error_hist[-12:]
      self.action_hist.append(float(final_action))
      if len(self.action_hist) > 12:
        self.action_hist = self.action_hist[-12:]
      if control_active:
        self.last_action = float(final_action)
        self.prev_action_applied = True
      return float(final_action)

    if self.aggressive_horizon_mode:
      self._update_dy_estimate(lat)

      H = int(self.aggressive_horizon)
      tgt_seq = self._build_horizon_seq(target_raw, list(getattr(future_plan, "lataccel", [])), H)
      v_seq = self._build_horizon_seq(float(state.v_ego), list(getattr(future_plan, "v_ego", [])), H)
      roll_seq = self._build_horizon_seq(float(state.roll_lataccel), list(getattr(future_plan, "roll_lataccel", [])), H)

      ref_dy = np.zeros(H, dtype=np.float64)
      for k in range(1, H):
        ref_dy[k] = (tgt_seq[k] - tgt_seq[k - 1]) / self.dt

      x0 = np.array([lat - tgt_seq[0], self.dy_est - ref_dy[0]], dtype=np.float64)
      weights = {
        "track_w": float(self.aggressive_track_weight),
        "dy_w": float(self.aggressive_dy_weight),
        "ddy_w": float(self.aggressive_ddy_weight),
        "du_w": float(self.aggressive_du_weight),
        "delta_w": float(self.aggressive_delta_weight),
        "barrier_w": float(self.aggressive_barrier_weight),
        "term_w": float(self.aggressive_terminal_weight),
      }

      A_mats, B_vecs, c_vecs, stage_bins = self._build_stage_dynamics(H, tgt_seq, ref_dy, v_seq, roll_seq, u_pid)
      K, kff = self._solve_lqr(A_mats, B_vecs, c_vecs, weights)

      saved_delta_u_max_abs = self.delta_u_max_abs
      try:
        self.delta_u_max_abs = float(self.aggressive_delta_u_max_abs)
        _, best_u_seq, _ = self._forward_plan(
          x0=x0,
          u_pid=u_pid,
          du_max_eff=self.aggressive_du_max_step,
          A_mats=A_mats,
          B_vecs=B_vecs,
          c_vecs=c_vecs,
          ref_dy=ref_dy,
          K=K,
          kff=kff,
          weights=weights,
          delta_overrides={},
        )

        if self.aggressive_polish_amp > 0.0 and H >= 2:
          p0 = float(min(self.aggressive_polish_amp, 0.90 * self.aggressive_du_max_step))
          p1 = 0.55 * p0
          candidates = [
            (0.0, 0.0), (-p0, 0.0), (p0, 0.0),
            (0.0, -p0), (0.0, p0),
            (-p0, -p0), (-p0, p0), (p0, -p0), (p0, p0),
            (-p1, 0.0), (p1, 0.0), (0.0, -p1), (0.0, p1),
          ]
          best_cost = self._prefix_sequence_cost(
            x0=x0,
            u_seq=best_u_seq,
            u_pid=u_pid,
            prev_u=float(self.last_action if self.prev_action_applied else np.clip(u_pid, self.u_min, self.u_max)),
            du_max_eff=self.aggressive_du_max_step,
            A_mats=A_mats,
            B_vecs=B_vecs,
            c_vecs=c_vecs,
            ref_dy=ref_dy,
            weights=weights,
            steps=self.aggressive_prefix_steps,
          )
          for d0, d1 in candidates:
            cst, u_seq, _ = self._forward_plan(
              x0=x0,
              u_pid=u_pid,
              du_max_eff=self.aggressive_du_max_step,
              A_mats=A_mats,
              B_vecs=B_vecs,
              c_vecs=c_vecs,
              ref_dy=ref_dy,
              K=K,
              kff=kff,
              weights=weights,
              delta_overrides={0: d0, 1: d1},
            )
            cst = self._prefix_sequence_cost(
              x0=x0,
              u_seq=u_seq,
              u_pid=u_pid,
              prev_u=float(self.last_action if self.prev_action_applied else np.clip(u_pid, self.u_min, self.u_max)),
              du_max_eff=self.aggressive_du_max_step,
              A_mats=A_mats,
              B_vecs=B_vecs,
              c_vecs=c_vecs,
              ref_dy=ref_dy,
              weights=weights,
              steps=self.aggressive_prefix_steps,
            )
            if cst < best_cost:
              best_cost = cst
              best_u_seq = u_seq
      finally:
        self.delta_u_max_abs = saved_delta_u_max_abs

      bins0, A0, B0, C0, D0, By0, Braw0, sign_flip0 = stage_bins[0]
      model_step0 = {"A": A0, "B": B0, "C": C0, "D": D0}
      u_mpc_raw = float(best_u_seq[0])
      if self.aggressive_sign_align_enable:
        u_mpc_raw, local_dir, sign_agree, sign_corrected = self._align_candidate_sign(
          lat,
          self.dy_est,
          float(state.roll_lataccel),
          target_raw,
          u_pid,
          u_mpc_raw,
          model_step0,
        )
      else:
        local_dir = {"desired_correction_sign": 0, "one_step_err_if_pid": 0.0}
        sign_agree = 1
        sign_corrected = 0

      u_mpc = float(np.clip(u_mpc_raw, u_pid - self.aggressive_max_delta_vs_pid, u_pid + self.aggressive_max_delta_vs_pid))
      u_mpc = float(np.clip(u_mpc, self.u_min, self.u_max))

      prev_u = float(self.last_action if self.prev_action_applied else np.clip(u_pid, self.u_min, self.u_max))
      pid_seq = np.full(H, float(u_pid), dtype=np.float64)
      mpc_seq = np.asarray(best_u_seq, dtype=np.float64).copy()
      mpc_seq[0] = float(u_mpc)
      pid_cost = self._prefix_sequence_cost(
        x0=x0,
        u_seq=pid_seq,
        u_pid=u_pid,
        prev_u=prev_u,
        du_max_eff=self.aggressive_du_max_step,
        A_mats=A_mats,
        B_vecs=B_vecs,
        c_vecs=c_vecs,
        ref_dy=ref_dy,
        weights=weights,
        steps=self.aggressive_prefix_steps,
      )
      mpc_cost = self._prefix_sequence_cost(
        x0=x0,
        u_seq=mpc_seq,
        u_pid=u_pid,
        prev_u=prev_u,
        du_max_eff=self.aggressive_du_max_step,
        A_mats=A_mats,
        B_vecs=B_vecs,
        c_vecs=c_vecs,
        ref_dy=ref_dy,
        weights=weights,
        steps=self.aggressive_prefix_steps,
      )

      improve = float(pid_cost - mpc_cost)
      improve_ratio = float(improve / max(abs(pid_cost), 1e-6))
      if improve_ratio <= self.aggressive_improve_floor:
        mix_eff = 0.0
      else:
        mix_eff = float(np.clip(
          self.aggressive_base_mix + self.aggressive_mix_gain * (improve_ratio - self.aggressive_improve_floor),
          0.0,
          self.aggressive_mix_max,
        ))

      blend_pre_clip = float(u_pid + mix_eff * (u_mpc - u_pid))
      final_action = self._apply_final_limits(blend_pre_clip, self.aggressive_du_max_step, control_active)
      clip_fallback = bool(self.aggressive_clip_fallback_enable and abs(final_action - blend_pre_clip) > 1e-12)
      if clip_fallback:
        final_action = float(u_pid)
        mix_used = 0.0
      else:
        mix_used = float(mix_eff)
        if abs(u_mpc - u_pid) > 1e-12:
          mix_used = float(np.clip((final_action - u_pid) / (u_mpc - u_pid), 0.0, 1.0))

      intervention_applied = bool(abs(final_action - u_pid) > 1e-12)
      delta_clip_applied = int(abs(final_action - blend_pre_clip) > 1e-12)

      if control_active:
        self.step_count += 1
        self.aggressive_steps += 1
        self.aggressive_interventions += int(intervention_applied)
        self.aggressive_mix_used_sum += mix_used
        self.aggressive_abs_delta_sum += abs(float(final_action - u_pid))
        self.aggressive_delta_clip_count += delta_clip_applied
        self.aggressive_improve_sum += improve_ratio
        self.blend_count += int(intervention_applied)
        self.mix_sum += mix_eff
        self.mix_sq_sum += mix_eff * mix_eff
        self.mix_used_sum += mix_used
        self.mix_used_sq_sum += mix_used * mix_used
        self.accepted_delta_sum += abs(float(final_action - u_pid))
        self.delta_clip_count += delta_clip_applied

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source=aggressive_horizon aggressive_horizon_mode=1 "
          f"u_pid_raw={u_pid:.10g} u_mpc_raw={u_mpc_raw:.10g} u_mpc={u_mpc:.10g} "
          f"mix_eff={mix_eff:.10g} mix_used={mix_used:.10g} improve={improve:.10g} improve_ratio={improve_ratio:.10g} "
          f"pid_cost={pid_cost:.10g} mpc_cost={mpc_cost:.10g} "
          f"blend_pre_clip={blend_pre_clip:.10g} final_action={final_action:.10g} "
          f"sign_agree={sign_agree} sign_corrected={sign_corrected} clip_fallback={int(clip_fallback)} "
          f"horizon={H} coeffs=a:{A0:.10g},b:{B0:.10g},c:{C0:.10g},d:{D0:.10g} "
          f"By={By0:.10g} Braw={Braw0:.10g} sign_flip={int(sign_flip0)}"
        )

      self.prev_target = target_raw
      self.prev_target_cond = target_raw
      self.prev_lat = lat
      self.prev_dy = self.dy_est
      self.dy_est = self.dy_est
      self.prev_v = float(state.v_ego)
      self.prev_roll = float(state.roll_lataccel)
      self.prev_model = {"A": A0, "B": B0, "C": C0, "D": D0}
      self.error_hist.append(float(target_raw - lat))
      if len(self.error_hist) > 12:
        self.error_hist = self.error_hist[-12:]
      self.action_hist.append(float(final_action))
      if len(self.action_hist) > 12:
        self.action_hist = self.action_hist[-12:]
      if control_active:
        self.last_action = float(final_action)
        self.prev_action_applied = True
      return float(final_action)

    if self.known_good_fixed_mix_mode and self.base_mix <= 1e-12:
      if control_active:
        self.step_count += 1
        self.known_good_steps += 1
        self.pid_exact_hard_bypass_steps += 1

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source=known_good_pid_exact pid_exact_hard_bypass_active=1 "
          f"simple_mode_active=1 known_good_fixed_mix_mode=1 "
          f"u_pid_raw={u_pid:.10g} final_action={u_pid:.10g} "
          f"any_post_processing_touched=0 any_non_pid_logic_touched=0 advanced_block_touched=0 "
          f"delta_clip_applied=0 rate_clip_applied=0"
        )

      return float(u_pid)

    if self.known_good_fixed_mix_mode:
      advanced_block_touched = 0
      if self.diag_mode and advanced_block_touched:
        raise RuntimeError("known_good_fixed_mix_mode touched an advanced block")

      legacy_path_active = 1
      any_nonlegacy_module_touched = 0
      err = float(target_raw - lat)
      err_abs = abs(err)
      dy_raw = 0.0 if self.prev_lat is None else float((lat - self.prev_lat) / self.dt)
      dy_abs = abs(dy_raw)
      if control_active:
        if self.known_good_prev_intervention_applied:
          harm = float(err_abs - self.known_good_prev_intervention_err_abs)
          self.known_good_recent_harm_ema = float(0.65 * self.known_good_recent_harm_ema + 0.35 * harm)
        else:
          self.known_good_recent_harm_ema = float(0.92 * self.known_good_recent_harm_ema)
      mix_eff = float(np.clip(self.base_mix, 0.0, 1.0))

      if self.fixed_mix_candidate_mode in ("simple_lqr", "simple_lqr_raw"):
        u_pred_raw, cand_diag = self._simple_predictive_action(target_raw, lat, dy_raw, state, future_plan, u_pid)
        if self.fixed_mix_candidate_mode == "simple_lqr_raw":
          u_pred_raw = float(np.clip(cand_diag["u_pred_raw"], self.u_min, self.u_max))
        repro_diag = {
          "H": int(cand_diag["H"]),
          "a": float(cand_diag["A"]),
          "b": float(cand_diag["B"]),
          "c": float(cand_diag["C"]),
          "d": float(cand_diag["D"]),
          "bins": cand_diag["bins"],
          "u_pred_raw": float(cand_diag["u_pred_raw"]),
          "candidate_mode": self.fixed_mix_candidate_mode,
          "sign_agree": int(cand_diag.get("sign_agree", 1)),
          "sign_corrected": int(cand_diag.get("sign_corrected", 0)),
        }
      elif self.fixed_mix_candidate_mode == "bc_preview":
        u_pred_raw, cand_diag = self._bc_preview_action(target_raw, lat, state, future_plan, u_pid)
        repro_diag = {
          "H": 1,
          "a": 0.0,
          "b": 0.0,
          "c": 0.0,
          "d": 0.0,
          "bins": {"vi0": 0, "vi1": 0, "ri0": 0, "ri1": 0},
          "u_pred_raw": float(u_pred_raw),
          "candidate_mode": "bc_preview",
          "sign_agree": 1,
          "sign_corrected": 0,
        }
      elif self.fixed_mix_candidate_mode == "legacy_bc":
        u_legacy, repro_diag = self._legacy_repro_candidate(target_raw, lat, state, future_plan, u_pid)
        u_bc, cand_diag = self._bc_preview_action(target_raw, lat, state, future_plan, u_pid)
        u_pred_raw = float((1.0 - self.fixed_mix_bc_candidate_mix) * u_legacy + self.fixed_mix_bc_candidate_mix * u_bc)
        repro_diag["u_pred_raw"] = float(u_pred_raw)
        repro_diag["candidate_mode"] = "legacy_bc"
      elif self.fixed_mix_candidate_mode == "nn_oracle":
        u_pred_raw = self._oracle_nn_action(target_raw, lat, state, future_plan, u_pid, u_pid)
        repro_diag = {
          "H": 1,
          "a": 0.0,
          "b": 0.0,
          "c": 0.0,
          "d": 0.0,
          "bins": {"vi0": 0, "vi1": 0, "ri0": 0, "ri1": 0},
          "u_pred_raw": float(u_pred_raw),
          "candidate_mode": "nn_oracle",
          "sign_agree": 1,
          "sign_corrected": 0,
        }
      elif self.fixed_mix_candidate_mode == "legacy_oracle":
        u_legacy, repro_diag = self._legacy_repro_candidate(target_raw, lat, state, future_plan, u_pid)
        u_oracle = self._oracle_nn_action(target_raw, lat, state, future_plan, u_pid, u_legacy)
        u_pred_raw = float((1.0 - self.fixed_mix_oracle_candidate_mix) * u_legacy + self.fixed_mix_oracle_candidate_mix * u_oracle)
        repro_diag["u_pred_raw"] = float(u_pred_raw)
        repro_diag["candidate_mode"] = "legacy_oracle"
      elif self.fixed_mix_candidate_mode == "bc_oracle":
        u_bc, _ = self._bc_preview_action(target_raw, lat, state, future_plan, u_pid)
        u_oracle = self._oracle_nn_action(target_raw, lat, state, future_plan, u_pid, u_bc)
        u_pred_raw = float((1.0 - self.fixed_mix_oracle_candidate_mix) * u_bc + self.fixed_mix_oracle_candidate_mix * u_oracle)
        repro_diag = {
          "H": 1,
          "a": 0.0,
          "b": 0.0,
          "c": 0.0,
          "d": 0.0,
          "bins": {"vi0": 0, "vi1": 0, "ri0": 0, "ri1": 0},
          "u_pred_raw": float(u_pred_raw),
          "candidate_mode": "bc_oracle",
          "sign_agree": 1,
          "sign_corrected": 0,
        }
      else:
        u_pred_raw, repro_diag = self._legacy_repro_candidate(target_raw, lat, state, future_plan, u_pid)
        repro_diag["candidate_mode"] = "legacy"
      nn_mix_scale = 1.0
      if self.fixed_mix_nn_mix_enable:
        nn_mix_scale = self._fixed_mix_nn_scale(target_raw, lat, state, future_plan, u_pid, u_pred_raw)
        if nn_mix_scale < self.fixed_mix_nn_gate_threshold:
          nn_mix_scale = 0.0
        mix_eff *= nn_mix_scale
      pred_delta_abs = abs(float(u_pred_raw - u_pid))
      if self.fixed_mix_delta_scale_enable and pred_delta_abs > self.fixed_mix_delta_scale_ref:
        delta_scale = max(self.fixed_mix_delta_scale_min, self.fixed_mix_delta_scale_ref / pred_delta_abs)
        mix_eff *= float(delta_scale)
      blend_pre_clip = float((1.0 - mix_eff) * u_pid + mix_eff * u_pred_raw)
      after_sat_clip = float(np.clip(blend_pre_clip, self.u_min, self.u_max))
      if control_active and self.prev_action_applied:
        after_rate_clip = float(np.clip(after_sat_clip, self.last_action - self.du_max_step, self.last_action + self.du_max_step))
      else:
        after_rate_clip = after_sat_clip
      blended_action = float(np.clip(after_rate_clip, self.u_min, self.u_max))
      clip_fallback = bool(
        self.fixed_mix_clip_fallback_enable
        and (
          abs(after_sat_clip - blend_pre_clip) > 1e-12
          or abs(after_rate_clip - after_sat_clip) > 1e-12
          or abs(blended_action - after_rate_clip) > 1e-12
        )
      )

      if self.fixed_mix_emergency_fallback_enable:
        emergency_prev_err_abs = float(self.known_good_emergency_prev_err_abs)
        emergency_err_rise = bool(control_active and err_abs > emergency_prev_err_abs + 1e-9)
        emergency_err_rise_count = int(self.known_good_emergency_err_rise_count + 1) if emergency_err_rise else 0
        emergency_trigger_dy = bool(control_active and dy_abs >= self.fixed_mix_emergency_dy_thresh)
        emergency_trigger_err = bool(control_active and err_abs >= self.fixed_mix_emergency_err_thresh)
        emergency_trigger_rise = bool(
          control_active
          and emergency_err_rise_count >= self.fixed_mix_emergency_err_rise_steps
          and dy_abs >= self.fixed_mix_emergency_dy_confirm
        )
        emergency_triggered = bool(
          emergency_trigger_dy
          or emergency_trigger_err
          or emergency_trigger_rise
        )
        emergency_active = bool(
          control_active
          and (
            getattr(self, "known_good_emergency_hold_count", 0) > 0
            or emergency_triggered
          )
        )

        if emergency_active:
          final_action = float(u_pid)
          mix_used = 0.0
          intervention_applied = False
          delta_clip_applied = int(abs(u_pid - blend_pre_clip) > 1e-12)

          if control_active:
            self.step_count += 1
            self.known_good_steps += 1
            self.known_good_advanced_touch_count += int(advanced_block_touched)
            self.fixed_mix_emergency_active_count += 1
            self.fixed_mix_emergency_fallback_count += 1
            self.fixed_mix_emergency_trigger_err_count += int(emergency_trigger_err)
            self.fixed_mix_emergency_trigger_dy_count += int(emergency_trigger_dy)
            self.fixed_mix_emergency_trigger_rise_count += int(emergency_trigger_rise)
            self.known_good_mix_used_sum += 0.0
            self.known_good_abs_delta_sum += 0.0
            self.known_good_delta_clip_count += delta_clip_applied
            self.mix_sum += mix_eff
            self.mix_sq_sum += mix_eff * mix_eff
            self.mix_used_sum += 0.0
            self.mix_used_sq_sum += 0.0
            self.accepted_delta_sum += 0.0
            self.delta_clip_count += delta_clip_applied

          if self.diag_mode and control_active and active_step <= self.diag_max_steps:
            bins0 = repro_diag["bins"]
            print(
              "TOP1_MPC_DIAG "
              f"step_idx={active_step} source=known_good_fixed_mix_emergency simple_mode_active=1 known_good_fixed_mix_mode=1 "
              f"legacy_path_active={legacy_path_active} any_nonlegacy_module_touched={any_nonlegacy_module_touched} advanced_block_touched={advanced_block_touched} "
              f"u_pid_raw={u_pid:.10g} u_pred_raw={u_pred_raw:.10g} "
              f"err={err:.10g} err_abs={err_abs:.10g} dy_abs={dy_abs:.10g} "
              f"blend_pre_clip={blend_pre_clip:.10g} after_rate_clip={after_rate_clip:.10g} after_sat_clip={after_sat_clip:.10g} final_action={final_action:.10g} "
              f"mix_used={mix_used:.10g} delta_clip_applied={delta_clip_applied} "
              f"emergency_active=1 emergency_hold_count={getattr(self, 'known_good_emergency_hold_count', 0)} "
              f"emergency_trigger_err={int(emergency_trigger_err)} emergency_trigger_dy={int(emergency_trigger_dy)} emergency_trigger_rise={int(emergency_trigger_rise)} "
              f"emergency_err_rise_count={emergency_err_rise_count} "
              f"horizon={repro_diag['H']} coeffs=a:{repro_diag['a']:.10g},b:{repro_diag['b']:.10g},c:{repro_diag['c']:.10g},d:{repro_diag['d']:.10g} "
              f"vi0={bins0['vi0']} vi1={bins0['vi1']} ri0={bins0['ri0']} ri1={bins0['ri1']} "
              f"control_active={int(control_active)}"
            )

          self.prev_target = target_raw
          self.prev_target_cond = target_raw
          self.prev_lat = lat
          self.prev_dy = dy_raw
          self.dy_est = dy_raw
          self.prev_v = float(state.v_ego)
          self.prev_roll = float(state.roll_lataccel)
          self.prev_model = None
          self.error_hist.append(float(target_raw - lat))
          if len(self.error_hist) > 12:
            self.error_hist = self.error_hist[-12:]
          self.action_hist.append(float(final_action))
          if len(self.action_hist) > 12:
            self.action_hist = self.action_hist[-12:]
          self.known_good_emergency_err_hist.append(float(err_abs))
          if len(self.known_good_emergency_err_hist) > 8:
            self.known_good_emergency_err_hist = self.known_good_emergency_err_hist[-8:]
          self.known_good_emergency_dy_hist.append(float(dy_abs))
          if len(self.known_good_emergency_dy_hist) > 8:
            self.known_good_emergency_dy_hist = self.known_good_emergency_dy_hist[-8:]
          if control_active:
            if emergency_triggered:
              self.known_good_emergency_hold_count = int(self.fixed_mix_emergency_hold_steps)
            if self.known_good_emergency_hold_count > 0:
              self.known_good_emergency_hold_count = max(0, int(self.known_good_emergency_hold_count) - 1)
            self.last_action = float(final_action)
            self.prev_action_applied = True
          self.known_good_emergency_prev_err_abs = float(err_abs)
          self.known_good_emergency_prev_dy_abs = float(dy_abs)
          self.known_good_emergency_err_rise_count = int(emergency_err_rise_count)
          return float(final_action)

      fixed_tail_sign_disagree = bool(
        abs(u_pred_raw - u_pid) > 1e-12
        and err_abs > self.veto_sign_err_thresh
        and (u_pred_raw - u_pid) * err < 0.0
      )
      fixed_tail_dy_veto = bool(self.fixed_mix_tail_veto_enable and dy_abs > self.veto_dy_thresh)
      fixed_tail_err_veto = bool(self.fixed_mix_tail_veto_enable and err_abs > self.veto_err_thresh)
      fixed_tail_delta_veto = bool(self.fixed_mix_tail_veto_enable and abs(u_pred_raw - u_pid) > self.veto_delta_thresh)
      fixed_tail_sign_veto = bool(self.fixed_mix_tail_veto_enable and fixed_tail_sign_disagree)
      fixed_tail_veto = bool(
        self.fixed_mix_tail_veto_enable
        and (fixed_tail_dy_veto or fixed_tail_err_veto or fixed_tail_delta_veto or fixed_tail_sign_veto)
      )

      if self.fixed_mix_tail_veto_enable:
        if fixed_tail_veto:
          final_action = float(u_pid)
          mix_used = 0.0
        else:
          final_action = blended_action
          mix_used = float(mix_eff)
          if abs(u_pred_raw - u_pid) > 1e-12:
            mix_used = float(np.clip((final_action - u_pid) / (u_pred_raw - u_pid), 0.0, 1.0))

        intervention_applied = bool(abs(final_action - u_pid) > 1e-12)
        delta_clip_applied = int(abs(final_action - blend_pre_clip) > 1e-12)

        if control_active:
          self.step_count += 1
          self.known_good_steps += 1
          self.known_good_interventions += int(intervention_applied)
          self.known_good_advanced_touch_count += int(advanced_block_touched)
          self.known_good_veto_count += int(fixed_tail_veto)
          self.fixed_mix_tail_veto_count += int(fixed_tail_veto)
          self.fixed_mix_sign_veto_count += int(fixed_tail_sign_veto)
          self.fixed_mix_err_veto_count += int(fixed_tail_err_veto)
          self.fixed_mix_dy_veto_count += int(fixed_tail_dy_veto)
          self.fixed_mix_delta_veto_count += int(fixed_tail_delta_veto)
          self.known_good_mix_used_sum += mix_used
          self.known_good_abs_delta_sum += abs(float(final_action - u_pid))
          self.known_good_delta_clip_count += delta_clip_applied
          self.blend_count += int(intervention_applied)
          self.mix_sum += mix_eff
          self.mix_sq_sum += mix_eff * mix_eff
          self.mix_used_sum += mix_used
          self.mix_used_sq_sum += mix_used * mix_used
          self.accepted_delta_sum += abs(float(final_action - u_pid))
          self.delta_clip_count += delta_clip_applied

        if self.diag_mode and control_active and active_step <= self.diag_max_steps:
          bins0 = repro_diag["bins"]
          print(
            "TOP1_MPC_DIAG "
            f"step_idx={active_step} source=known_good_fixed_mix_tail_veto simple_mode_active=1 known_good_fixed_mix_mode=1 "
            f"legacy_path_active={legacy_path_active} any_nonlegacy_module_touched={any_nonlegacy_module_touched} advanced_block_touched={advanced_block_touched} "
            f"u_pid_raw={u_pid:.10g} u_pred_raw={u_pred_raw:.10g} "
            f"err={err:.10g} err_abs={err_abs:.10g} dy_abs={dy_abs:.10g} "
            f"blend_pre_clip={blend_pre_clip:.10g} after_rate_clip={after_rate_clip:.10g} after_sat_clip={after_sat_clip:.10g} final_action={final_action:.10g} "
            f"mix_used={mix_used:.10g} delta_clip_applied={delta_clip_applied} fixed_tail_veto={int(fixed_tail_veto)} "
            f"veto_dy={int(fixed_tail_dy_veto)} veto_err={int(fixed_tail_err_veto)} veto_delta={int(fixed_tail_delta_veto)} veto_sign={int(fixed_tail_sign_veto)} "
            f"horizon={repro_diag['H']} coeffs=a:{repro_diag['a']:.10g},b:{repro_diag['b']:.10g},c:{repro_diag['c']:.10g},d:{repro_diag['d']:.10g} "
            f"vi0={bins0['vi0']} vi1={bins0['vi1']} ri0={bins0['ri0']} ri1={bins0['ri1']} "
            f"control_active={int(control_active)}"
          )

        self.prev_target = target_raw
        self.prev_target_cond = target_raw
        self.prev_lat = lat
        self.prev_dy = dy_raw
        self.dy_est = dy_raw
        self.prev_v = float(state.v_ego)
        self.prev_roll = float(state.roll_lataccel)
        self.prev_model = None
        self.error_hist.append(float(target_raw - lat))
        if len(self.error_hist) > 12:
          self.error_hist = self.error_hist[-12:]
        self.action_hist.append(float(final_action))
        if len(self.action_hist) > 12:
          self.action_hist = self.action_hist[-12:]
        if control_active:
          self.last_action = float(final_action)
          self.prev_action_applied = True
        return float(final_action)

      blended_delta = float(blended_action - u_pid)
      blended_delta_abs = abs(blended_delta)
      nontrivial_blend = bool(blended_delta_abs >= self.veto_min_mix_delta)

      a0 = float(repro_diag["a"])
      b0 = float(repro_diag["b"])
      c0 = float(repro_diag["c"])
      d0 = float(repro_diag["d"])
      roll0 = float(state.roll_lataccel)
      y_next_pid = float(a0 * lat + b0 * u_pid + c0 * roll0 + d0)
      y_next_blend = float(a0 * lat + b0 * blended_action + c0 * roll0 + d0)
      local_proxy_pid = float(abs(y_next_pid - target_raw))
      local_proxy_blend = float(abs(y_next_blend - target_raw))
      local_proxy_harm = bool(local_proxy_blend > local_proxy_pid + 1e-9)

      prev_err_abs = float(self.known_good_prev_intervention_err_abs)
      prev_dy_abs = float(self.known_good_prev_dy_abs)
      prior_blended = bool(self.known_good_prev_intervention_applied)

      err_rise = bool(prior_blended and nontrivial_blend and err_abs > prev_err_abs + 1e-9)
      dy_rise = bool(prior_blended and nontrivial_blend and dy_abs > prev_dy_abs + 1e-9)
      err_rise_count = int(self.known_good_err_rise_count + 1) if err_rise else 0
      dy_rise_count = int(self.known_good_dy_rise_count + 1) if dy_rise else 0

      sign_disagree = bool(
        nontrivial_blend
        and abs(u_pid) > 1e-4
        and abs(u_pred_raw) > 1e-4
        and u_pid * u_pred_raw < 0.0
      )
      sign_streak = int(self.known_good_sign_disagree_streak + 1) if sign_disagree else 0

      local_harm_streak = int(self.known_good_local_harm_streak + 1) if (prior_blended and nontrivial_blend and local_proxy_harm) else 0

      recent_actions = list(self.action_hist[-max(self.veto_flip_window - 1, 0):])
      flip_actions = recent_actions + ([blended_action] if nontrivial_blend else [])
      filtered_actions = [float(x) for x in flip_actions if abs(float(x)) > 1e-4]
      flip_count = 0
      if len(filtered_actions) >= 2:
        for i in range(1, len(filtered_actions)):
          if filtered_actions[i - 1] * filtered_actions[i] < 0.0:
            flip_count += 1
      err_not_improving = bool(prior_blended and err_abs >= prev_err_abs - 1e-9)
      local_behavior_worsening = bool(
        err_not_improving
        or local_proxy_harm
        or err_rise_count > 0
        or dy_rise_count > 0
      )

      harm_veto = bool(
        self.simple_veto_mode
        and (
          err_rise_count >= self.veto_err_rise_steps
          or dy_rise_count >= self.veto_dy_rise_steps
          or local_harm_streak >= self.veto_proxy_harm_steps
        )
      )
      sign_veto = bool(
        self.simple_veto_mode
        and sign_streak >= self.veto_sign_streak
        and local_behavior_worsening
      )
      flip_veto = bool(
        self.simple_veto_mode
        and prior_blended
        and nontrivial_blend
        and flip_count >= self.veto_flip_count_th
        and local_behavior_worsening
      )

      hold_veto = bool(self.simple_veto_mode and getattr(self, "known_good_veto_hold_count", 0) > 0)
      veto_triggered = bool(harm_veto or sign_veto or flip_veto or hold_veto)

      if clip_fallback:
        final_action = float(u_pid)
        mix_used = 0.0
      elif veto_triggered:
        final_action = float(u_pid)
        mix_used = 0.0
      else:
        final_action = blended_action
        mix_used = float(mix_eff)
        if abs(u_pred_raw - u_pid) > 1e-12:
          mix_used = float(np.clip((final_action - u_pid) / (u_pred_raw - u_pid), 0.0, 1.0))

      pre_residual_action = float(final_action)
      post_residual_delta = 0.0
      post_residual_applied = False
      if (
        self.fixed_mix_post_residual_enable
        and self.fixed_mix_post_residual_model is not None
        and control_active
        and not clip_fallback
        and not veto_triggered
      ):
        residual_delta = self._fixed_mix_post_residual_delta(
          target_raw,
          lat,
          state,
          future_plan,
          u_pid,
          u_pred_raw,
          pre_residual_action,
          mix_used,
        )
        if abs(residual_delta) > 1e-12:
          residual_action = float(pre_residual_action + self.fixed_mix_post_residual_scale * residual_delta)
          residual_action = self._apply_final_limits(residual_action, self.du_max_step, control_active)
          post_residual_delta = float(residual_action - pre_residual_action)
          final_action = float(residual_action)
          post_residual_applied = bool(abs(post_residual_delta) > 1e-12)

      intervention_applied = bool(abs(final_action - u_pid) > 1e-12)
      delta_clip_applied = int(abs(final_action - blend_pre_clip) > 1e-12)

      if control_active:
        self.step_count += 1
        self.known_good_steps += 1
        self.known_good_interventions += int(intervention_applied)
        self.known_good_advanced_touch_count += int(advanced_block_touched)
        self.known_good_veto_count += int(veto_triggered or clip_fallback)
        self.known_good_sign_disagree_count += int(sign_disagree)
        self.known_good_large_delta_count += int(nontrivial_blend)
        self.known_good_recent_harm_count += int(local_proxy_harm)
        self.known_good_err_veto_count += int(err_rise_count >= self.veto_err_rise_steps)
        self.known_good_dy_veto_count += int(dy_rise_count >= self.veto_dy_rise_steps)
        self.known_good_flip_veto_count += int(flip_veto)
        self.known_good_sign_veto_count += int(sign_veto)
        self.known_good_hold_veto_count += int(hold_veto)
        self.known_good_harm_veto_count += int(harm_veto)
        self.known_good_mix_used_sum += mix_used
        self.known_good_abs_delta_sum += abs(float(final_action - u_pid))
        self.known_good_delta_clip_count += delta_clip_applied
        self.known_good_post_residual_count += int(post_residual_applied)
        self.known_good_post_residual_abs_sum += abs(post_residual_delta)
        self.blend_count += int(intervention_applied)
        self.mix_sum += mix_eff
        self.mix_sq_sum += mix_eff * mix_eff
        self.mix_used_sum += mix_used
        self.mix_used_sq_sum += mix_used * mix_used
        self.accepted_delta_sum += abs(float(final_action - u_pid))
        self.delta_clip_count += delta_clip_applied

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        bins0 = repro_diag["bins"]
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source=known_good_fixed_mix simple_mode_active=1 known_good_fixed_mix_mode=1 "
          f"legacy_path_active={legacy_path_active} any_nonlegacy_module_touched={any_nonlegacy_module_touched} advanced_block_touched={advanced_block_touched} "
          f"u_pid_raw={u_pid:.10g} u_pred_raw={u_pred_raw:.10g} "
          f"err={err:.10g} err_abs={err_abs:.10g} dy_abs={dy_abs:.10g} "
          f"blend_pre_clip={blend_pre_clip:.10g} after_rate_clip={after_rate_clip:.10g} after_sat_clip={after_sat_clip:.10g} final_action={final_action:.10g} "
          f"mix_used={mix_used:.10g} nn_mix_scale={nn_mix_scale:.10g} clip_fallback={int(clip_fallback)} delta_clip_applied={delta_clip_applied} veto_triggered={int(veto_triggered)} "
          f"pre_residual_action={pre_residual_action:.10g} post_residual_delta={post_residual_delta:.10g} post_residual_applied={int(post_residual_applied)} "
          f"sign_disagree={int(sign_disagree)} sign_streak={sign_streak} err_rise_count={err_rise_count} dy_rise_count={dy_rise_count} local_harm_streak={local_harm_streak} "
          f"flip_count={flip_count} local_proxy_pid={local_proxy_pid:.10g} local_proxy_blend={local_proxy_blend:.10g} "
          f"hold_count={getattr(self, 'known_good_veto_hold_count', 0)} "
          f"veto_harm={int(harm_veto)} veto_sign={int(sign_veto)} veto_flip={int(flip_veto)} veto_hold={int(hold_veto)} "
          f"horizon={repro_diag['H']} coeffs=a:{repro_diag['a']:.10g},b:{repro_diag['b']:.10g},c:{repro_diag['c']:.10g},d:{repro_diag['d']:.10g} "
          f"vi0={bins0['vi0']} vi1={bins0['vi1']} ri0={bins0['ri0']} ri1={bins0['ri1']} "
          f"control_active={int(control_active)}"
        )

      self.prev_target = target_raw
      self.prev_target_cond = target_raw
      self.prev_lat = lat
      self.prev_dy = dy_raw
      self.dy_est = dy_raw
      self.prev_v = float(state.v_ego)
      self.prev_roll = float(state.roll_lataccel)
      self.prev_model = None
      self.error_hist.append(float(target_raw - lat))
      if len(self.error_hist) > 12:
        self.error_hist = self.error_hist[-12:]
      self.action_hist.append(float(final_action))
      if len(self.action_hist) > 12:
        self.action_hist = self.action_hist[-12:]
      if control_active:
        self.last_action = float(final_action)
        self.prev_action_applied = True
        if self.simple_veto_mode and (harm_veto or sign_veto or flip_veto):
          self.known_good_veto_hold_count = int(self.veto_hold_steps)
          self.known_good_reentry_pending = True
        elif self.simple_veto_mode and getattr(self, "known_good_veto_hold_count", 0) > 0:
          self.known_good_veto_hold_count = max(0, int(self.known_good_veto_hold_count) - 1)
        if self.simple_veto_mode and getattr(self, "known_good_reentry_pending", False) and intervention_applied:
          self.known_good_reentry_pending = False
          self.known_good_reentry_count = getattr(self, "known_good_reentry_count", 0) + 1
        self.known_good_sign_disagree_streak = int(sign_streak)
        self.known_good_err_rise_count = int(err_rise_count)
        self.known_good_dy_rise_count = int(dy_rise_count)
        self.known_good_local_harm_streak = int(local_harm_streak)
        self.known_good_blend_recent.append(int(intervention_applied))
        if len(self.known_good_blend_recent) > self.veto_flip_window:
          self.known_good_blend_recent = self.known_good_blend_recent[-self.veto_flip_window:]
        self.known_good_prev_intervention_applied = bool(intervention_applied)
        self.known_good_prev_intervention_err_abs = float(err_abs)
        self.known_good_prev_dy_abs = float(dy_abs)
        self.last_known_good_trace = {
          "u_pid": float(u_pid),
          "u_pred_raw": float(u_pred_raw),
          "mix_eff": float(mix_eff),
          "mix_used": float(mix_used),
          "pre_residual_action": float(pre_residual_action),
          "final_action": float(final_action),
          "post_residual_delta": float(post_residual_delta),
          "post_residual_applied": float(post_residual_applied),
          "err": float(err),
          "dy": float(dy_raw),
        }
        if self.fixed_mix_emergency_fallback_enable:
          emergency_prev_err_abs = float(self.known_good_emergency_prev_err_abs)
          emergency_err_rise = bool(err_abs > emergency_prev_err_abs + 1e-9)
          self.known_good_emergency_err_rise_count = int(self.known_good_emergency_err_rise_count + 1) if emergency_err_rise else 0
          self.known_good_emergency_prev_err_abs = float(err_abs)
          self.known_good_emergency_prev_dy_abs = float(dy_abs)
          self.known_good_emergency_err_hist.append(float(err_abs))
          if len(self.known_good_emergency_err_hist) > 8:
            self.known_good_emergency_err_hist = self.known_good_emergency_err_hist[-8:]
          self.known_good_emergency_dy_hist.append(float(dy_abs))
          if len(self.known_good_emergency_dy_hist) > 8:
            self.known_good_emergency_dy_hist = self.known_good_emergency_dy_hist[-8:]
      return float(final_action)

    if self.legacy_repro_mode and self.base_mix <= 1e-12:
      if control_active:
        self.step_count += 1
        self.pid_exact_hard_bypass_steps += 1

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source=pid_exact_hard_bypass pid_exact_hard_bypass_active=1 "
          f"u_pid_raw={u_pid:.10g} final_action={u_pid:.10g} "
          f"any_post_processing_touched=0 any_non_pid_logic_touched=0 "
          f"delta_clip_applied=0 rate_clip_applied=0"
        )

      return float(u_pid)

    if self.legacy_repro_mode:
      legacy_path_active = 1
      any_nonlegacy_module_touched = 0
      dy_raw = 0.0 if self.prev_lat is None else float((lat - self.prev_lat) / self.dt)

      u_pred_raw, repro_diag = self._legacy_repro_candidate(target_raw, lat, state, future_plan, u_pid)
      blend_pre_clip = float((1.0 - self.base_mix) * u_pid + self.base_mix * u_pred_raw)
      after_sat_clip = float(np.clip(blend_pre_clip, self.u_min, self.u_max))
      if control_active and self.prev_action_applied:
        after_rate_clip = float(np.clip(after_sat_clip, self.last_action - self.du_max_step, self.last_action + self.du_max_step))
      else:
        after_rate_clip = after_sat_clip
      final_action = float(np.clip(after_rate_clip, self.u_min, self.u_max))

      if control_active:
        self.step_count += 1
        self.blend_count += int(abs(final_action - u_pid) > 1e-12)
        self.mix_sum += self.base_mix
        self.mix_sq_sum += self.base_mix * self.base_mix
        self.mix_used_sum += self.base_mix
        self.mix_used_sq_sum += self.base_mix * self.base_mix
        self.accepted_delta_sum += abs(float(final_action - u_pid))
        self.delta_clip_count += int(abs(final_action - blend_pre_clip) > 1e-12)

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        bins0 = repro_diag["bins"]
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source=legacy_repro legacy_path_active={legacy_path_active} any_nonlegacy_module_touched={any_nonlegacy_module_touched} "
          f"u_pid_raw={u_pid:.10g} u_pred_raw={u_pred_raw:.10g} "
          f"blend_pre_clip={blend_pre_clip:.10g} after_rate_clip={after_rate_clip:.10g} after_sat_clip={after_sat_clip:.10g} final_action={final_action:.10g} "
          f"horizon={repro_diag['H']} coeffs=a:{repro_diag['a']:.10g},b:{repro_diag['b']:.10g},c:{repro_diag['c']:.10g},d:{repro_diag['d']:.10g} "
          f"vi0={bins0['vi0']} vi1={bins0['vi1']} ri0={bins0['ri0']} ri1={bins0['ri1']} "
          f"control_active={int(control_active)}"
        )

      self.prev_target = target_raw
      self.prev_target_cond = target_raw
      self.prev_lat = lat
      self.prev_dy = dy_raw
      self.dy_est = dy_raw
      self.prev_v = float(state.v_ego)
      self.prev_roll = float(state.roll_lataccel)
      self.prev_model = None
      self.error_hist.append(float(target_raw - lat))
      if len(self.error_hist) > 12:
        self.error_hist = self.error_hist[-12:]
      self.action_hist.append(float(final_action))
      if len(self.action_hist) > 12:
        self.action_hist = self.action_hist[-12:]
      if control_active:
        self.last_action = float(final_action)
        self.prev_action_applied = True
      return float(final_action)

    if self.legacy_fixed_mix_mode:
      legacy_path_active = 1
      any_nonlegacy_module_touched = 0
      self._update_dy_estimate(lat)

      H = int(np.clip(self.horizon, 6, 30))
      mix_eff = float(np.clip(self.base_mix, 0.0, 1.0))

      if self.legacy_pid_exact_mode and mix_eff <= 1e-12:
        source = "legacy_pid_exact"
        u_pid_after_pid_clip = float(u_pid)
        blend_pre_clip = float(u_pid)
        blend_post_rate = float(u_pid)
        blend_post_sat = float(u_pid)
        action = float(u_pid)
        clamp_or_override = 0
        model_coeffs_used = "pid_exact"

        if control_active:
          self.step_count += 1
          self.mix_sum += mix_eff
          self.mix_sq_sum += mix_eff * mix_eff
          self.mix_used_sum += 0.0
          self.mix_used_sq_sum += 0.0

        if self.diag_mode and control_active and active_step <= self.diag_max_steps:
          print(
            "TOP1_MPC_DIAG "
            f"step_idx={active_step} source={source} legacy_path_active={legacy_path_active} any_nonlegacy_module_touched={any_nonlegacy_module_touched} "
            f"err={target_raw - lat:.10g} target={target_raw:.10g} lat={lat:.10g} dy={self.dy_est:.10g} "
            f"u_pid_raw={u_pid:.10g} u_pid_after_pid_clip={u_pid_after_pid_clip:.10g} u_mpc_raw={u_pid:.10g} "
            f"blend_pre_clip={blend_pre_clip:.10g} blend_post_rate={blend_post_rate:.10g} blend_post_sat={blend_post_sat:.10g} "
            f"final_action={action:.10g} action={action:.10g} mix_eff={mix_eff:.6g} clamp_or_override={clamp_or_override} "
            f"model_coeffs_used={model_coeffs_used}"
          )

        self.prev_target = target_raw
        self.prev_lat = lat
        self.prev_dy = self.dy_est
        self.prev_v = float(state.v_ego)
        self.prev_roll = float(state.roll_lataccel)
        self.prev_model = None
        self.error_hist.append(float(target_raw - lat))
        if len(self.error_hist) > 12:
          self.error_hist = self.error_hist[-12:]
        self.action_hist.append(action)
        if len(self.action_hist) > 12:
          self.action_hist = self.action_hist[-12:]
        if control_active:
          self.last_action = float(action)
          self.prev_action_applied = True
        return float(action)

      tgt_seq = self._build_horizon_seq(target_raw, list(getattr(future_plan, "lataccel", [])), H)
      v_seq = self._build_horizon_seq(float(state.v_ego), list(getattr(future_plan, "v_ego", [])), H)
      roll_seq = self._build_horizon_seq(float(state.roll_lataccel), list(getattr(future_plan, "roll_lataccel", [])), H)

      ref_dy = np.zeros(H, dtype=np.float64)
      for k in range(1, H):
        ref_dy[k] = (tgt_seq[k] - tgt_seq[k - 1]) / self.dt

      x0 = np.array([lat - tgt_seq[0], self.dy_est - ref_dy[0]], dtype=np.float64)
      legacy_weights = {
        "track_w": float(self.tracking_weight),
        "dy_w": float(self.dy_weight),
        "ddy_w": float(self.ddy_weight),
        "du_w": float(self.du_weight),
        "delta_w": float(self.delta_weight),
        "barrier_w": float(self.barrier_weight),
        "term_w": float(self.terminal_weight),
      }

      A_mats, B_vecs, c_vecs, stage_bins = self._build_stage_dynamics(H, tgt_seq, ref_dy, v_seq, roll_seq, u_pid)
      K, kff = self._solve_lqr(A_mats, B_vecs, c_vecs, legacy_weights)

      base_cost, best_u_seq, _ = self._forward_plan(
        x0=x0,
        u_pid=u_pid,
        du_max_eff=self.du_max_step,
        A_mats=A_mats,
        B_vecs=B_vecs,
        c_vecs=c_vecs,
        ref_dy=ref_dy,
        K=K,
        kff=kff,
        weights=legacy_weights,
        delta_overrides={},
      )

      if self.polish_amp > 0.0 and H >= 2:
        p0 = float(min(self.polish_amp, 0.85 * self.du_max_step))
        p1 = 0.55 * p0
        candidates = [
          (0.0, 0.0), (-p0, 0.0), (p0, 0.0),
          (0.0, -p0), (0.0, p0),
          (-p0, -p0), (-p0, p0), (p0, -p0), (p0, p0),
          (-p1, 0.0), (p1, 0.0), (0.0, -p1), (0.0, p1),
        ]
        for d0, d1 in candidates:
          cst, u_seq, _ = self._forward_plan(
            x0=x0,
            u_pid=u_pid,
            du_max_eff=self.du_max_step,
            A_mats=A_mats,
            B_vecs=B_vecs,
            c_vecs=c_vecs,
            ref_dy=ref_dy,
            K=K,
            kff=kff,
            weights=legacy_weights,
            delta_overrides={0: d0, 1: d1},
          )
          if cst < base_cost:
            base_cost = cst
            best_u_seq = u_seq

      u_mpc_raw = float(best_u_seq[0])
      bins0, A0, B0, C0, D0, By0, Braw0, sign_flip0 = stage_bins[0]
      model_step0 = {"A": A0, "B": B0, "C": C0, "D": D0}
      u_mpc_raw, local_dir, sign_agree, sign_corrected = self._align_candidate_sign(
        lat, self.dy_est, float(state.roll_lataccel), target_raw, u_pid, u_mpc_raw, model_step0
      )
      u_mpc = float(np.clip(u_mpc_raw, u_pid - self.max_mpc_delta_vs_pid, u_pid + self.max_mpc_delta_vs_pid))
      u_mpc = float(np.clip(u_mpc, self.u_min, self.u_max))
      u_pid_after_pid_clip = float(u_pid)

      blend_pre_clip = float(u_pid + mix_eff * (u_mpc - u_pid))
      blend_post_sat = float(np.clip(blend_pre_clip, self.u_min, self.u_max))
      sat_clipped = int(abs(blend_post_sat - blend_pre_clip) > 1e-12)
      if control_active and self.prev_action_applied:
        blend_post_rate = float(np.clip(blend_post_sat, self.last_action - self.du_max_step, self.last_action + self.du_max_step))
      else:
        blend_post_rate = blend_post_sat
      rate_clipped = int(abs(blend_post_rate - blend_post_sat) > 1e-12)
      action = float(np.clip(blend_post_rate, self.u_min, self.u_max))
      final_sat_clipped = int(abs(action - blend_post_rate) > 1e-12)

      clamp_or_override = int(sat_clipped or rate_clipped or final_sat_clipped or abs(u_mpc - u_mpc_raw) > 1e-12)
      source = "legacy_blend"
      mix_apply = mix_eff

      prev_u = float(self.last_action if self.prev_action_applied else u_pid)
      one_step = self._one_step_cost_metrics(lat, self.dy_est, float(state.roll_lataccel), target_raw, u_pid, u_mpc, model_step0, prev_u)
      if self.sign_assert_mode and not sign_agree and abs(u_mpc_raw - u_pid) > 1e-6:
        raise RuntimeError(
          f"Legacy sign mismatch: target={target_raw} lat={lat} u_pid={u_pid} u_mpc_raw={u_mpc_raw} desired_sign={local_dir['desired_correction_sign']} B={B0} By={By0} Braw={Braw0}"
        )

      if control_active:
        self.step_count += 1
        self.accept_count += 1
        self.blend_count += 1
        self.mix_sum += mix_eff
        self.mix_sq_sum += mix_eff * mix_eff
        self.mix_used_sum += mix_apply
        self.mix_used_sq_sum += mix_apply * mix_apply
        self.accepted_delta_sum += abs(float(u_mpc - u_pid))
        self.accepted_err_improve_sum += float(one_step["err_pid"] - one_step["err_mpc"])
        self.delta_clip_count += int(clamp_or_override)

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        model_coeffs_used = f"A={A0:.7g},B={B0:.7g},C={C0:.7g},D={D0:.7g}"
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source={source} legacy_path_active={legacy_path_active} any_nonlegacy_module_touched={any_nonlegacy_module_touched} "
          f"err={target_raw - lat:.10g} desired_correction_sign={local_dir['desired_correction_sign']} sign_agree={sign_agree} sign_corrected={sign_corrected} "
          f"u_pid_raw={u_pid:.10g} u_pid_after_pid_clip={u_pid_after_pid_clip:.10g} u_mpc_raw={u_mpc_raw:.10g} u_mpc={u_mpc:.10g} "
          f"blend_pre_clip={blend_pre_clip:.10g} blend_post_rate={blend_post_rate:.10g} blend_post_sat={blend_post_sat:.10g} "
          f"final_action={action:.10g} action={action:.10g} mix_eff={mix_eff:.6g} "
          f"sat_clipped={sat_clipped} rate_clipped={rate_clipped} final_sat_clipped={final_sat_clipped} clamp_or_override={clamp_or_override} "
          f"one_step_err_if_pid={local_dir['one_step_err_if_pid']:.10g} one_step_err_if_mpc={target_raw - one_step['y_next_mpc']:.10g} "
          f"target={target_raw:.10g} lat={lat:.10g} dy={self.dy_est:.10g} c_pid={one_step['c_pid']:.10g} c_mpc={one_step['c_mpc']:.10g} "
          f"model_coeffs_used={model_coeffs_used} "
          f"By={By0:.7g} Braw={Braw0:.7g} sign_flip={int(sign_flip0)} vi0={bins0['vi0']} vi1={bins0['vi1']} ri0={bins0['ri0']} ri1={bins0['ri1']}"
        )

      self.prev_target = target_raw
      self.prev_lat = lat
      self.prev_dy = self.dy_est
      self.prev_v = float(state.v_ego)
      self.prev_roll = float(state.roll_lataccel)
      self.prev_model = {"A": A0, "B": B0, "C": C0, "D": D0}
      self.error_hist.append(float(target_raw - lat))
      if len(self.error_hist) > 12:
        self.error_hist = self.error_hist[-12:]
      self.action_hist.append(action)
      if len(self.action_hist) > 12:
        self.action_hist = self.action_hist[-12:]
      if control_active:
        self.last_action = float(action)
        self.prev_action_applied = True
      return float(action)

    if self.simple_hybrid_mode and not self.ff_only_mode:
      self._update_dy_estimate(lat)
      err = float(target_raw - lat)
      err_abs = abs(err)
      dy_abs = abs(self.dy_est)
      mix_eff = self._simple_schedule_mix(err, self.dy_est)
      gate_name = "off"
      delta_cap = 0.0
      if err_abs < self.safe_err_gate_1 and dy_abs < self.safe_dy_gate_1:
        gate_name = "calm"
        delta_cap = self.safe_calm_delta_cap
      elif err_abs < self.safe_err_gate_2 and dy_abs < self.safe_dy_gate_2:
        gate_name = "mild"
        delta_cap = self.safe_mild_delta_cap

      aggressive_override = bool(err_abs >= self.safe_err_gate_2 or dy_abs >= self.safe_dy_gate_2)
      predictive_considered = bool(gate_name != "off" and not aggressive_override)
      u_pred, simple_diag = self._simple_predictive_action(target_raw, lat, self.dy_est, state, future_plan, u_pid)

      blended_u = float((1.0 - mix_eff) * u_pid + mix_eff * u_pred)
      delta_raw = float(blended_u - u_pid)
      delta = float(np.clip(delta_raw, -delta_cap, delta_cap))
      final_u = float(u_pid + delta)

      prev_err = float(self.error_hist[-1]) if self.error_hist else err
      prev_u = float(self.last_action if self.prev_action_applied else u_pid)
      recent_actions = list(self.action_hist[-3:])
      recent_targets = list(self.target_hist[-3:])
      sign_samples = [float(x) for x in (recent_actions + [u_pid]) if abs(float(x)) > 1e-4]
      sign_flips = 0
      for i in range(1, len(sign_samples)):
        if sign_samples[i - 1] * sign_samples[i] < 0.0:
          sign_flips += 1

      target_flip_veto = False
      target_samples = recent_targets + [target_raw]
      if len(target_samples) >= 3:
        s1 = float(target_samples[-2] - target_samples[-3])
        s2 = float(target_samples[-1] - target_samples[-2])
        target_flip_veto = bool(
          abs(s1) > self.safe_target_flip_slope
          and abs(s2) > self.safe_target_flip_slope
          and s1 * s2 < 0.0
        )

      err_growth_veto = bool(err_abs > abs(prev_err) + self.safe_err_growth_veto)
      dy_growth_veto = bool(dy_abs > abs(self.prev_dy) + self.safe_dy_growth_veto)
      oscillation_veto = bool(sign_flips > 1)
      recovery_trend_veto = bool(
        self.prev_action_applied
        and err_abs + self.safe_recovery_trend_margin < abs(prev_err)
        and abs(final_u - u_pid) > 1e-12
        and abs(u_pid - self.last_action) > 1e-12
        and np.sign(final_u - u_pid) != np.sign(u_pid - self.last_action)
      )
      limit_veto = bool(
        abs(final_u) > self.safe_rate_limit_margin * max(abs(self.u_min), abs(self.u_max))
        or (control_active and self.prev_action_applied and abs(final_u - self.last_action) > self.safe_rate_limit_margin * self.du_max_step)
      )

      model_step = {
        "A": float(simple_diag["A"]),
        "B": float(simple_diag["B"]),
        "C": float(simple_diag["C"]),
        "D": float(simple_diag["D"]),
      }
      target_next = float(getattr(future_plan, "lataccel", [target_raw])[0] if len(getattr(future_plan, "lataccel", [])) > 0 else target_raw)
      one_step = self._one_step_cost_metrics(lat, self.dy_est, float(state.roll_lataccel), target_next, u_pid, final_u, model_step, prev_u)
      local_adv = float(one_step["c_pid"] - one_step["c_mpc"])
      err_improve = float(one_step["err_pid"] - one_step["err_mpc"])
      recent_accept_window = min(12, self.accept_budget_window)
      recent_accept_count = int(sum(self.simple_accept_recent[-self.accept_budget_window:])) if self.simple_accept_recent else 0
      recent_accept_small = int(sum(self.simple_accept_recent[-recent_accept_window:])) if self.simple_accept_recent else 0
      recent_action_sign_change = bool(sign_flips > 0)
      recent_target_slope_change = bool(target_flip_veto)
      sensitive_adv_req = bool(
        gate_name != "calm"
        or recent_accept_small > 0
        or recent_action_sign_change
        or recent_target_slope_change
      )
      base_local_adv_req = self.local_adv_min_calm if gate_name == "calm" else self.local_adv_min_mild
      if sensitive_adv_req:
        if gate_name == "calm":
          local_adv_required = self.local_adv_min_mild
        else:
          local_adv_required = self.local_adv_min_mild + self.local_adv_min_calm
      else:
        local_adv_required = base_local_adv_req
      local_adv_pass = bool(predictive_considered and local_adv >= local_adv_required)
      high_conviction = bool(local_adv >= local_adv_required + self.local_adv_min_calm)
      marginal_accept = bool(local_adv_pass and not high_conviction)
      accept_budget_max = max(1, int(np.floor(self.accept_budget_window * self.accept_budget_max_fraction + 1e-9))) if self.accept_budget_max_fraction > 0.0 else 0
      budget_veto = bool(predictive_considered and accept_budget_max > 0 and recent_accept_count >= accept_budget_max)
      marginal_budget_veto = bool(
        predictive_considered
        and marginal_accept
        and self.marginal_accept_suppression_after_n > 0
        and recent_accept_count >= self.marginal_accept_suppression_after_n
      )
      must_help_veto = bool(predictive_considered and not local_adv_pass)

      vetoed = bool(
        aggressive_override
        or mix_eff <= 1e-12
        or (predictive_considered and err_growth_veto)
        or (predictive_considered and dy_growth_veto)
        or (predictive_considered and oscillation_veto)
        or (predictive_considered and recovery_trend_veto)
        or (predictive_considered and target_flip_veto)
        or (predictive_considered and limit_veto)
        or must_help_veto
        or marginal_budget_veto
        or budget_veto
      )

      if vetoed:
        mix_eff = 0.0
        action = float(u_pid)
        delta_clipped = 0
        mix_used = 0.0
      else:
        action = self._apply_final_limits(final_u, self.du_max_step, control_active)
        delta_clipped = int(abs(delta - delta_raw) > 1e-12 or abs(action - final_u) > 1e-12)
        if abs(u_pred - u_pid) > 1e-12:
          mix_used = float(np.clip((action - u_pid) / (u_pred - u_pid), 0.0, 1.0))
        else:
          mix_used = 0.0

      intervention_applied = bool(abs(action - u_pid) > 1e-12)
      applied_delta = float(action - u_pid)

      if control_active:
        self.step_count += 1
        self.simple_mode_steps += 1
        if gate_name == "calm":
          self.simple_calm_gate_count += 1
        elif gate_name == "mild":
          self.simple_mild_gate_count += 1
        self.blend_count += int(mix_used > 1e-9)
        self.mix_sum += mix_eff
        self.mix_sq_sum += mix_eff * mix_eff
        self.mix_used_sum += mix_used
        self.mix_used_sq_sum += mix_used * mix_used
        self.accepted_delta_sum += abs(applied_delta)
        self.delta_clip_count += delta_clipped
        self.simple_veto_err_growth += int(predictive_considered and err_growth_veto)
        self.simple_veto_dy_growth += int(predictive_considered and dy_growth_veto)
        self.simple_veto_oscillation += int(predictive_considered and oscillation_veto)
        self.simple_veto_recovery_trend += int(predictive_considered and recovery_trend_veto)
        self.simple_veto_target_flip += int(predictive_considered and target_flip_veto)
        self.simple_veto_limit += int(predictive_considered and limit_veto)
        self.simple_veto_must_help += int(must_help_veto)
        self.simple_veto_budget += int(budget_veto or marginal_budget_veto)
        self.simple_predictive_considered_count += int(predictive_considered)
        self.simple_local_adv_pass_count += int(local_adv_pass)
        if intervention_applied:
          self.simple_accept_count += 1
          self.simple_marginal_accept_count += int(marginal_accept)
          self.simple_high_conviction_accept_count += int(high_conviction)
          if gate_name == "calm":
            self.simple_calm_accept_count += 1
          elif gate_name == "mild":
            self.simple_mild_accept_count += 1
          self.simple_local_adv_sum += local_adv
          self.simple_signed_delta_sum += applied_delta
          self.simple_max_abs_delta = max(self.simple_max_abs_delta, abs(applied_delta))
          self.simple_accepted_steps.append(int(active_step))
          if self.simple_accept_count == 1:
            self.simple_worst_accepted_local_adv = local_adv
          else:
            self.simple_worst_accepted_local_adv = min(self.simple_worst_accepted_local_adv, local_adv)

      if control_active:
        self.simple_accept_recent.append(int(intervention_applied))
        if len(self.simple_accept_recent) > self.accept_budget_window:
          self.simple_accept_recent = self.simple_accept_recent[-self.accept_budget_window:]

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        bins0 = simple_diag["bins"]
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source=simple_hybrid simple_mode_active=1 gate={gate_name} "
          f"target={target_raw:.10g} lat={lat:.10g} dy={self.dy_est:.10g} err={err:.10g} "
          f"u_pid={u_pid:.10g} u_pred_raw={simple_diag['u_pred_raw']:.10g} u_pred={u_pred:.10g} "
          f"blend_pre_clip={blended_u:.10g} action={action:.10g} "
          f"mix={mix_eff:.6g} mix_used={mix_used:.6g} aggressive_override={int(aggressive_override)} delta_clipped={delta_clipped} "
          f"local_adv={local_adv:.10g} local_adv_req={local_adv_required:.10g} local_adv_pass={int(local_adv_pass)} marginal_accept={int(marginal_accept)} high_conviction={int(high_conviction)} err_improve={err_improve:.10g} must_help_veto={int(must_help_veto)} budget_veto={int(budget_veto)} marginal_budget_veto={int(marginal_budget_veto)} "
          f"err_growth_veto={int(err_growth_veto)} dy_growth_veto={int(dy_growth_veto)} oscillation_veto={int(oscillation_veto)} recovery_trend_veto={int(recovery_trend_veto)} target_flip_veto={int(target_flip_veto)} limit_veto={int(limit_veto)} "
          f"sign_agree={simple_diag['sign_agree']} sign_corrected={simple_diag['sign_corrected']} "
          f"A={simple_diag['A']:.7g} B={simple_diag['B']:.7g} C={simple_diag['C']:.7g} D={simple_diag['D']:.7g} "
          f"vi0={bins0['vi0']} vi1={bins0['vi1']} ri0={bins0['ri0']} ri1={bins0['ri1']}"
        )

      if self.diag_mode and control_active and intervention_applied:
        print(
          "TOP1_MPC_INTERVENTION "
          f"step_idx={active_step} gate={gate_name} target={target_raw:.10g} lat={lat:.10g} dy={self.dy_est:.10g} "
          f"err={err:.10g} err_abs={err_abs:.10g} u_pid={u_pid:.10g} u_pred={u_pred:.10g} final_u={action:.10g} applied_delta={applied_delta:.10g} "
          f"prev_actions={recent_actions} prev_targets={recent_targets} local_c_pid={one_step['c_pid']:.10g} local_c_pred={one_step['c_mpc']:.10g} local_adv={local_adv:.10g}"
        )

      self.prev_target = target_raw
      self.prev_target_cond = target_raw
      self.prev_lat = lat
      self.prev_dy = self.dy_est
      self.prev_v = float(state.v_ego)
      self.prev_roll = float(state.roll_lataccel)
      self.prev_model = model_step
      self.target_hist.append(target_raw)
      if len(self.target_hist) > 12:
        self.target_hist = self.target_hist[-12:]
      self.error_hist.append(err)
      if len(self.error_hist) > 12:
        self.error_hist = self.error_hist[-12:]
      self.action_hist.append(float(action))
      if len(self.action_hist) > 12:
        self.action_hist = self.action_hist[-12:]
      if control_active:
        self.last_action = float(action)
        self.prev_action_applied = True
      return float(action)

    target_cond, dtarget_cond, curvature = self._condition_target(target_raw)
    self._update_dy_estimate(lat)
    self._update_observer(self.dy_est)

    # Optional FF-only path for ultra-fast sweeps.
    if self.ff_only_mode:
      ff_gain = self.ff_k0 + self.ff_k1 * float(np.clip(state.v_ego, 0.0, 40.0))
      ff_raw = ff_gain * (target_cond - lat)
      ff_beta = self.dt / (self.ff_tau + self.dt)
      self.ff_state = float(self.ff_state + ff_beta * (ff_raw - self.ff_state))

      mix_eff = self.base_mix
      u_mpc_candidate = float(u_pid + self.ff_mix * self.ff_state)
      u_mpc_candidate = float(np.clip(u_mpc_candidate, u_pid - self.max_mpc_delta_vs_pid, u_pid + self.max_mpc_delta_vs_pid))

      source = "pid" if mix_eff <= 1e-12 else "blend_direct"
      if source.startswith("pid"):
        action = float(u_pid)
      else:
        u_sel = float(u_pid + mix_eff * (u_mpc_candidate - u_pid))
        action = self._apply_final_limits(u_sel, self.du_max_step, control_active)

      if control_active:
        mix_apply = mix_eff if source == "blend_direct" else 0.0
        self.step_count += 1
        self.blend_count += int(source == "blend_direct")
        self.mix_sum += mix_eff
        self.mix_sq_sum += mix_eff * mix_eff
        self.mix_used_sum += mix_apply
        self.mix_used_sq_sum += mix_apply * mix_apply

      if self.diag_mode and control_active and active_step <= self.diag_max_steps:
        print(
          "TOP1_MPC_DIAG "
          f"step_idx={active_step} source={source} target={target_raw:.8g} target_cond={target_cond:.8g} "
          f"lat={lat:.8g} dy={self.dy_est:.8g} u_pid={u_pid:.8g} u_mpc={u_mpc_candidate:.8g} action={action:.8g} mix={mix_eff:.4f}"
        )

      self.prev_target = target_raw
      self.prev_lat = lat
      self.prev_dy = self.dy_est
      self.prev_v = float(state.v_ego)
      self.prev_roll = float(state.roll_lataccel)
      if control_active:
        self.last_action = float(action)
        self.prev_action_applied = True
      return float(action)

    # Build regime/risk features
    err = float(target_cond - lat)
    prev_err = float(self.error_hist[-1]) if self.error_hist else err
    err_growth = float(abs(err) - abs(prev_err))
    jerk_proxy = float((self.dy_est - self.prev_dy) / self.dt)
    sat_ratio = float(abs(self.last_action) / (max(abs(self.u_min), abs(self.u_max)) + 1e-9))

    if self.regime_enable:
      risk, regime = self._classify_regime(err, self.dy_est, dtarget_cond, sat_ratio, jerk_proxy, err_growth, self.last_innovation)
    else:
      risk, regime = 0.0, 0

    oscillatory = False
    if len(self.error_hist) >= 2:
      oscillatory = (self.error_hist[-1] * self.error_hist[-2] < 0.0)

    if self.tail_mode_enable:
      tail_trigger = self._tail_signal(err, self.dy_est, jerk_proxy, sat_ratio, risk)
      if tail_trigger:
        self.tail_hold_count = self.tail_hold_steps
      elif self.tail_hold_count > 0:
        self.tail_hold_count -= 1
      tail_mode = bool(tail_trigger or self.tail_hold_count > 0)
    else:
      tail_trigger = False
      self.tail_hold_count = 0
      tail_mode = False

    recovering = bool(tail_mode and not oscillatory and (abs(err) > 0.65 * self.tail_err_thresh or abs(self.dy_est) > 0.60 * self.tail_dy_thresh))

    mix_eff = self._schedule_mix(self.base_mix, err, self.dy_est, jerk_proxy, dtarget_cond, risk, self.last_innovation, tail_mode)
    if tail_mode and oscillatory:
      mix_eff = float(min(mix_eff, self.blend_tail_cap))
    elif recovering:
      mix_eff = float(np.clip(mix_eff + 0.5 * self.recovery_mix_bonus, self.mix_min, self.mix_max))

    H = self._schedule_horizon(regime, risk)
    H = int(np.clip(H, 6, self.horizon_aggressive))

    weights = self._schedule_weights(regime, risk, tail_mode, oscillatory)

    if recovering:
      self.recovery_count += 1

    du_max_eff = float(np.clip(self.du_max_step, 0.04, self.du_max_step))
    if tail_mode:
      du_max_eff = float(np.clip(self.du_max_step * (self.recovery_du_scale if recovering else 0.85), 0.05, 0.55))

    # Reference sequences (conditioned current target + raw future)
    tgt_seq = self._build_horizon_seq(target_cond, list(getattr(future_plan, "lataccel", [])), H)
    v_seq = self._build_horizon_seq(float(state.v_ego), list(getattr(future_plan, "v_ego", [])), H)
    roll_seq = self._build_horizon_seq(float(state.roll_lataccel), list(getattr(future_plan, "roll_lataccel", [])), H)

    ref_dy = np.zeros(H, dtype=np.float64)
    ref_dy[0] = dtarget_cond
    for k in range(1, H):
      ref_dy[k] = (tgt_seq[k] - tgt_seq[k - 1]) / self.dt

    x0 = np.array([lat - tgt_seq[0], self.dy_est - ref_dy[0]], dtype=np.float64)

    A_mats, B_vecs, c_vecs, stage_bins = self._build_stage_dynamics(H, tgt_seq, ref_dy, v_seq, roll_seq, u_pid)
    bins0, A0, B0, C0, D0, By0, Braw0, sign_flip0 = stage_bins[0]
    model_step0 = {"A": A0, "B": B0, "C": C0, "D": D0}
    prev_u = float(self.last_action if self.prev_action_applied else u_pid)

    best_candidate_family = "pid"
    candidate_pool = []
    if self.multi_candidate_enable:
      _, candidate_pool = self._generate_mpc_candidates(
        x0=x0,
        u_pid=u_pid,
        prev_u=prev_u,
        du_max_eff=du_max_eff,
        A_mats=A_mats,
        B_vecs=B_vecs,
        c_vecs=c_vecs,
        ref_dy=ref_dy,
        weights=weights,
        H=H,
        gate_regime=regime,
      )
      candidate_count = int(len(candidate_pool))
    else:
      K, kff = self._solve_lqr(A_mats, B_vecs, c_vecs, weights)
      _, u_seq_single, _ = self._forward_plan(
        x0=x0,
        u_pid=u_pid,
        du_max_eff=du_max_eff,
        A_mats=A_mats,
        B_vecs=B_vecs,
        c_vecs=c_vecs,
        ref_dy=ref_dy,
        K=K,
        kff=kff,
        weights=weights,
        delta_overrides={},
      )
      candidate_pool = [{
        **self._score_candidate_plan(
          x0=x0,
          u_seq=u_seq_single,
          u_pid=u_pid,
          prev_u=prev_u,
          du_max_eff=du_max_eff,
          A_mats=A_mats,
          B_vecs=B_vecs,
          c_vecs=c_vecs,
          ref_dy=ref_dy,
          weights=weights,
        ),
        "family": "medium_predictive",
        "u_seq": np.asarray(u_seq_single, dtype=np.float64),
      }]
      candidate_count = 1

    pid_candidate_summary = self._score_candidate_plan(
      x0=x0,
      u_seq=np.full(H, u_pid, dtype=np.float64),
      u_pid=u_pid,
      prev_u=prev_u,
      du_max_eff=du_max_eff,
      A_mats=A_mats,
      B_vecs=B_vecs,
      c_vecs=c_vecs,
      ref_dy=ref_dy,
      weights=weights,
    )
    pid_entry = {
      **pid_candidate_summary,
      "family": "pid",
      "u_seq": np.full(H, u_pid, dtype=np.float64),
    }
    if not any(str(c.get("family", "")) == "pid" for c in candidate_pool):
      candidate_pool = [pid_entry] + list(candidate_pool)

    pid_score = float(pid_candidate_summary["score"])
    pid_tail_risk = float(pid_candidate_summary["tail_risk"])
    pid_sat_penalty = float(pid_candidate_summary["saturation_penalty"])
    pid_osc_penalty = float(pid_candidate_summary["oscillation_penalty"])
    pid_max_dy = float(pid_candidate_summary["max_dy_abs"])
    pid_max_du = float(pid_candidate_summary["max_du"])

    prefix_steps_rank = min(self.accept_prefix_steps, H)
    pid_prefix_rank = np.full(prefix_steps_rank, u_pid, dtype=np.float64)
    accept_weights_rank = {
      "track_w": float(weights["track_w"] * (1.10 + 0.08 * regime)),
      "dy_w": float(weights["dy_w"]),
      "ddy_w": float(0.50 * weights["ddy_w"]),
      "du_w": float(0.20 * weights["du_w"]),
      "delta_w": float(0.15 * weights["delta_w"]),
      "barrier_w": float(0.10 * weights["barrier_w"]),
      "term_w": float(0.50 * weights["term_w"]),
    }
    cost_pid_pred_rank = self._prefix_sequence_cost(
      x0=x0,
      u_seq=pid_prefix_rank,
      u_pid=u_pid,
      prev_u=prev_u,
      du_max_eff=du_max_eff,
      A_mats=A_mats,
      B_vecs=B_vecs,
      c_vecs=c_vecs,
      ref_dy=ref_dy,
      weights=accept_weights_rank,
      steps=prefix_steps_rank,
    )

    ranked_candidates = []
    for cand in candidate_pool:
      family = str(cand.get("family", "pid"))
      if family == "pid":
        continue
      cand_seq = np.asarray(cand.get("u_seq", np.full(H, u_pid, dtype=np.float64)), dtype=np.float64)
      cand_prefix = np.full(prefix_steps_rank, u_pid, dtype=np.float64)
      copy_n = min(prefix_steps_rank, cand_seq.size)
      if copy_n > 0:
        cand_prefix[:copy_n] = cand_seq[:copy_n]
      cost_cand_pred = self._prefix_sequence_cost(
        x0=x0,
        u_seq=cand_prefix,
        u_pid=u_pid,
        prev_u=prev_u,
        du_max_eff=du_max_eff,
        A_mats=A_mats,
        B_vecs=B_vecs,
        c_vecs=c_vecs,
        ref_dy=ref_dy,
        weights=accept_weights_rank,
        steps=prefix_steps_rank,
      )
      cand_u0 = float(np.clip(cand_seq[0] if cand_seq.size else u_pid, self.u_min, self.u_max))
      one_step_cand = self._one_step_cost_metrics(lat, self.dy_est, float(state.roll_lataccel), target_cond, u_pid, cand_u0, model_step0, prev_u)
      tail_penalty = float(
        max(0.0, float(cand.get("tail_risk", 0.0)) - pid_tail_risk)
        + 0.70 * max(0.0, float(cand.get("saturation_penalty", 0.0)) - pid_sat_penalty)
        + 0.60 * max(0.0, float(cand.get("oscillation_penalty", 0.0)) - pid_osc_penalty)
        + 0.55 * max(0.0, float(cand.get("max_dy_abs", 0.0)) - pid_max_dy)
        + 0.45 * max(0.0, float(cand.get("max_du", 0.0)) - pid_max_du)
        + 0.50 * max(0.0, one_step_cand["err_mpc"] - one_step_cand["err_pid"])
        + 0.20 * max(0.0, abs(one_step_cand["dy_next_mpc"]) - abs(one_step_cand["dy_next_pid"]))
      )
      score_relative = float(cost_pid_pred_rank - cost_cand_pred - self.tail_penalty_weight * tail_penalty)
      family_quality = float(self.family_quality_ema.get(family, 0.0))
      adjusted_relative = float(score_relative + 0.15 * max(0.0, family_quality) - 0.45 * max(0.0, -family_quality))
      cand["score_relative"] = score_relative
      cand["tail_penalty"] = tail_penalty
      cand["adjusted_relative"] = adjusted_relative
      ranked_candidates.append(cand)

    if ranked_candidates:
      ranked_candidates.sort(key=lambda item: item["adjusted_relative"], reverse=True)
      best_candidate = ranked_candidates[0]
      best_u_seq = np.asarray(best_candidate["u_seq"], dtype=np.float64)
      best_candidate_score = float(best_candidate["score"])
      best_candidate_tail_risk = float(best_candidate["tail_risk"])
      best_candidate_family = str(best_candidate["family"])
      best_tail_penalty = float(best_candidate["tail_penalty"])
      best_score_relative = float(best_candidate["score_relative"])
      best_family_quality = float(self.family_quality_ema.get(best_candidate_family, 0.0))
    else:
      best_candidate = pid_entry
      best_u_seq = np.full(H, u_pid, dtype=np.float64)
      best_candidate_score = pid_score
      best_candidate_tail_risk = pid_tail_risk
      best_candidate_family = "pid"
      best_tail_penalty = 0.0
      best_score_relative = 0.0
      best_family_quality = 0.0

    u_mpc_candidate = float(best_u_seq[0])

    # Residual correction on top of predictive candidate.
    res_feat = np.array(
      [
        float(err),
        float(self.dy_est),
        float(dtarget_cond),
        float(self.last_innovation),
        float(sat_ratio),
      ],
      dtype=np.float64,
    )

    residual = 0.0
    if self.residual_enable and not (self.residual_disable_tail and tail_mode):
      residual = float(self.residual_gain * (self.res_bias + float(self.res_w @ res_feat)))
      residual = float(np.clip(residual, -self.residual_clip, self.residual_clip))
      u_mpc_candidate += residual

    self.res_feat_prev = res_feat

    u_mpc_candidate, local_dir, sign_agree, sign_corrected = self._align_candidate_sign(
      lat, self.dy_est, float(state.roll_lataccel), target_cond, u_pid, u_mpc_candidate, model_step0
    )

    # Align candidate with PID envelope.
    u_mpc_candidate = float(np.clip(u_mpc_candidate, u_pid - self.max_mpc_delta_vs_pid, u_pid + self.max_mpc_delta_vs_pid))
    u_mpc_candidate = float(np.clip(u_mpc_candidate, self.u_min, self.u_max))
    prev_u = float(self.last_action if self.prev_action_applied else u_pid)

    one_step = self._one_step_cost_metrics(lat, self.dy_est, float(state.roll_lataccel), target_cond, u_pid, u_mpc_candidate, model_step0, prev_u)
    if self.sign_assert_mode and not sign_agree and abs(u_mpc_candidate - u_pid) > 1e-6:
      raise RuntimeError(
        f"Predictive sign mismatch: target={target_cond} lat={lat} u_pid={u_pid} u_mpc={u_mpc_candidate} desired_sign={local_dir['desired_correction_sign']} B={B0} By={By0} Braw={Braw0}"
      )

    prefix_steps = min(self.accept_prefix_steps, H)
    pid_prefix = np.full(prefix_steps, u_pid, dtype=np.float64)
    mpc_prefix = np.full(prefix_steps, u_pid, dtype=np.float64)
    if prefix_steps > 0:
      copy_n = min(prefix_steps, len(best_u_seq))
      if copy_n > 0:
        mpc_prefix[:copy_n] = np.asarray(best_u_seq[:copy_n], dtype=np.float64)
      mpc_prefix[0] = u_mpc_candidate

    accept_weights = {
      "track_w": float(weights["track_w"] * (1.10 + 0.08 * regime)),
      "dy_w": float(weights["dy_w"]),
      "ddy_w": float(0.50 * weights["ddy_w"]),
      "du_w": float(0.20 * weights["du_w"]),
      "delta_w": float(0.15 * weights["delta_w"]),
      "barrier_w": float(0.10 * weights["barrier_w"]),
      "term_w": float(0.50 * weights["term_w"]),
    }
    cost_pid_pred = self._prefix_sequence_cost(
      x0=x0,
      u_seq=pid_prefix,
      u_pid=u_pid,
      prev_u=prev_u,
      du_max_eff=du_max_eff,
      A_mats=A_mats,
      B_vecs=B_vecs,
      c_vecs=c_vecs,
      ref_dy=ref_dy,
      weights=accept_weights,
      steps=prefix_steps,
    )
    cost_mpc_pred = self._prefix_sequence_cost(
      x0=x0,
      u_seq=mpc_prefix,
      u_pid=u_pid,
      prev_u=prev_u,
      du_max_eff=du_max_eff,
      A_mats=A_mats,
      B_vecs=B_vecs,
      c_vecs=c_vecs,
      ref_dy=ref_dy,
      weights=accept_weights,
      steps=prefix_steps,
    )

    accepted = False
    source = "pid"
    tail_forced = 0
    delta_clip = 0
    mix_apply = 0.0
    pred_advantage = float(cost_pid_pred - cost_mpc_pred)
    cost_ratio = float(cost_mpc_pred / max(cost_pid_pred, 1e-6))

    desired_sign = int(local_dir["desired_correction_sign"])
    cand_delta = float(u_mpc_candidate - u_pid)
    cand_sign = 0
    if cand_delta > 1e-12:
      cand_sign = 1
    elif cand_delta < -1e-12:
      cand_sign = -1

    sign_reject = bool(self.sign_check_enable and desired_sign != 0 and cand_sign != 0 and cand_sign != desired_sign)

    gate_regime = regime
    transition_signal = bool(
      abs(err) > 0.42 * self.regime_err_thresh
      or abs(self.dy_est) > 0.48 * self.regime_dy_thresh
      or abs(dtarget_cond) > 0.60 * self.regime_step_thresh
      or sat_ratio > 0.82 * self.regime_sat_thresh
      or abs(curvature) > 0.65 * self.regime_step_thresh
    )
    if gate_regime == 0 and transition_signal:
      gate_regime = 1

    delta_limit = float(self.accept_max_delta_relaxed if (recovering or gate_regime >= 2) else self.accept_max_delta_normal)
    delta_limit = float(min(delta_limit, self.max_mpc_delta_vs_pid))
    magnitude_reject = bool(self.magnitude_check_enable and abs(cand_delta) > 1.8 * delta_limit + 1e-12)

    recovery_hold_active = False
    if self.recovery_hold_count > 0:
      self.recovery_hold_count -= 1
      recovery_hold_active = True

    family_agreement, candidate_spread = self._candidate_consensus(candidate_pool, u_pid)
    score_pid = pid_score
    score_best = float(best_candidate_score)
    score_relative = float(best_score_relative)
    family_quality = float(best_family_quality)
    tail_penalty = float(best_tail_penalty)

    tail_signal = bool(
      self._tail_signal(err, self.dy_est, jerk_proxy, sat_ratio, risk)
      or (tail_mode and tail_penalty > 0.03)
      or (tail_mode and one_step["err_mpc"] > one_step["err_pid"] * 1.06)
      or (score_relative <= 0.0 and best_candidate_tail_risk > pid_tail_risk + 0.01)
    )
    if tail_signal and self.mpc_suppress_after_tail:
      self.recovery_hold_count = max(self.recovery_hold_count, self.recovery_hold_steps)
      recovery_hold_active = True

    tail_reject = bool((tail_signal and self.mpc_suppress_after_tail) or recovery_hold_active)
    hard_reject = bool(sign_reject or magnitude_reject or tail_reject)

    trust_before_tail, trust_after_tail, relative_norm = self._compute_trust(
      score_relative=score_relative,
      score_pid=score_pid,
      pred_advantage=pred_advantage,
      cost_pid_pred=cost_pid_pred,
      one_step=one_step,
      family_agreement=family_agreement,
      family_quality=family_quality,
      tail_penalty=tail_penalty,
      cand_delta=cand_delta,
      delta_limit=delta_limit,
      risk=risk,
      tail_mode=tail_mode,
      recovering=recovering,
      hard_reject=hard_reject,
    )

    candidate_quality_raw = float(score_relative)
    candidate_quality = float(np.tanh(candidate_quality_raw / max(1e-6, 0.20 * max(0.15, score_pid))))
    candidate_quality = float(np.clip(candidate_quality, -1.0, 1.0))
    self.consistency_state = float(np.clip(
      self.consistency_alpha * self.consistency_state + (1.0 - self.consistency_alpha) * candidate_quality,
      -1.0,
      1.0,
    ))

    soft_score = float(np.clip(trust_before_tail, 0.0, 1.0))
    score_thresh = float(self.trust_strict_threshold if self.strict_safe_mode else 0.08)
    cost_reject = bool((not hard_reject) and score_relative <= 0.0)

    if hard_reject or mix_eff <= 1e-9 or score_relative <= 0.0:
      mix_apply = 0.0
      accepted = False
    else:
      trust_effective = float(trust_after_tail if self.strict_safe_mode else np.clip(trust_after_tail + 0.10 * self.trust_ramp, 0.0, 1.0))
      mix_apply = float(mix_eff * trust_effective)
      if abs(cand_delta) > 1e-9:
        mix_apply = float(min(mix_apply, delta_limit / abs(cand_delta)))
      if tail_mode:
        mix_apply = float(min(mix_apply, self.blend_tail_cap))
      mix_apply = float(np.clip(mix_apply, 0.0, self.mix_max))
      accepted = bool(mix_apply >= max(self.accept_min_mix, 0.12 * mix_eff) and abs(cand_delta) > 1e-8)

    if accepted and pred_advantage > 0.0 and score_relative > 0.0 and not hard_reject:
      gain = float(np.clip(trust_after_tail, 0.0, 1.0))
      self.trust_ramp = float(min(self.trust_ramp_max, self.trust_ramp * (1.0 - 0.05) + self.trust_ramp_up * (0.15 + 0.85 * gain)))
    else:
      decay = float(self.trust_ramp_down * (1.30 if (hard_reject or score_relative <= 0.0) else 1.0))
      self.trust_ramp = float(max(0.0, self.trust_ramp * (1.0 - decay) - 0.10 * self.trust_ramp_up))

    if tail_reject:
      source = "pid_tail_hold" if recovery_hold_active else "pid_tail"
      tail_forced = 1
      accepted = False
      mix_apply = 0.0
    elif sign_reject:
      source = "pid_sign_reject"
      accepted = False
      mix_apply = 0.0
    elif magnitude_reject:
      source = "pid_mag_reject"
      accepted = False
      mix_apply = 0.0
    elif score_relative <= 0.0 or mix_apply <= 1e-9 or abs(cand_delta) <= 1e-8:
      source = "pid_soft"
      accepted = False
      mix_apply = 0.0
    else:
      source = "blend_direct"

    if source.startswith("pid"):
      action = float(u_pid)
      u_selected = u_pid
    else:
      u_selected = float(u_pid + mix_apply * (u_mpc_candidate - u_pid))
      u_selected = float(np.clip(u_selected, u_pid - self.max_mpc_delta_vs_pid, u_pid + self.max_mpc_delta_vs_pid))
      u_selected = float(np.clip(u_selected, self.u_min, self.u_max))
      du_cap = float(min(self.soft_pid_du_cap, du_max_eff))
      action = self._apply_final_limits(u_selected, du_cap, control_active)
      if abs(action - u_selected) > 1e-12:
        delta_clip = 1

    if control_active:
      self.step_count += 1
      self.accept_count += int(accepted)
      self.blend_count += int(not source.startswith("pid"))
      self.tail_forced_count += int(tail_forced)
      self.delta_clip_count += int(delta_clip)
      self.hard_reject_count += int(hard_reject)
      self.cost_reject_count += int(cost_reject)
      self.sign_reject_count += int(sign_reject)
      self.magnitude_reject_count += int(magnitude_reject)
      self.tail_reject_count += int(tail_reject)
      self.recovery_hold_active_count += int(recovery_hold_active)
      self.accepted_delta_sum += abs(float(u_mpc_candidate - u_pid)) if accepted else 0.0
      self.accepted_err_improve_sum += float(one_step["err_pid"] - one_step["err_mpc"]) if accepted else 0.0
      self.accepted_adv_sum += float(pred_advantage) if accepted else 0.0
      self.accept_regime_counts[regime] += int(accepted)
      self.gate_regime_counts[gate_regime] += 1
      self.accept_gate_regime_counts[gate_regime] += int(accepted)
      if accepted:
        self.accepted_adv_values.append(float(pred_advantage))
        self.accepted_score_relative_sum += float(score_relative)
        self.accepted_score_relative_values.append(float(score_relative))
        if best_candidate_family in self.family_accepted_adv_sum:
          self.family_accepted_adv_sum[best_candidate_family] += float(pred_advantage)
          self.family_accepted_count[best_candidate_family] += 1.0
        if len(self.accepted_adv_values) > 512:
          self.accepted_adv_values = self.accepted_adv_values[-512:]
        if len(self.accepted_score_relative_values) > 512:
          self.accepted_score_relative_values = self.accepted_score_relative_values[-512:]
      self.candidate_quality_values.append(float(candidate_quality))
      self.score_relative_values.append(float(score_relative))
      if score_relative > 0.0:
        self.trust_positive_sum += trust_after_tail
        self.trust_positive_count += 1.0
      else:
        self.trust_negative_sum += trust_after_tail
        self.trust_negative_count += 1.0
      if len(self.candidate_quality_values) > 512:
        self.candidate_quality_values = self.candidate_quality_values[-512:]
      if len(self.score_relative_values) > 512:
        self.score_relative_values = self.score_relative_values[-512:]

      for cand in ranked_candidates:
        fam = str(cand.get("family", "pid"))
        quality_obs = float(np.clip(cand.get("score_relative", 0.0), -0.5, 0.5))
        self.family_quality_ema[fam] = float(self.family_quality_alpha * self.family_quality_ema[fam] + (1.0 - self.family_quality_alpha) * quality_obs)
      if best_candidate_family in self.family_used_adv_ema:
        adv_obs = float(np.clip(pred_advantage / max(1e-6, cost_pid_pred), -0.5, 0.5))
        self.family_used_adv_ema[best_candidate_family] = float(self.family_quality_alpha * self.family_used_adv_ema[best_candidate_family] + (1.0 - self.family_quality_alpha) * adv_obs)

      self.mix_sum += mix_eff
      self.mix_sq_sum += mix_eff * mix_eff
      self.mix_used_sum += mix_apply
      self.mix_used_sq_sum += mix_apply * mix_apply
      self.trust_ramp_sum += self.trust_ramp
      self.consistency_sum += self.consistency_state
      self.candidate_quality_sum += candidate_quality
      self.trust_before_tail_sum += trust_before_tail
      self.trust_after_tail_sum += trust_after_tail
      self.trust_values.append(float(trust_after_tail))
      if len(self.trust_values) > 512:
        self.trust_values = self.trust_values[-512:]
      self.family_agreement_sum += family_agreement
      self.candidate_spread_sum += candidate_spread
      self.score_pid_sum += score_pid
      self.score_best_sum += score_best
      self.score_gap_sum += score_relative
      self.score_relative_sum += score_relative
      self.tail_penalty_sum += tail_penalty
      self.candidate_count_sum += float(candidate_count)
      self.best_candidate_score_sum += float(best_candidate_score)
      self.tail_risk_sum += float(best_candidate_tail_risk)
      if best_candidate_family in self.candidate_family_counts:
        self.candidate_family_counts[best_candidate_family] += 1.0
      self.mix_used_gate_sum[gate_regime] += mix_apply
      self.regime_counts[regime] += 1
      self.tail_mode_count += int(tail_mode)
      self.residual_abs_sum += abs(residual)

    if self.diag_mode and control_active and active_step <= self.diag_max_steps:
      print(
        "TOP1_MPC_DIAG "
        f"step_idx={active_step} target={target_raw:.10g} target_cond={target_cond:.10g} lat={lat:.10g} dy={self.dy_est:.10g} "
        f"err={target_cond - lat:.10g} desired_correction_sign={local_dir['desired_correction_sign']} sign_agree={sign_agree} sign_corrected={sign_corrected} "
        f"u_pid_raw={u_pid:.10g} u_mpc_raw={u_mpc_candidate:.10g} u_pid={u_pid:.10g} u_mpc={u_mpc_candidate:.10g} action={action:.10g} source={source} "
        f"mix={mix_eff:.6g} mix_apply={mix_apply:.6g} risk={risk:.6g} regime={regime} gate_regime={gate_regime} tail_mode={int(tail_mode)} recovering={int(recovering)} candidate_count={candidate_count} best_candidate_score={best_candidate_score:.6g} candidate_family={best_candidate_family} tail_risk={best_candidate_tail_risk:.6g} "
        f"H={H} c_pid={one_step['c_pid']:.10g} c_mpc={one_step['c_mpc']:.10g} cost_pid_pred={cost_pid_pred:.10g} cost_mpc_pred={cost_mpc_pred:.10g} cost_ratio={cost_ratio:.10g} pred_adv={pred_advantage:.10g} "
        f"err_pid={one_step['err_pid']:.10g} err_mpc={one_step['err_mpc']:.10g} one_step_err_if_pid={local_dir['one_step_err_if_pid']:.10g} one_step_err_if_mpc={target_cond - one_step['y_next_mpc']:.10g} "
        f"du_cap={du_max_eff:.6g} residual={residual:.8g} innovation={self.last_innovation:.8g} accepted={int(accepted)} hard_reject={int(hard_reject)} cost_reject={int(cost_reject)} sign_reject={int(sign_reject)} magnitude_reject={int(magnitude_reject)} tail_reject={int(tail_reject)} recovery_hold={int(recovery_hold_active)} trust_ramp={self.trust_ramp:.6g} trust_before_tail={trust_before_tail:.6g} trust_after_tail={trust_after_tail:.6g} consistency={self.consistency_state:.6g} candidate_quality={candidate_quality:.6g} family_agreement={family_agreement:.6g} candidate_spread={candidate_spread:.6g} soft_score={soft_score:.6g} score_thresh={score_thresh:.6g} score_pid={score_pid:.6g} score_best={score_best:.6g} score_relative={score_relative:.6g} family_quality={family_quality:.6g} tail_penalty={tail_penalty:.6g} delta_limit={delta_limit:.6g} delta_clip={delta_clip} "
        f"A={A0:.7g} B={B0:.7g} By={By0:.7g} Braw={Braw0:.7g} sign_flip={int(sign_flip0)} C={C0:.7g} D={D0:.7g} fast_bias={self.fast_bias:.8g} slow_bias={self.slow_bias:.8g} "
        f"vi0={bins0['vi0']} vi1={bins0['vi1']} ri0={bins0['ri0']} ri1={bins0['ri1']} wv={bins0['wv']:.5g} wr={bins0['wr']:.5g}"
      )

    # Update state
    self.prev_target = target_raw
    self.prev_lat = lat
    self.prev_dy = self.dy_est
    self.prev_v = float(state.v_ego)
    self.prev_roll = float(state.roll_lataccel)
    self.prev_model = {"A": A0, "B": B0, "C": C0, "D": D0}

    self.error_hist.append(err)
    if len(self.error_hist) > 12:
      self.error_hist = self.error_hist[-12:]

    self.action_hist.append(action)
    if len(self.action_hist) > 12:
      self.action_hist = self.action_hist[-12:]

    if control_active:
      self.last_action = float(action)
      self.prev_action_applied = True

    return float(action)


_OriginalTop1MPCController = Controller

from . import top1_mpc_tailblend as _tailblend_mod


def _apply_tailblend_env_from_top1_config():
  os.environ.setdefault("TOP1_MPC_TAILBLEND_ENABLE_SEGMENT_MODEL", "0")
  os.environ.setdefault("TOP1_MPC_TAILBLEND_ENABLE_RUNTIME_DELTA", "0")
  os.environ.setdefault("TOP1_MPC_TAILBLEND_ENABLE_RUNTIME_SPLINE", "0")
  os.environ.setdefault("TOP1_MPC_TAILBLEND_ENABLE_ONLINE_LIBRARY", "1")
  raw_cfg = os.getenv("TOP1_MPC_CONFIG", "").strip()
  if not raw_cfg:
    return
  try:
    cfg = json.loads(raw_cfg)
  except Exception:
    return
  if not isinstance(cfg, dict):
    return
  key_map = {
    "bank_path": "TOP1_MPC_TAILBLEND_BANK_PATH",
    "online_library_path": "TOP1_MPC_TAILBLEND_ONLINE_LIBRARY_PATH",
    "enable_online_library": "TOP1_MPC_TAILBLEND_ENABLE_ONLINE_LIBRARY",
    "online_knn_topk": "TOP1_MPC_TAILBLEND_ONLINE_KNN_TOPK",
    "online_knn_max_dist": "TOP1_MPC_TAILBLEND_ONLINE_KNN_MAX_DIST",
    "online_knn_cost_weight": "TOP1_MPC_TAILBLEND_ONLINE_KNN_COST_WEIGHT",
    "replay_mix": "TOP1_MPC_TAILBLEND_MIX",
    "replay_error_gain": "TOP1_MPC_TAILBLEND_ERR_GAIN",
    "replay_error_scale": "TOP1_MPC_TAILBLEND_ERR_SCALE",
    "replay_max_delta": "TOP1_MPC_TAILBLEND_MAX_DELTA",
    "enable_segment_model": "TOP1_MPC_TAILBLEND_ENABLE_SEGMENT_MODEL",
    "segment_model_path": "TOP1_MPC_TAILBLEND_SEGMENT_MODEL_PATH",
    "enable_runtime_delta": "TOP1_MPC_TAILBLEND_ENABLE_RUNTIME_DELTA",
    "runtime_delta_path": "TOP1_MPC_TAILBLEND_RUNTIME_DELTA_PATH",
    "enable_runtime_spline": "TOP1_MPC_TAILBLEND_ENABLE_RUNTIME_SPLINE",
    "runtime_spline_path": "TOP1_MPC_TAILBLEND_RUNTIME_SPLINE_PATH",
  }
  for key, env_key in key_map.items():
    if key in cfg:
      os.environ[env_key] = str(cfg[key])


_tailblend_mod._make_fallback_controller = lambda: _OriginalTop1MPCController()


class Controller(_tailblend_mod.Controller):
  DEFAULT_CONFIG = {
    "enable_online_library": 1,
    "enable_segment_model": 0,
    "enable_runtime_delta": 0,
    "enable_runtime_spline": 0,
    "online_knn_topk": 8,
    "online_knn_cost_weight": 0.08,
  }

  def __init__(self):
    _apply_tailblend_env_from_top1_config()
    super().__init__()
