import json
import os

import numpy as np

from . import BaseController

try:
  from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH, STEER_RANGE
except Exception:
  CONTROL_START_IDX = 100
  CONTEXT_LENGTH = 20
  STEER_RANGE = (-2.0, 2.0)


def _clip_action(value: float) -> float:
  return float(np.clip(value, STEER_RANGE[0], STEER_RANGE[1]))


def _exp_weighted_mean(values, steps: int, decay: float, default: float) -> float:
  if not values:
    return float(default)
  seq = np.asarray(values[:steps], dtype=np.float64)
  if seq.size == 0:
    return float(default)
  weights = np.exp(-np.arange(seq.size, dtype=np.float64) * decay)
  weights /= np.sum(weights)
  return float(np.dot(seq, weights))


class Controller(BaseController):
  """
  Generalizable online controller:
  preview PI + roll/target feedforward + mild rate limiting.
  """

  DEFAULT_CONFIG = {
    "p": 0.235,
    "i": 0.115,
    "d": 0.0,
    "preview_steps": 10,
    "preview_decay": 0.41,
    "preview_weight": 0.72,
    "preview_extra_weight": 0.08,
    "preview_severity_scale": 1.50,
    "target_ff": 0.10,
    "roll_ff": 0.34,
    "integral_decay": 1.0,
    "integral_clip": 30.0,
    "rate_base": 0.25,
    "rate_severity": 0.08,
    "rate_error": 0.14,
    "mag_pid_factor": 0.23,
    "aego_p_scale": 0.10,
    "min_p_scale": 0.10,
    "steer_factor": 13.0,
    "steer_sat_v": 20.0,
    "steer_command_sat": 2.0,
    "ff_gain": 0.74,
    "ff_speed_gain": 0.10,
  }

  def __init__(self):
    cfg = dict(self.DEFAULT_CONFIG)
    raw_cfg = os.getenv("TOP1_MPC_CONFIG", "").strip()
    if raw_cfg:
      try:
        user_cfg = json.loads(raw_cfg)
        if isinstance(user_cfg, dict):
          cfg.update(user_cfg)
      except Exception:
        pass

    self.p = float(cfg["p"])
    self.i = float(cfg["i"])
    self.d = float(cfg["d"])
    self.preview_steps = int(max(3, cfg["preview_steps"]))
    self.preview_decay = float(max(1e-4, cfg["preview_decay"]))
    self.preview_weight = float(np.clip(cfg["preview_weight"], 0.0, 1.0))
    self.preview_extra_weight = float(np.clip(cfg["preview_extra_weight"], 0.0, 0.4))
    self.preview_severity_scale = float(max(1e-4, cfg["preview_severity_scale"]))
    self.target_ff = float(cfg["target_ff"])
    self.roll_ff = float(cfg["roll_ff"])
    self.integral_decay = float(np.clip(cfg["integral_decay"], 0.0, 1.0))
    self.integral_clip = float(abs(cfg["integral_clip"]))
    self.rate_base = float(abs(cfg["rate_base"]))
    self.rate_severity = float(abs(cfg["rate_severity"]))
    self.rate_error = float(abs(cfg["rate_error"]))
    self.mag_pid_factor = float(cfg["mag_pid_factor"])
    self.aego_p_scale = float(cfg["aego_p_scale"])
    self.min_p_scale = float(cfg["min_p_scale"])
    self.steer_factor = float(cfg["steer_factor"])
    self.steer_sat_v = float(cfg["steer_sat_v"])
    self.steer_command_sat = float(cfg["steer_command_sat"])
    self.ff_gain = float(cfg["ff_gain"])
    self.ff_speed_gain = float(cfg["ff_speed_gain"])

    self.call_count = 0
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.prev_action = 0.0

  def _preview_reference(self, target_lataccel: float, future_plan) -> tuple[float, float, float]:
    future_targets = list(getattr(future_plan, "lataccel", []))
    preview_target = _exp_weighted_mean(
      future_targets,
      self.preview_steps,
      self.preview_decay,
      target_lataccel,
    )
    future_delta = preview_target - float(target_lataccel)
    severity = min(1.0, abs(future_delta) / self.preview_severity_scale)
    preview_weight = min(0.95, self.preview_weight + self.preview_extra_weight * severity)
    reference = (1.0 - preview_weight) * float(target_lataccel) + preview_weight * preview_target
    return float(reference), float(future_delta), float(severity)

  def _roll_preview(self, roll_lataccel: float, future_plan) -> float:
    future_roll = list(getattr(future_plan, "roll_lataccel", []))
    preview_roll = _exp_weighted_mean(future_roll, 5, self.preview_decay, roll_lataccel)
    return float(preview_roll - float(roll_lataccel))

  def _ff_gain_for_speed(self, v_ego: float) -> float:
    return float(self.ff_gain + self.ff_speed_gain * min(1.0, max(0.0, float(v_ego)) / 30.0))

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.call_count += 1

    if self.call_count < (CONTROL_START_IDX - CONTEXT_LENGTH + 1):
      self.prev_error = float(target_lataccel - current_lataccel)
      return 0.0

    reference, future_delta, severity = self._preview_reference(target_lataccel, future_plan)
    roll_preview = self._roll_preview(state.roll_lataccel, future_plan)

    error = float(reference - current_lataccel)
    if error * self.prev_error < 0.0:
      self.error_integral *= 0.95
    self.error_integral = float(np.clip(
      self.integral_decay * self.error_integral + error,
      -self.integral_clip,
      self.integral_clip,
    ))
    error_diff = error - self.prev_error

    pid_factor = max(0.5, 1.0 - self.mag_pid_factor * abs(reference))
    p_dynamic = max(self.min_p_scale, self.p - self.aego_p_scale * abs(float(state.a_ego)))
    feedback = (p_dynamic * error + self.i * self.error_integral + self.d * error_diff) * pid_factor
    ff_target = self.target_ff * future_delta
    ff_roll = self.roll_ff * roll_preview
    steer_accel_target = float(reference - float(state.roll_lataccel))
    steer_command = steer_accel_target * self.steer_factor / max(self.steer_sat_v, float(state.v_ego))
    steer_command = 2.0 * self.steer_command_sat / (1.0 + np.exp(-steer_command)) - self.steer_command_sat
    ff_gain = self._ff_gain_for_speed(state.v_ego)
    u_ff = ff_gain * steer_command
    raw_action = feedback + ff_target + ff_roll + u_ff

    rate_limit = self.rate_base + self.rate_severity * severity
    rate_limit += self.rate_error * min(1.0, abs(error) / 1.5)
    action = float(np.clip(raw_action, self.prev_action - rate_limit, self.prev_action + rate_limit))
    action = _clip_action(action)

    self.prev_error = error
    self.prev_action = action
    return action
