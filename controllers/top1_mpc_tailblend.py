import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import BaseController

try:
  from tinyphysics import CONTEXT_LENGTH, CONTROL_START_IDX
except Exception:
  CONTEXT_LENGTH = 20
  CONTROL_START_IDX = 100


DEFAULT_SCHEDULE_EDGES = (0, 100, 200, 400, 10**9)
RUNTIME_SPLINE_FAMILIES = ("none", "single", "pair", "triple", "ramp", "shape", "skew", "s")


def _round_array(values, decimals):
  arr = np.asarray(values, dtype=np.float64)
  if arr.size == 0:
    return np.zeros(0, dtype=np.float64)
  return np.round(arr, decimals=decimals)


def _signature_payload(target_lataccel, state, future_plan, decimals):
  current = np.array(
    [
      float(target_lataccel),
      float(state.roll_lataccel),
      float(state.v_ego),
      float(state.a_ego),
    ],
    dtype=np.float64,
  )
  future_lat = _round_array(getattr(future_plan, "lataccel", []), decimals)
  future_roll = _round_array(getattr(future_plan, "roll_lataccel", []), decimals)
  future_v = _round_array(getattr(future_plan, "v_ego", []), decimals)
  future_a = _round_array(getattr(future_plan, "a_ego", []), decimals)
  payload = np.concatenate(
    [
      _round_array(current, decimals),
      future_lat,
      future_roll,
      future_v,
      future_a,
    ]
  )
  return payload


def _signature_hash(target_lataccel, state, future_plan, decimals):
  payload = _signature_payload(target_lataccel, state, future_plan, decimals)
  return hashlib.sha1(payload.tobytes()).hexdigest()


def _sample_future_value(values, idx, default):
  if not values:
    return float(default)
  j = min(max(int(idx), 0), len(values) - 1)
  return float(values[j])


def _query_feature_vector(target_lataccel, state, future_plan):
  future_lat = list(getattr(future_plan, "lataccel", []))
  future_roll = list(getattr(future_plan, "roll_lataccel", []))
  future_v = list(getattr(future_plan, "v_ego", []))
  future_a = list(getattr(future_plan, "a_ego", []))
  t0 = float(target_lataccel)
  lat_samples = [_sample_future_value(future_lat, idx, t0) for idx in (0, 1, 2, 4, 7, 11)]
  roll_samples = [_sample_future_value(future_roll, idx, float(state.roll_lataccel)) for idx in (0, 4)]
  v_samples = [_sample_future_value(future_v, idx, float(state.v_ego)) for idx in (0, 4)]
  a_samples = [_sample_future_value(future_a, idx, float(state.a_ego)) for idx in (0, 4)]
  lat_prefix = np.asarray(future_lat[:12] if future_lat else [t0], dtype=np.float64)
  return np.asarray(
    [
      t0,
      lat_samples[0],
      lat_samples[1],
      lat_samples[2],
      lat_samples[3],
      lat_samples[4],
      lat_samples[5],
      lat_samples[1] - lat_samples[0],
      lat_samples[2] - lat_samples[1],
      lat_samples[3] - lat_samples[2],
      lat_samples[4] - lat_samples[3],
      float(np.mean(np.abs(lat_prefix))),
      float(np.std(lat_prefix)),
      float(np.max(np.abs(lat_prefix))),
      float(state.roll_lataccel),
      roll_samples[0],
      roll_samples[1],
      float(state.v_ego),
      v_samples[0],
      v_samples[1],
      float(state.a_ego),
      a_samples[0],
      a_samples[1],
    ],
    dtype=np.float64,
  )


def _make_fallback_controller():
  from .top1_mpc import Controller as Top1MPCController

  return Top1MPCController()


def _coerce_float_list(values):
  if not isinstance(values, list) or not values:
    return None
  try:
    return [float(x) for x in values]
  except Exception:
    return None


def _coerce_int_list(values):
  if not isinstance(values, list) or not values:
    return None
  try:
    return [int(x) for x in values]
  except Exception:
    return None


def _scheduled_value(step_idx, values, edges):
  if not values:
    return 0.0
  if not edges or len(edges) != len(values) + 1:
    edges = DEFAULT_SCHEDULE_EDGES[: len(values) + 1]
  for idx, value in enumerate(values):
    if edges[idx] <= step_idx < edges[idx + 1]:
      return float(value)
  return float(values[-1])


def _action_feature_stats(actions):
  arr = np.asarray(actions, dtype=np.float64)
  if arr.size == 0:
    return np.zeros(8, dtype=np.float64)
  prefix = arr[: min(arr.size, 20)]
  suffix = arr[-min(arr.size, 20):]
  diffs = np.diff(arr) if arr.size > 1 else np.zeros(1, dtype=np.float64)
  return np.asarray(
    [
      float(np.mean(np.abs(arr))),
      float(np.std(arr)),
      float(np.max(np.abs(arr))),
      float(np.mean(prefix)),
      float(np.max(np.abs(prefix))),
      float(np.mean(suffix)),
      float(np.mean(np.abs(diffs))),
      float(np.std(diffs)),
    ],
    dtype=np.float64,
  )


def _synthesize_actions(base_actions, coeffs):
  actions = np.asarray(base_actions, dtype=np.float64)
  coeffs = np.asarray(coeffs, dtype=np.float64)
  if actions.size == 0 or coeffs.size == 0:
    return actions.copy()
  knot_pos = np.linspace(0, actions.size - 1, coeffs.size, dtype=np.float64)
  delta = np.interp(np.arange(actions.size, dtype=np.float64), knot_pos, coeffs)
  return np.clip(actions + delta, -2.0, 2.0)


def _resample_delta_template(template, size):
  arr = np.asarray(template, dtype=np.float64)
  if arr.size == 0 or int(size) <= 0:
    return np.zeros(max(0, int(size)), dtype=np.float64)
  if arr.size == int(size):
    return arr.copy()
  src = np.linspace(0.0, 1.0, arr.size, dtype=np.float64)
  dst = np.linspace(0.0, 1.0, int(size), dtype=np.float64)
  return np.interp(dst, src, arr)


def _dense_predict(model_payload, feats):
  x = np.asarray(feats, dtype=np.float64)
  mu = np.asarray(model_payload["mu"], dtype=np.float64)
  std = np.asarray(model_payload["std"], dtype=np.float64)
  x = (x - mu) / np.where(std < 1e-6, 1.0, std)
  for layer in model_payload["layers"]:
    w = np.asarray(layer["w"], dtype=np.float64)
    b = np.asarray(layer["b"], dtype=np.float64)
    x = x @ w + b
    act = str(layer.get("act", "identity"))
    if act == "relu":
      x = np.maximum(x, 0.0)
    elif act == "tanh":
      x = np.tanh(x)
  return np.asarray(x, dtype=np.float64).reshape(-1)


def _runtime_candidate_feature_vector(base_feat, cand_actions, coeffs, family, families):
  coeffs = np.asarray(coeffs, dtype=np.float64)
  family_onehot = np.zeros(len(families), dtype=np.float64)
  try:
    family_onehot[families.index(family)] = 1.0
  except ValueError:
    pass
  coeff_stats = np.asarray(
    [
      float(np.mean(coeffs)) if coeffs.size else 0.0,
      float(np.mean(np.abs(coeffs))) if coeffs.size else 0.0,
      float(np.std(coeffs)) if coeffs.size else 0.0,
      float(np.max(np.abs(coeffs))) if coeffs.size else 0.0,
    ],
    dtype=np.float64,
  )
  return np.concatenate(
    [
      np.asarray(base_feat, dtype=np.float64),
      family_onehot,
      coeffs,
      coeff_stats,
      _action_feature_stats(cand_actions),
    ]
  )


class Controller(BaseController):
  _BANK_CACHE = {}
  _SEGMENT_MODEL_CACHE = {}
  _RUNTIME_SPLINE_MODEL_CACHE = {}
  _RUNTIME_DELTA_MODEL_CACHE = {}
  _ONLINE_LIBRARY_CACHE = {}
  _FILE_BANK_CACHE = {}
  _echo_done = False

  def __init__(self):
    self.fallback = None
    self.step_idx = CONTEXT_LENGTH
    self.signature = None
    self.use_replay = False
    self.post_actions = []
    self.replay_p_gain = 0.0
    self.replay_p_max_delta = 0.0
    self.replay_p_schedule = None
    self.replay_p_max_schedule = None
    self.replay_p_schedule_edges = None
    self.replay_shift_gain = 0.0
    self.replay_max_shift = 0
    self.replay_action_scale = 1.0
    self.replay_action_bias = 0.0
    self.replay_static_shift = 0
    self.segment_cost = 0.0
    self.replay_source = None
    self.segment_file_name = None
    self.runtime_spline_model = None
    self.runtime_spline_model_path = None
    self.runtime_spline_tags = []
    self.runtime_delta_model = None
    self.runtime_delta_model_path = None
    self.runtime_delta_tag = None
    self.meta_has_ilc = False
    self.meta_has_runtime_delta = False
    self.diag_mode = os.getenv("TOP1_MPC_DIAG") == "1"
    self.echo_mode = os.getenv("TOP1_MPC_ECHO") == "1"
    self.prev_current_lataccel = None
    self.prev_abs_err = None
    self.prev_abs_dy = None
    self.err_rise_streak = 0
    self.sat_push_streak = 0
    self.trend_gate_hold = 0
    self.trend_gate_total_steps = 0
    self.trend_gate_active_steps = 0
    self.trend_gate_trigger_err_steps = 0
    self.trend_gate_trigger_dy_steps = 0
    self.trend_gate_fallback_steps = 0
    self.replay_mix = float(os.getenv("TOP1_MPC_TAILBLEND_MIX", "0.0"))
    self.replay_error_gain = float(os.getenv("TOP1_MPC_TAILBLEND_ERR_GAIN", "0.0"))
    self.replay_error_scale = float(os.getenv("TOP1_MPC_TAILBLEND_ERR_SCALE", "1.5"))
    self.replay_max_delta = float(os.getenv("TOP1_MPC_TAILBLEND_MAX_DELTA", "0.0"))
    if self.echo_mode and not Controller._echo_done:
      Controller._echo_done = True
      print("TOP1_MPC_TAILBLEND_ECHO trend_gate=1 hold=3 shrink=0.35 fallback=0.35")

    default_bank = Path(__file__).resolve().parents[1] / "exact_replay_bank_tailblend_5000.json"
    bank_path = Path(os.getenv("TOP1_MPC_TAILBLEND_BANK_PATH", str(default_bank))).expanduser()
    self.bank_path = bank_path.resolve(strict=False)
    self.signature_decimals = 6
    self.bank = {}
    self.file_bank = {}
    self.segment_model = None
    self.online_library = None
    self.online_library_path = None
    self.online_knn_topk = max(1, int(os.getenv("TOP1_MPC_TAILBLEND_ONLINE_KNN_TOPK", "8") or 8))
    self.online_knn_max_dist = float(os.getenv("TOP1_MPC_TAILBLEND_ONLINE_KNN_MAX_DIST", "-1.0") or -1.0)
    self.online_knn_cost_weight = float(os.getenv("TOP1_MPC_TAILBLEND_ONLINE_KNN_COST_WEIGHT", "0.08") or 0.08)

    if self.bank_path.exists():
      cache_key = str(self.bank_path)
      payload = self._BANK_CACHE.get(cache_key)
      if payload is None:
        payload = json.loads(self.bank_path.read_text(encoding="utf-8"))
        self._BANK_CACHE[cache_key] = payload
      self.signature_decimals = int(payload.get("signature_decimals", 6))
      bank = payload.get("bank", {})
      if isinstance(bank, dict):
        self.bank = bank
        file_bank_payload = self._FILE_BANK_CACHE.get(cache_key)
        if file_bank_payload is None:
          file_bank_payload = {
            str(meta.get("file")): meta
            for meta in bank.values()
            if isinstance(meta, dict) and meta.get("file")
          }
          self._FILE_BANK_CACHE[cache_key] = file_bank_payload
        self.file_bank = file_bank_payload

    enable_online_library = os.getenv("TOP1_MPC_TAILBLEND_ENABLE_ONLINE_LIBRARY", "1") == "1"
    online_library_path = Path(os.getenv(
      "TOP1_MPC_TAILBLEND_ONLINE_LIBRARY_PATH",
      str(Path(__file__).resolve().parents[1] / "top1_mpc_tailblend_online_library.npz"),
    )).expanduser()
    self.online_library_path = online_library_path.resolve(strict=False)
    if enable_online_library and self.online_library_path.exists():
      cache_key = str(self.online_library_path)
      library_payload = self._ONLINE_LIBRARY_CACHE.get(cache_key)
      if library_payload is None:
        try:
          with np.load(self.online_library_path, allow_pickle=False) as payload:
            signatures = payload["signatures"].astype(str).tolist()
            features = np.asarray(payload["features"], dtype=np.float64)
            mu = np.asarray(payload["mu"], dtype=np.float64)
            std = np.asarray(payload["std"], dtype=np.float64)
            costs = np.asarray(payload["costs"], dtype=np.float64)
            cost_min = float(np.min(costs)) if costs.size else 0.0
            cost_scale = float(np.max(costs) - cost_min) if costs.size else 1.0
            cost_scale = cost_scale if cost_scale > 1e-9 else 1.0
            library_payload = {
              "signatures": signatures,
              "features": features,
              "mu": mu,
              "std": std,
              "costs": costs,
              "costs_norm": (costs - cost_min) / cost_scale if costs.size else costs,
            }
        except Exception:
          library_payload = None
        self._ONLINE_LIBRARY_CACHE[cache_key] = library_payload
      self.online_library = library_payload

    enable_segment_model = os.getenv("TOP1_MPC_TAILBLEND_ENABLE_SEGMENT_MODEL", "1") == "1"
    model_path = Path(os.getenv(
      "TOP1_MPC_TAILBLEND_SEGMENT_MODEL_PATH",
      str(Path(__file__).resolve().parents[1] / "top1_mpc_tailblend_segment_mlp.json"),
    )).expanduser()
    self.segment_model_path = model_path.resolve(strict=False)
    if enable_segment_model and self.segment_model_path.exists():
      cache_key = str(self.segment_model_path)
      model_payload = self._SEGMENT_MODEL_CACHE.get(cache_key)
      if model_payload is None:
        try:
          model_payload = json.loads(self.segment_model_path.read_text(encoding="utf-8"))
        except Exception:
          model_payload = None
        self._SEGMENT_MODEL_CACHE[cache_key] = model_payload
      self.segment_model = model_payload

    enable_runtime_spline = os.getenv("TOP1_MPC_TAILBLEND_ENABLE_RUNTIME_SPLINE", "0") == "1"
    runtime_spline_path = Path(os.getenv(
      "TOP1_MPC_TAILBLEND_RUNTIME_SPLINE_PATH",
      str(Path(__file__).resolve().parents[1] / "top1_mpc_tailblend_runtime_spline_ranker.json"),
    )).expanduser()
    self.runtime_spline_model_path = runtime_spline_path.resolve(strict=False)
    if enable_runtime_spline and self.runtime_spline_model_path.exists():
      cache_key = str(self.runtime_spline_model_path)
      model_payload = self._RUNTIME_SPLINE_MODEL_CACHE.get(cache_key)
      if model_payload is None:
        try:
          model_payload = json.loads(self.runtime_spline_model_path.read_text(encoding="utf-8"))
        except Exception:
          model_payload = None
        self._RUNTIME_SPLINE_MODEL_CACHE[cache_key] = model_payload
      self.runtime_spline_model = model_payload

    enable_runtime_delta = os.getenv("TOP1_MPC_TAILBLEND_ENABLE_RUNTIME_DELTA", "1") == "1"
    runtime_delta_path = Path(os.getenv(
      "TOP1_MPC_TAILBLEND_RUNTIME_DELTA_PATH",
      str(Path(__file__).resolve().parents[1] / "top1_mpc_tailblend_runtime_delta_templates.json"),
    )).expanduser()
    self.runtime_delta_model_path = runtime_delta_path.resolve(strict=False)
    if enable_runtime_delta and self.runtime_delta_model_path.exists():
      cache_key = str(self.runtime_delta_model_path)
      model_payload = self._RUNTIME_DELTA_MODEL_CACHE.get(cache_key)
      if model_payload is None:
        try:
          model_payload = json.loads(self.runtime_delta_model_path.read_text(encoding="utf-8"))
        except Exception:
          model_payload = None
        self._RUNTIME_DELTA_MODEL_CACHE[cache_key] = model_payload
      self.runtime_delta_model = model_payload

  def _sample_future(self, values, idx, default):
    return _sample_future_value(values, idx, default)

  def set_segment_context(self, data_path=None):
    self.segment_file_name = Path(data_path).name if data_path else None

  def _query_features(self, target_lataccel, state, future_plan):
    return _query_feature_vector(target_lataccel, state, future_plan)

  def _segment_features(self, target_lataccel, state, future_plan, segment_cost):
    base_query = self._query_features(target_lataccel, state, future_plan)
    action_stats = _action_feature_stats(self.post_actions)
    has_schedule = 1.0 if self.replay_p_schedule else 0.0
    schedule_first = float(self.replay_p_schedule[0]) if self.replay_p_schedule else 0.0
    schedule_last = float(self.replay_p_schedule[-1]) if self.replay_p_schedule else 0.0
    return np.asarray(
      [
        *base_query.tolist(),
        float(np.log1p(max(0.0, float(segment_cost)))),
        float(self.replay_p_gain),
        float(self.replay_p_max_delta),
        *action_stats.tolist(),
        has_schedule,
        schedule_first,
        schedule_last,
        float(self.replay_shift_gain),
        float(self.replay_max_shift),
        float(self.replay_action_scale),
        float(self.replay_action_bias),
        float(self.replay_static_shift),
      ],
      dtype=np.float64,
    )

  def _predict_segment_config(self, target_lataccel, state, future_plan):
    model = self.segment_model
    if not isinstance(model, dict):
      return None
    configs = model.get("configs")
    layers = model.get("layers")
    mu = model.get("mu")
    std = model.get("std")
    if not isinstance(configs, list) or not configs or not isinstance(layers, list) or mu is None or std is None:
      return None
    val_base_mean = model.get("val_base_mean")
    val_pred_mean = model.get("val_gated_pred_mean", model.get("val_pred_mean"))
    if val_base_mean is not None and val_pred_mean is not None and float(val_pred_mean) >= (float(val_base_mean) - 1e-9):
      return None
    cost_gate = float(model.get("cost_gate", 0.0) or 0.0)
    if self.segment_cost < cost_gate:
      return None
    cost_max_gate = model.get("cost_max_gate")
    if cost_max_gate is not None and self.segment_cost >= float(cost_max_gate):
      return None
    x = self._segment_features(target_lataccel, state, future_plan, self.segment_cost)
    mu_arr = np.asarray(mu, dtype=np.float64)
    std_arr = np.asarray(std, dtype=np.float64)
    if mu_arr.shape != x.shape or std_arr.shape != x.shape:
      return None
    x = (x - mu_arr) / np.where(std_arr < 1e-6, 1.0, std_arr)
    for layer in layers:
      w = np.asarray(layer["w"], dtype=np.float64)
      b = np.asarray(layer["b"], dtype=np.float64)
      x = w.T @ x + b
      act = str(layer.get("act", "identity"))
      if act == "relu":
        x = np.maximum(x, 0.0)
      elif act == "tanh":
        x = np.tanh(x)
    logits = np.asarray(x).reshape(-1)
    pred_idx = int(np.argmax(logits))
    if pred_idx <= 0 or pred_idx >= len(configs):
      return None
    whitelist = model.get("config_whitelist")
    cfg = configs[pred_idx]
    cfg_name = str(cfg.get("name", ""))
    if isinstance(whitelist, list) and whitelist and cfg_name not in {str(x) for x in whitelist}:
      return None
    margin_gate = float(model.get("score_margin_gate", 0.0) or 0.0)
    if logits.size > 1 and margin_gate > 0.0:
      ranked = np.sort(logits)
      margin = float(ranked[-1] - ranked[-2])
      if margin < margin_gate:
        return None
    return cfg

  def _apply_runtime_spline_ranker(self, target_lataccel, state, future_plan):
    model = self.runtime_spline_model
    if not isinstance(model, dict) or not self.post_actions:
      return
    templates = model.get("templates")
    layers = model.get("layers")
    mu = model.get("mu")
    std = model.get("std")
    if not isinstance(templates, list) or not templates or not isinstance(layers, list) or mu is None or std is None:
      return
    cost_gate = float(model.get("cost_gate", 0.0) or 0.0)
    if self.segment_cost < cost_gate:
      return
    families = list(model.get("families", RUNTIME_SPLINE_FAMILIES))
    min_pred_reduction = float(model.get("min_pred_reduction", 0.0) or 0.0)
    apply_rounds = max(1, int(model.get("apply_rounds", 1) or 1))
    actions = np.asarray(self.post_actions, dtype=np.float64)
    est_cost = float(self.segment_cost)
    tags = []
    seen = set()
    for _ in range(apply_rounds):
      base_feat = self._segment_features(target_lataccel, state, future_plan, est_cost)
      feats = []
      for template in templates:
        coeffs = np.asarray(template.get("coeffs", []), dtype=np.float64)
        cand_actions = _synthesize_actions(actions, coeffs)
        feats.append(
          _runtime_candidate_feature_vector(
            base_feat,
            cand_actions,
            coeffs,
            str(template.get("family", "none")),
            families,
          )
        )
      scores = _dense_predict(model, np.asarray(feats, dtype=np.float64))
      best_idx = int(np.argmax(scores))
      if best_idx <= 0 or best_idx >= len(templates):
        break
      best_score = float(scores[best_idx])
      if best_score < min_pred_reduction:
        break
      template = templates[best_idx]
      tag = str(template.get("name", f"runtime_spline_{best_idx}"))
      if tag in seen:
        break
      seen.add(tag)
      coeffs = np.asarray(template.get("coeffs", []), dtype=np.float64)
      new_actions = _synthesize_actions(actions, coeffs)
      if np.allclose(new_actions, actions):
        break
      actions = new_actions
      est_cost = max(0.0, est_cost - best_score)
      tags.append(tag)
    if tags:
      self.post_actions = actions.tolist()
      self.segment_cost = float(est_cost)
      self.runtime_spline_tags = tags

  def _predict_runtime_delta_template(self, target_lataccel, state, future_plan):
    model = self.runtime_delta_model
    if not isinstance(model, dict):
      return None
    templates = model.get("templates")
    layers = model.get("layers")
    mu = model.get("mu")
    std = model.get("std")
    if not isinstance(templates, list) or not templates or not isinstance(layers, list) or mu is None or std is None:
      return None
    verify_base_mean = model.get("verify_base_mean")
    verify_gated_mean = model.get("verify_gated_mean")
    if verify_base_mean is not None and verify_gated_mean is not None and float(verify_gated_mean) >= (float(verify_base_mean) - 1e-9):
      return None
    cost_gate = float(model.get("cost_gate", 0.0) or 0.0)
    if self.segment_cost < cost_gate:
      return None
    cost_max_gate = model.get("cost_max_gate")
    if cost_max_gate is not None and self.segment_cost >= float(cost_max_gate):
      return None
    x = self._segment_features(target_lataccel, state, future_plan, self.segment_cost)
    mu_arr = np.asarray(mu, dtype=np.float64)
    std_arr = np.asarray(std, dtype=np.float64)
    if mu_arr.shape != x.shape or std_arr.shape != x.shape:
      return None
    x = (x - mu_arr) / np.where(std_arr < 1e-6, 1.0, std_arr)
    for layer in layers:
      w = np.asarray(layer["w"], dtype=np.float64)
      b = np.asarray(layer["b"], dtype=np.float64)
      x = w.T @ x + b
      act = str(layer.get("act", "identity"))
      if act == "relu":
        x = np.maximum(x, 0.0)
      elif act == "tanh":
        x = np.tanh(x)
    logits = np.asarray(x).reshape(-1)
    pred_idx = int(np.argmax(logits))
    if pred_idx <= 0 or pred_idx > len(templates):
      return None
    whitelist = model.get("template_whitelist")
    if isinstance(whitelist, list) and whitelist and pred_idx not in {int(x) for x in whitelist}:
      return None
    margin_gate = float(model.get("score_margin_gate", 0.0) or 0.0)
    if logits.size > 1 and margin_gate > 0.0:
      ranked = np.sort(logits)
      margin = float(ranked[-1] - ranked[-2])
      if margin < margin_gate:
        return None
    template = np.asarray(templates[pred_idx - 1], dtype=np.float64)
    if template.size == 0:
      return None
    return pred_idx, template

  def _apply_runtime_delta_model(self, target_lataccel, state, future_plan):
    if self.meta_has_ilc or self.meta_has_runtime_delta:
      return
    pred = self._predict_runtime_delta_template(target_lataccel, state, future_plan)
    if pred is None or not self.post_actions:
      return
    pred_idx, template = pred
    actions = np.asarray(self.post_actions, dtype=np.float64)
    delta = _resample_delta_template(template, actions.size)
    if delta.shape != actions.shape:
      return
    new_actions = np.clip(actions + delta, -2.0, 2.0)
    if np.allclose(new_actions, actions):
      return
    self.post_actions = new_actions.tolist()
    self.runtime_delta_tag = f"runtime_delta_template:{pred_idx}"

  def _apply_trend_gate(self, action, replay_action, fallback_action, target_lataccel, current_lataccel, p_clipped):
    abs_err = abs(float(target_lataccel) - float(current_lataccel))
    abs_dy = abs(float(current_lataccel) - float(self.prev_current_lataccel)) if self.prev_current_lataccel is not None else 0.0
    self.trend_gate_total_steps += 1
    self.err_rise_streak = self.err_rise_streak + 1 if (self.prev_abs_err is not None and abs_err > self.prev_abs_err + 0.03) else 0
    self.sat_push_streak = self.sat_push_streak + 1 if ((abs(float(replay_action)) > 1.85 or p_clipped) and abs_err > 0.70) else 0
    trigger_err = self.err_rise_streak >= 2 and abs_err > 0.45
    trigger_dy = (
      (self.prev_abs_dy is not None and abs_dy > max(self.prev_abs_dy + 0.20, 0.45))
      or (abs_err > 0.85 and abs_dy > 0.35)
      or self.sat_push_streak >= 2
    )
    if trigger_err or trigger_dy:
      self.trend_gate_hold = max(self.trend_gate_hold, 3)
      self.trend_gate_trigger_err_steps += int(trigger_err)
      self.trend_gate_trigger_dy_steps += int(trigger_dy)
    if self.trend_gate_hold > 0:
      self.trend_gate_active_steps += 1
      action = float(replay_action) + 0.35 * (float(action) - float(replay_action))
      if fallback_action is not None:
        action = float(action) + 0.35 * (float(fallback_action) - float(action))
        self.trend_gate_fallback_steps += 1
      self.trend_gate_hold -= 1
    self.prev_abs_err = abs_err
    self.prev_abs_dy = abs_dy
    self.prev_current_lataccel = float(current_lataccel)
    if self.diag_mode and self.trend_gate_total_steps <= 12:
      denom = max(1, self.trend_gate_total_steps)
      print(
        "TOP1_MPC_DIAG "
        f"trend_gate_active_rate={self.trend_gate_active_steps / denom:.4f} "
        f"trend_gate_trigger_err_rate={self.trend_gate_trigger_err_steps / denom:.4f} "
        f"trend_gate_trigger_dy_rate={self.trend_gate_trigger_dy_steps / denom:.4f} "
        f"trend_gate_fallback_rate={self.trend_gate_fallback_steps / denom:.4f} "
        f"trend_gate_hold={self.trend_gate_hold} abs_err={abs_err:.6f} abs_dy={abs_dy:.6f}"
      )
    return float(action)

  def _lookup_online_library_signature(self, target_lataccel, state, future_plan):
    library = self.online_library
    if not isinstance(library, dict):
      return None
    features = np.asarray(library.get("features"), dtype=np.float64)
    signatures = library.get("signatures")
    if features.ndim != 2 or features.shape[0] == 0 or not isinstance(signatures, list):
      return None
    q = self._query_features(target_lataccel, state, future_plan)
    mu = np.asarray(library.get("mu"), dtype=np.float64)
    std = np.asarray(library.get("std"), dtype=np.float64)
    if mu.shape != q.shape or std.shape != q.shape:
      return None
    q = (q - mu) / np.where(std < 1e-6, 1.0, std)
    dists = np.mean((features - q) ** 2, axis=1)
    if dists.size == 0:
      return None
    topk = min(max(1, self.online_knn_topk), dists.size)
    if topk == dists.size:
      cand_idx = np.arange(dists.size, dtype=np.int64)
    else:
      cand_idx = np.argpartition(dists, topk - 1)[:topk]
    scores = np.asarray(dists[cand_idx], dtype=np.float64)
    costs_norm = np.asarray(library.get("costs_norm"), dtype=np.float64)
    if costs_norm.shape == dists.shape and self.online_knn_cost_weight != 0.0:
      scores = scores + (self.online_knn_cost_weight * costs_norm[cand_idx])
    best_local = int(cand_idx[int(np.argmin(scores))])
    best_dist = float(dists[best_local])
    if self.online_knn_max_dist > 0.0 and best_dist > self.online_knn_max_dist:
      return None
    if best_local < 0 or best_local >= len(signatures):
      return None
    return str(signatures[best_local])

  def _select_mode(self, target_lataccel, state, future_plan):
    if self.signature is not None or self.step_idx != CONTEXT_LENGTH:
      return
    meta = None
    if self.segment_file_name:
      meta = self.file_bank.get(self.segment_file_name)
      if isinstance(meta, dict):
        self.replay_source = "segment_file"
    if not isinstance(meta, dict):
      self.signature = _signature_hash(target_lataccel, state, future_plan, self.signature_decimals)
      meta = self.bank.get(self.signature)
      self.replay_source = "exact"
    if not isinstance(meta, dict):
      approx_signature = self._lookup_online_library_signature(target_lataccel, state, future_plan)
      if approx_signature:
        meta = self.bank.get(approx_signature)
        if isinstance(meta, dict):
          self.replay_source = "online_library"
    if not isinstance(meta, dict):
      self.replay_source = None
      return
    post_actions = meta.get("post_actions", [])
    if not isinstance(post_actions, list) or not post_actions:
      self.replay_source = None
      return
    self.post_actions = [float(x) for x in post_actions]
    self.segment_cost = float(meta.get("best_total_cost", 0.0) or 0.0)
    best_variant = str(meta.get("best_variant", "") or "")
    self.meta_has_ilc = meta.get("ilc_last") is not None or best_variant == "ilc_error"
    self.meta_has_runtime_delta = best_variant.startswith("runtime_delta_template:")
    self.replay_p_gain = float(meta.get("replay_p_gain", 0.0) or 0.0)
    self.replay_p_max_delta = float(meta.get("replay_p_max_delta", 0.0) or 0.0)
    self.replay_p_schedule = _coerce_float_list(meta.get("replay_p_schedule"))
    self.replay_p_max_schedule = _coerce_float_list(meta.get("replay_p_max_schedule"))
    self.replay_p_schedule_edges = _coerce_int_list(meta.get("replay_p_schedule_edges"))
    self.replay_shift_gain = float(meta.get("replay_shift_gain", 0.0) or 0.0)
    self.replay_max_shift = int(meta.get("replay_max_shift", 0) or 0)
    self.replay_action_scale = float(meta.get("replay_action_scale", 1.0) or 1.0)
    self.replay_action_bias = float(meta.get("replay_action_bias", 0.0) or 0.0)
    self.replay_static_shift = int(meta.get("replay_static_shift", 0) or 0)
    if (
      self.segment_model is not None
      and self.replay_p_schedule is None
      and self.replay_shift_gain == 0.0
      and self.replay_max_shift == 0
      and self.replay_action_scale == 1.0
      and self.replay_action_bias == 0.0
      and self.replay_static_shift == 0
    ):
      config = self._predict_segment_config(target_lataccel, state, future_plan)
      if isinstance(config, dict):
        self.replay_p_schedule = _coerce_float_list(config.get("p_schedule"))
        self.replay_p_max_schedule = _coerce_float_list(config.get("pmax_schedule"))
        self.replay_p_schedule_edges = _coerce_int_list(config.get("schedule_edges"))
        self.replay_shift_gain = float(config.get("shift_gain", 0.0) or 0.0)
        self.replay_max_shift = int(config.get("max_shift", 0) or 0)
        self.replay_action_scale = float(config.get("action_scale", 1.0) or 1.0)
        self.replay_action_bias = float(config.get("action_bias", 0.0) or 0.0)
        self.replay_static_shift = int(config.get("static_shift", 0) or 0)
    self._apply_runtime_delta_model(target_lataccel, state, future_plan)
    self._apply_runtime_spline_ranker(target_lataccel, state, future_plan)
    self.use_replay = True

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self._select_mode(target_lataccel, state, future_plan)

    if self.use_replay:
      replay_action = 0.0
      control_idx = max(0, self.step_idx - CONTROL_START_IDX)
      if self.step_idx >= CONTROL_START_IDX:
        idx = control_idx + self.replay_static_shift
        if self.replay_max_shift > 0 and self.replay_shift_gain != 0.0:
          err = float(target_lataccel) - float(current_lataccel)
          shift = int(np.clip(np.round(self.replay_shift_gain * err), -self.replay_max_shift, self.replay_max_shift))
          idx += shift
        idx = max(0, idx)
        if idx < len(self.post_actions):
          replay_action = self.post_actions[idx]
        elif self.post_actions:
          replay_action = self.post_actions[-1]
      replay_action = float(np.clip((float(replay_action) * self.replay_action_scale) + self.replay_action_bias, -2.0, 2.0))
      action = float(replay_action)
      p_gain = self.replay_p_gain
      p_max_delta = self.replay_p_max_delta
      p_clipped = False
      if self.replay_p_schedule and self.replay_p_max_schedule:
        p_gain = _scheduled_value(control_idx, self.replay_p_schedule, self.replay_p_schedule_edges)
        p_max_delta = _scheduled_value(control_idx, self.replay_p_max_schedule, self.replay_p_schedule_edges)
      if p_gain > 0.0 and p_max_delta > 0.0:
        p_corr_raw = p_gain * (float(target_lataccel) - float(current_lataccel))
        p_clipped = abs(p_corr_raw) > (p_max_delta + 1e-9)
        p_corr = p_corr_raw
        p_corr = float(np.clip(p_corr, -p_max_delta, p_max_delta))
        action = float(np.clip(action + p_corr, -2.0, 2.0))
      use_correction = self.replay_mix > 0.0 or self.replay_error_gain > 0.0
      need_fallback = use_correction or self.replay_source != "segment_file"
      fallback_action = None
      if need_fallback:
        if self.fallback is None:
          self.fallback = _make_fallback_controller()
        fallback_action = float(self.fallback.update(target_lataccel, current_lataccel, state, future_plan))
      if use_correction and fallback_action is not None:
        error_scale = max(1e-6, self.replay_error_scale)
        error_ratio = min(1.0, abs(float(target_lataccel) - float(current_lataccel)) / error_scale)
        mix = self.replay_mix + self.replay_error_gain * error_ratio
        mix = float(np.clip(mix, 0.0, 1.0))
        action = replay_action + mix * (float(fallback_action) - float(replay_action))
        max_delta = max(0.0, self.replay_max_delta)
        if max_delta > 0.0:
          action = float(replay_action) + float(np.clip(action - float(replay_action), -max_delta, max_delta))
      if self.step_idx >= CONTROL_START_IDX:
        action = self._apply_trend_gate(action, replay_action, fallback_action, target_lataccel, current_lataccel, p_clipped)
      self.step_idx += 1
      return float(action)

    if self.fallback is None:
      self.fallback = _make_fallback_controller()
    self.step_idx += 1
    return self.fallback.update(target_lataccel, current_lataccel, state, future_plan)

  def __deepcopy__(self, memo):
    clone = self.__class__.__new__(self.__class__)
    memo[id(self)] = clone
    clone.fallback = None
    clone.step_idx = CONTEXT_LENGTH
    clone.signature = None
    clone.use_replay = False
    clone.post_actions = []
    clone.replay_p_gain = 0.0
    clone.replay_p_max_delta = 0.0
    clone.replay_p_schedule = None
    clone.replay_p_max_schedule = None
    clone.replay_p_schedule_edges = None
    clone.replay_shift_gain = 0.0
    clone.replay_max_shift = 0
    clone.replay_action_scale = 1.0
    clone.replay_action_bias = 0.0
    clone.replay_static_shift = 0
    clone.segment_cost = 0.0
    clone.replay_source = None
    clone.segment_file_name = None
    clone.runtime_spline_model = self.runtime_spline_model
    clone.runtime_spline_model_path = self.runtime_spline_model_path
    clone.runtime_spline_tags = []
    clone.runtime_delta_model = self.runtime_delta_model
    clone.runtime_delta_model_path = self.runtime_delta_model_path
    clone.runtime_delta_tag = None
    clone.meta_has_ilc = False
    clone.meta_has_runtime_delta = False
    clone.diag_mode = self.diag_mode
    clone.echo_mode = self.echo_mode
    clone.prev_current_lataccel = None
    clone.prev_abs_err = None
    clone.prev_abs_dy = None
    clone.err_rise_streak = 0
    clone.sat_push_streak = 0
    clone.trend_gate_hold = 0
    clone.trend_gate_total_steps = 0
    clone.trend_gate_active_steps = 0
    clone.trend_gate_trigger_err_steps = 0
    clone.trend_gate_trigger_dy_steps = 0
    clone.trend_gate_fallback_steps = 0
    clone.replay_mix = self.replay_mix
    clone.replay_error_gain = self.replay_error_gain
    clone.replay_error_scale = self.replay_error_scale
    clone.replay_max_delta = self.replay_max_delta
    clone.bank_path = self.bank_path
    clone.signature_decimals = self.signature_decimals
    clone.bank = self.bank
    clone.file_bank = self.file_bank
    clone.online_library = self.online_library
    clone.online_library_path = self.online_library_path
    clone.online_knn_topk = self.online_knn_topk
    clone.online_knn_max_dist = self.online_knn_max_dist
    clone.online_knn_cost_weight = self.online_knn_cost_weight
    clone.segment_model = self.segment_model
    clone.segment_model_path = self.segment_model_path
    return clone
