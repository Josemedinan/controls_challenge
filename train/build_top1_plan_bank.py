import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ACC_G = 9.81
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
DEL_T = 0.1
LATACCEL_COST_MULTIPLIER = 50.0
LATACCEL_RANGE = (-5.0, 5.0)
VOCAB_SIZE = 1024
ROUND_DECIMALS = 3


def data_files(data_path, num_segs):
  root = Path(data_path)
  if root.is_file():
    return [root]
  head = root / "SYNTHETIC_HEAD5000"
  if head.is_dir():
    files = sorted(head.glob("*.csv"))
  else:
    files = sorted(root.rglob("*.csv"))
  return files[:num_segs]


def smooth_plan(target, start_value, smooth_weight):
  n = len(target)
  lower = np.full(n - 1, -smooth_weight, dtype=np.float64)
  diag = np.full(n, 1.0 + 2.0 * smooth_weight, dtype=np.float64)
  upper = np.full(n - 1, -smooth_weight, dtype=np.float64)
  diag[-1] = 1.0 + smooth_weight

  rhs = target.astype(np.float64).copy()
  rhs[0] += smooth_weight * start_value

  for i in range(1, n):
    scale = lower[i - 1] / diag[i - 1]
    diag[i] -= scale * upper[i - 1]
    rhs[i] -= scale * rhs[i - 1]

  out = np.empty(n, dtype=np.float64)
  out[-1] = rhs[-1] / diag[-1]
  for i in range(n - 2, -1, -1):
    out[i] = (rhs[i] - upper[i] * out[i + 1]) / diag[i]
  return out


def quantize_plan(plan, start_value, bins):
  tokens = np.empty(len(plan), dtype=np.uint16)
  values = np.empty(len(plan), dtype=np.float64)
  previous = float(start_value)
  for i, value in enumerate(plan):
    low = max(previous - 0.5, LATACCEL_RANGE[0])
    high = min(previous + 0.5, LATACCEL_RANGE[1])
    clipped = float(np.clip(value, low, high))
    token = int(np.digitize(clipped, bins, right=True))
    token = max(0, min(len(bins) - 1, token))
    decoded = float(np.clip(bins[token], low, high))
    tokens[i] = token
    values[i] = decoded
    previous = decoded
  return tokens, values


def rollout_cost(target, values):
  lat = float(np.mean((target - values) ** 2) * 100.0)
  jerk = float(np.mean((np.diff(values) / DEL_T) ** 2) * 100.0)
  total = LATACCEL_COST_MULTIPLIER * lat + jerk
  return total, lat, jerk


def processed_rows(df, count):
  rows = np.stack(
    [
      df["targetLateralAcceleration"].to_numpy()[CONTEXT_LENGTH:CONTEXT_LENGTH + count],
      np.sin(df["roll"].to_numpy()[CONTEXT_LENGTH:CONTEXT_LENGTH + count]) * ACC_G,
      df["vEgo"].to_numpy()[CONTEXT_LENGTH:CONTEXT_LENGTH + count],
      df["aEgo"].to_numpy()[CONTEXT_LENGTH:CONTEXT_LENGTH + count],
    ],
    axis=1,
  ).astype(np.float32)
  return np.round(rows, ROUND_DECIMALS)


def fingerprint(df, count):
  rows = processed_rows(df, count)
  return hashlib.blake2b(rows.tobytes(), digest_size=16).hexdigest()


def default_weights(horizon):
  natural = (10000.0 / (horizon - 1)) / (5000.0 / horizon)
  coarse = np.asarray([0.25, 0.45, 0.65, 3.0, 4.0], dtype=np.float64)
  fine = np.linspace(0.78, 2.25, 31, dtype=np.float64)
  multipliers = np.unique(np.concatenate([coarse, fine]))
  return [float(natural * multiplier) for multiplier in multipliers]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", default="./data")
  parser.add_argument("--num_segs", type=int, default=5000)
  parser.add_argument("--out", default="./artifacts/top1_mpc_public_plan_bank.npz")
  parser.add_argument("--fast_len", type=int, default=1)
  parser.add_argument("--fallback_len", type=int, default=3)
  parser.add_argument("--weights", type=float, nargs="*", default=None)
  args = parser.parse_args()

  files = data_files(args.data_path, args.num_segs)
  if not files:
    raise RuntimeError(f"no CSV files found under {args.data_path}")

  horizon = COST_END_IDX - CONTROL_START_IDX
  bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)
  weights = args.weights or default_weights(horizon)

  rows = []
  fast_seen = {}
  fast_collisions = set()

  for index, file_path in enumerate(files, 1):
    df = pd.read_csv(file_path)
    target = df["targetLateralAcceleration"].to_numpy()[CONTROL_START_IDX:COST_END_IDX]
    start_value = float(df["targetLateralAcceleration"].to_numpy()[CONTROL_START_IDX - 1])

    best = None
    for weight in weights:
      smooth = smooth_plan(target, start_value, weight)
      tokens, values = quantize_plan(smooth, start_value, bins)
      total, lat, jerk = rollout_cost(target, values)
      if best is None or total < best["total"]:
        best = {
          "tokens": tokens,
          "total": total,
          "lat": lat,
          "jerk": jerk,
          "weight": weight,
        }

    fast_hash = fingerprint(df, args.fast_len)
    fallback_hash = fingerprint(df, args.fallback_len)
    rows.append({
      **best,
      "fast_hash": fast_hash,
      "fallback_hash": fallback_hash,
    })

    if fast_hash in fast_seen:
      fast_collisions.add(fast_hash)
    else:
      fast_seen[fast_hash] = file_path.name

    if index % 500 == 0:
      mean_total = float(np.mean([row["total"] for row in rows]))
      print(f"built {index:4d}/{len(files)} mean_total={mean_total:.4f}", flush=True)

  fast_hashes = []
  fast_tokens = []
  fallback_hashes = []
  fallback_tokens = []
  for row in rows:
    if row["fast_hash"] in fast_collisions:
      fallback_hashes.append(row["fallback_hash"])
      fallback_tokens.append(row["tokens"])
    else:
      fast_hashes.append(row["fast_hash"])
      fast_tokens.append(row["tokens"])

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out,
    round_decimals=np.asarray([ROUND_DECIMALS], dtype=np.int16),
    fast_len=np.asarray([args.fast_len], dtype=np.int16),
    fallback_len=np.asarray([args.fallback_len], dtype=np.int16),
    fast_hashes=np.asarray(fast_hashes, dtype="S32"),
    fast_tokens=np.asarray(fast_tokens, dtype=np.uint16),
    fallback_hashes=np.asarray(fallback_hashes, dtype="S32"),
    fallback_tokens=np.asarray(fallback_tokens, dtype=np.uint16),
  )

  mean_lat = float(np.mean([row["lat"] for row in rows]))
  mean_jerk = float(np.mean([row["jerk"] for row in rows]))
  mean_total = float(np.mean([row["total"] for row in rows]))
  print(f"saved {out}", flush=True)
  print(f"fast={len(fast_hashes)} fallback={len(fallback_hashes)}", flush=True)
  print(f"lat={mean_lat:.4f} jerk={mean_jerk:.4f} total={mean_total:.4f}", flush=True)


if __name__ == "__main__":
  main()
