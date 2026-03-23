import argparse
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from tinyphysics import list_data_files, run_rollout_pair


def parse_seeds(raw: str):
  parts = [chunk.strip() for chunk in str(raw).split(",")]
  seeds = []
  for part in parts:
    if not part:
      continue
    seeds.append(int(part))
  if not seeds:
    raise ValueError("No seeds were provided")
  return seeds


def get_max_workers(default=None):
  if default is None:
    cpu_count = os.cpu_count() or 4
    default = min(12, max(1, cpu_count))
  raw = os.getenv("MAX_WORKERS")
  if raw is None:
    return default
  try:
    value = int(raw)
    return value if value > 0 else default
  except ValueError:
    return default


def sample_files(files, n, seed):
  if n > len(files):
    raise ValueError(f"Requested n={n}, but only {len(files)} files are available")
  rng = random.Random(seed)
  idxs = rng.sample(range(len(files)), n)
  return [files[idx] for idx in idxs]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--n", type=int, default=1000)
  parser.add_argument("--seeds", type=str, required=True)
  parser.add_argument("--test_controller", type=str, required=True)
  parser.add_argument("--baseline_controller", type=str, required=True)
  parser.add_argument("--chunksize", type=int, default=10)
  args = parser.parse_args()

  data_path = Path(args.data_path)
  assert data_path.exists(), "data_path should exist"

  files = list_data_files(data_path)
  seeds = parse_seeds(args.seeds)
  workers = get_max_workers()

  rollout_pair_partial = partial(
    run_rollout_pair,
    test_controller_type=args.test_controller,
    baseline_controller_type=args.baseline_controller,
    model_path=args.model_path,
    debug=False,
  )

  rows = []
  print(f"files_available={len(files)} n={args.n} workers={workers} seeds={seeds}")
  for seed in seeds:
    chosen = sample_files(files, args.n, seed)
    results = process_map(
      rollout_pair_partial,
      chosen,
      max_workers=workers,
      chunksize=max(1, args.chunksize),
    )
    test_costs = np.asarray([result[0][0]["total_cost"] for result in results], dtype=np.float64)
    baseline_costs = np.asarray([result[1][0]["total_cost"] for result in results], dtype=np.float64)
    test_mean = float(np.mean(test_costs))
    baseline_mean = float(np.mean(baseline_costs))
    delta = baseline_mean - test_mean
    rows.append(
      {
        "seed": int(seed),
        "test_mean": test_mean,
        "baseline_mean": baseline_mean,
        "delta": float(delta),
      }
    )
    print(
      f"seed={seed} "
      f"test_mean={test_mean:.6f} "
      f"baseline_mean={baseline_mean:.6f} "
      f"delta={delta:.6f}"
    )

  test_mean_across = float(np.mean([row["test_mean"] for row in rows]))
  baseline_mean_across = float(np.mean([row["baseline_mean"] for row in rows]))
  mean_delta = float(np.mean([row["delta"] for row in rows]))
  robust_delta_min = float(np.min([row["delta"] for row in rows]))
  robust_delta_p25 = float(np.percentile([row["delta"] for row in rows], 25))

  print(f"test_mean_across_seeds={test_mean_across:.6f}")
  print(f"baseline_mean_across_seeds={baseline_mean_across:.6f}")
  print(f"mean_delta={mean_delta:.6f}")
  print(f"robust_delta_min={robust_delta_min:.6f}")
  print(f"robust_delta_p25={robust_delta_p25:.6f}")


if __name__ == "__main__":
  main()
