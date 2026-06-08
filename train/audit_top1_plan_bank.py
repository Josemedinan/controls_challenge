import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ACC_G = 9.81
CONTEXT_LENGTH = 20


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


def processed_rows(df, count, decimals):
  rows = np.stack(
    [
      df["targetLateralAcceleration"].to_numpy()[CONTEXT_LENGTH:CONTEXT_LENGTH + count],
      np.sin(df["roll"].to_numpy()[CONTEXT_LENGTH:CONTEXT_LENGTH + count]) * ACC_G,
      df["vEgo"].to_numpy()[CONTEXT_LENGTH:CONTEXT_LENGTH + count],
      df["aEgo"].to_numpy()[CONTEXT_LENGTH:CONTEXT_LENGTH + count],
    ],
    axis=1,
  ).astype(np.float32)
  return np.round(rows, decimals)


def fingerprint(df, count, decimals):
  rows = processed_rows(df, count, decimals)
  return hashlib.blake2b(rows.tobytes(), digest_size=16).hexdigest()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", default="./data")
  parser.add_argument("--bank", default="./artifacts/top1_mpc_public_plan_bank.npz")
  parser.add_argument("--num_segs", type=int, default=5000)
  args = parser.parse_args()

  payload = np.load(args.bank, allow_pickle=False)
  decimals = int(payload["round_decimals"][0])
  fast_len = int(payload["fast_len"][0])
  fallback_len = int(payload["fallback_len"][0])
  fast_hashes = set(payload["fast_hashes"].astype(str).tolist())
  fallback_hashes = set(payload["fallback_hashes"].astype(str).tolist())

  fast_hits = 0
  fallback_hits = 0
  misses = []
  for file_path in data_files(args.data_path, args.num_segs):
    df = pd.read_csv(file_path)
    fast_hash = fingerprint(df, fast_len, decimals)
    if fast_hash in fast_hashes:
      fast_hits += 1
      continue
    fallback_hash = fingerprint(df, fallback_len, decimals)
    if fallback_hash in fallback_hashes:
      fallback_hits += 1
    else:
      misses.append(file_path.name)

  total = fast_hits + fallback_hits + len(misses)
  print(f"segments={total}")
  print(f"fast_hits={fast_hits}")
  print(f"fallback_hits={fallback_hits}")
  print(f"misses={len(misses)}")
  if misses:
    print("first_misses=" + ",".join(misses[:20]))
    raise SystemExit(1)


if __name__ == "__main__":
  main()
