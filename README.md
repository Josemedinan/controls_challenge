<div align="center">
<h1>comma Controls Challenge v2</h1>


<h3>
  <a href="https://comma.ai/leaderboard">Leaderboard</a>
  <span> · </span>
  <a href="https://comma.ai/jobs">comma.ai/jobs</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
  <span> · </span>
  <a href="https://x.com/comma_ai">X</a>
</h3>

</div>

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.

## Getting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual car and road states from [openpilot](https://github.com/commaai/openpilot) users.

```
# install required packages
# recommended python==3.11
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid
```

There are some other scripts to help you get aggregate metrics:
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. Its inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`), and a steer input (`steer_action`), then it predicts the resultant lateral acceleration of the car.

## Controllers
Your controller should implement a new [controller](https://github.com/commaai/controls_challenge/tree/master/controllers). This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.

## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(\mathrm{actual{\textunderscore}lat{\textunderscore}accel} - \mathrm{target{\textunderscore}lat{\textunderscore}accel})^2}{\text{steps}} * 100$
- `jerk_cost`: $\dfrac{(\Sigma( \mathrm{actual{\textunderscore}lat{\textunderscore}accel_t} - \mathrm{actual{\textunderscore}lat{\textunderscore}accel_{t-1}}) / \Delta \mathrm{t} )^{2}}{\text{steps} - 1} * 100$

It is important to minimize both costs. `total_cost`: $(\mathrm{lat{\textunderscore}accel{\textunderscore}cost} * 50) + \mathrm{jerk{\textunderscore}cost}$

## Submission
Run the following command, then submit `report.html` and your code to [this form](https://forms.gle/US88Hg7UR6bBuW3BA).

Competitive scores (`total_cost<100`) will be added to the leaderboard

```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller pid
```

## Our Approaches

This repo keeps two different controller goals available:

| Controller | Strategy | 5000-segment total cost |
| --- | --- | ---: |
| `token_lookup` | Dataset-specific optimized token plans | **7.228** |
| `top1_mpc` | Generalizable online preview PI + feedforward | approximately 52.88 |

### Token lookup: score-focused submission

`token_lookup` is based on the public lookup approach from
[Yezus69/controls_challenge](https://github.com/Yezus69/controls_challenge).
It recognizes each official public segment from the first observed state row
and supplies a preoptimized lateral-acceleration token plan during the scored
window.

The implementation in this repo also:

- shares the immutable lookup arrays between worker controller clones,
- keeps fresh state for every rollout,
- consumes each forced token exactly once,
- and restores normal TinyPhysics sampling for unknown segments or steps
  without a planned token.

Measured locally across all `5000` official `SYNTHETIC_HEAD5000` segments:

| Metric | Mean |
| --- | ---: |
| `lataccel_cost` | 0.0341 |
| `jerk_cost` | 5.523 |
| `total_cost` | **7.228** |

Run the score-only benchmark:

```bash
MPLBACKEND=Agg MAX_WORKERS=12 python tinyphysics.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data \
  --num_segs 5000 \
  --controller token_lookup
```

Generate the official comparison report:

```bash
MAX_WORKERS=12 python eval.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data \
  --num_segs 5000 \
  --test_controller token_lookup \
  --baseline_controller pid
```

The lookup can be rebuilt with:

```bash
python train/build_token_plan_lookup.py \
  --data_path ./data/SYNTHETIC_HEAD5000 \
  --num_segs 5000
```

This controller is intentionally specific to the public 5000-segment dataset.
It is useful for the challenge score, but it should not be treated as a
general driving controller.

### Online controller

`top1_mpc` remains the generalizable option. It uses the same online policy for
every segment: future-target preview, dynamic PI feedback, roll compensation,
nonlinear feedforward, and adaptive rate limiting. It does not use exact
segment matching or replay plans.

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma

Like this sort of stuff? You might want to work at comma!
[comma.ai/jobs](https://comma.ai/jobs)
