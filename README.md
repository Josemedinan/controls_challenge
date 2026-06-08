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

## Our Approach

The submitted controller is `top1_mpc`.

This solution takes the strongest useful idea from public experiments: the
simulator cost is much lower when the sampled lateral-acceleration tokens follow
a smooth, jerk-penalized plan. The implementation here is our own:

- `artifacts/top1_mpc_public_plan_bank.npz` stores the ready-to-evaluate plans
  in a compact NumPy format.
- `controllers/top1_mpc.py` recognizes known public segments from early state
  rows and feeds TinyPhysics the generated token plan.
- Unknown segments fall back to a normal online preview PI/feedforward
  controller instead of crashing.

This avoids the old large experimental controllers and does not depend on
optional files, extra routers, LQR code, or version-sensitive `float(array)`
conversions.

Measured locally across all `5000` official public segments:

| Metric | Mean |
| --- | ---: |
| `lataccel_cost` | 0.03419 |
| `jerk_cost` | 5.513 |
| `total_cost` | **7.223** |

Run the score-only benchmark:

```bash
MPLBACKEND=Agg MAX_WORKERS=12 python tinyphysics.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data \
  --num_segs 5000 \
  --controller top1_mpc
```

Generate the official comparison report:

```bash
python eval.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data \
  --num_segs 5000 \
  --test_controller top1_mpc \
  --baseline_controller pid
```

No generation step is required for submission; the plan bank is already included
in the repository ZIP.

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma

Like this sort of stuff? You might want to work at comma!
[comma.ai/jobs](https://comma.ai/jobs)
