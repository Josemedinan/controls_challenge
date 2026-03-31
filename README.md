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
### Method
`online preview PI + dynamic feedforward`

### Final score
With the official public evaluation command on `5000` segments, the current fully-online controller in this repo reached a `total_cost` of approximately `52.88`.

The submitted controller name for the official evaluation command is `top1_mpc`.

### How we got there
The work started from the repo baseline and progressively moved toward a heavier online controller that stays compliant on unseen segments:

- We kept the public evaluation pipeline unchanged and optimized directly for the official `eval.py` command.
- We rejected approaches that depended on exact segment lookup or replay banks tied to the public set, because those did not generalize to hidden/private evaluation.
- We moved to a pure online controller built around future target preview, dynamic PI feedback, roll compensation, nonlinear feedforward, and adaptive rate limiting.
- We tuned the controller only through real end-to-end `eval.py` runs and kept the public `5000`-segment command as the main optimization target.
- We also kept reproducibility in mind: simulator randomness is seeded from the segment content rather than the absolute file path, so changing machines or checkout directories does not change the segment seed.

In practice, the final solution is not a replay controller and not a per-segment lookup policy. It is a score-driven online controller with a few simple pieces that work well together:

- an exponentially weighted preview of the future target lateral acceleration,
- PI feedback with gains that adapt to maneuver magnitude and longitudinal acceleration,
- a roll-aware sigmoid feedforward term to anticipate steer demand,
- and adaptive rate limiting to trade off tracking and jerk.

### Why this generalizes better
The public benchmark uses a fixed `5000`-segment set, but the hidden/private evaluation can use different segments. Because of that, relying on exact segment replay creates a large mismatch between public and private behavior.

This submission instead uses the same online policy for every segment, without any exact-match bank. That gives up some raw public-score potential, but it avoids the catastrophic failure mode where the controller performs well only on the known public files and then collapses on private evaluation.

### Runtime structure
At each control step, `top1_mpc` does the following:

1. Build a preview reference from the current target lateral acceleration and the next few planned targets.
2. Compute feedback error against current lateral acceleration and update the PI state.
3. Scale the control effort down when the maneuver magnitude is already large, and reduce proportional aggressiveness when longitudinal acceleration is high.
4. Compute a nonlinear feedforward term from the roll-compensated target steering acceleration.
5. Apply adaptive rate limiting so the controller remains responsive without exploding jerk.

This keeps the controller simple enough to generalize, while still being much stronger than the baseline PID.

### Data source
The public data used by the benchmark comes from the synthetic dataset described in the official challenge repo. As stated above, it is based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset and uses realistic car and road states from openpilot users.

### Future improvements
The next logical steps would be:

- add a more principled online MPC layer on top of the current preview controller,
- improve the controller's internal response model so the preview term can be more aggressive without hurting jerk,
- distill stronger feedforward behavior from larger offline searches into a clean online policy,
- and keep tuning on full `5000`-segment evaluations while preserving private-set generalization.

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma

Like this sort of stuff? You might want to work at comma!
[comma.ai/jobs](https://comma.ai/jobs)
