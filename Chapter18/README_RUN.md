# MountainCar Code Guide

This folder contains the code used for the MountainCar reinforcement learning
project. The main training programs are based on Chapter 18 from
Deep Reinforcement Learning Hands-On, Third Edition.

## 1. Create and activate Python environment

Use Python 3.11 when possible.

```powershell
cd E:\Deep-Reinforcement-Learning-Hands-On-Third-Edition
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv311\Scripts\Activate.ps1
```

## 2. Run the RiverSwim exploration example

```powershell
python Chapter18\riverswim.py -n 1000
```

This script is text-only. It prints how often each RiverSwim state was visited.

## 3. Train MountainCar with DQN

```powershell
python Chapter18\mcar_dqn.py -n dqn_run -p egreedy
```

Other DQN variants:

```powershell
python Chapter18\mcar_dqn.py -n dqn_long -p egreedy-long
python Chapter18\mcar_dqn.py -n dqn_noisy -p noisynet
python Chapter18\mcar_dqn.py -n dqn_counts -p counts
python Chapter18\mcar_dqn.py -n dqn_shaped -p shaped
```

The best checkpoint is saved automatically under `saves/`, for example:

```text
saves/mcar_dqn_egreedy_dqn_run_best.dat
```

## 4. Train MountainCar with PPO

```powershell
python Chapter18\mcar_ppo.py -n ppo_run -p ppo
```

Other PPO variants:

```powershell
python Chapter18\mcar_ppo.py -n ppo_noisy -p noisynet
python Chapter18\mcar_ppo.py -n ppo_counts -p counts
python Chapter18\mcar_ppo.py -n ppo_shaped -p shaped
python Chapter18\mcar_ppo.py -n ppo_distill -p distill
```

The best checkpoint is saved automatically under `saves/`, for example:

```text
saves/mcar_ppo_ppo_ppo_run_best.dat
```

## 5. Watch training with TensorBoard

```powershell
tensorboard --logdir runs
```

Open this URL in the browser:

```text
http://localhost:6006
```

Important metrics:

- `train/avg_test_reward`: the main score. Higher is better. For MountainCar,
  moving from about `-200` toward `-130` means the agent is learning.
- `train/test_reward`: reward from the latest evaluation episode.
- `train/test_steps`: how many steps the evaluation episode lasted.
- `train/avg_loss`: average training loss.

For `counts` and `shaped` runs, episode rewards are measured on the modified
training environment. Use `train/avg_test_reward` and `train/test_steps` for the
real MountainCar score because evaluation still uses the original environment.
If `test_steps` stays at `200`, the car has not reached the flag yet.

## 6. Play a trained model

Run PPO checkpoint with a visible game window:

```powershell
python Chapter18\mcar_play.py --model saves\mcar_ppo_ppo_ppo_run_best.dat --render
```

Run DQN checkpoint with a visible game window:

```powershell
python Chapter18\mcar_play.py --model saves\mcar_dqn_egreedy_dqn_run_best.dat --render
```

The script reads the algorithm and model type from the checkpoint metadata.

## 7. Record demo video

Record PPO demo video:

```powershell
python Chapter18\mcar_play.py --model saves\mcar_ppo_ppo_ppo_run_best.dat --episodes 3 --record videos\ppo_demo
```

Record DQN demo video:

```powershell
python Chapter18\mcar_play.py --model saves\mcar_dqn_egreedy_dqn_run_best.dat --episodes 3 --record videos\dqn_demo
```

The generated `.mp4` files are saved inside the selected `videos/` subfolder.

## 8. Suggested experiment set

For the project report, train at least these two baseline runs:

```powershell
python Chapter18\mcar_dqn.py -n dqn_run -p egreedy
python Chapter18\mcar_ppo.py -n ppo_run -p ppo
```

Then compare them in TensorBoard using `train/avg_test_reward`.

For a practical demo, train one exploration-aided run and one shaped-reward run:

```powershell
python Chapter18\mcar_ppo.py -n ppo_counts -p counts
python Chapter18\mcar_ppo.py -n ppo_shaped -p shaped
```

Use the checkpoint whose `train/avg_test_reward` is best and whose
`train/test_steps` drops below `200`.
