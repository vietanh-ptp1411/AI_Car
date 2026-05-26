#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from lib import dqn_extra, ppo


DEFAULT_ENV_NAME = "MountainCar-v0"


def build_dqn(params_name: str, obs_size: int, n_actions: int) -> torch.nn.Module:
    if params_name == "noisynet":
        return dqn_extra.MountainCarNoisyNetDQN(obs_size, n_actions)
    return dqn_extra.MountainCarBaseDQN(obs_size, n_actions)


def build_ppo(params_name: str, obs_size: int, n_actions: int) -> torch.nn.Module:
    if params_name == "noisynet":
        return ppo.MountainCarNoisyNetsPPO(obs_size, n_actions)
    return ppo.MountainCarBasePPO(obs_size, n_actions)


def load_checkpoint(model_path: Path, algo_arg: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(
            "Unsupported checkpoint format. Train again with mcar_dqn.py or mcar_ppo.py "
            "after this update so the checkpoint includes metadata."
        )

    algo = checkpoint.get("algo", algo_arg)
    if algo_arg != "auto" and algo != algo_arg:
        raise ValueError(f"Checkpoint algo is {algo!r}, but --algo is {algo_arg!r}")

    params_name = checkpoint.get("params", "egreedy" if algo == "dqn" else "ppo")
    obs_size = checkpoint.get("obs_size", 2)
    n_actions = checkpoint.get("n_actions", 3)

    if algo == "dqn":
        net = build_dqn(params_name, obs_size, n_actions)
    elif algo == "ppo":
        net = build_ppo(params_name, obs_size, n_actions)
    else:
        raise ValueError(f"Unsupported algorithm in checkpoint: {algo!r}")

    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    net.eval()
    return checkpoint, net


@torch.no_grad()
def select_action(algo: str, net: torch.nn.Module, obs: np.ndarray,
                  device: torch.device, sample: bool) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    if algo == "dqn":
        q_vals = net(obs_t)
        return int(torch.argmax(q_vals, dim=1).item())

    policy = net.actor(obs_t)
    if sample:
        probs = F.softmax(policy, dim=1)
        return int(torch.multinomial(probs, num_samples=1).item())
    return int(torch.argmax(policy, dim=1).item())


def make_env(env_name: str, render: bool, record_dir: str | None):
    if record_dir:
        env = gym.make(env_name, render_mode="rgb_array")
        return gym.wrappers.RecordVideo(
            env,
            video_folder=record_dir,
            episode_trigger=lambda episode_id: True,
            name_prefix="mountaincar",
        )
    if render:
        return gym.make(env_name, render_mode="human")
    return gym.make(env_name)


def main():
    parser = argparse.ArgumentParser(description="Play MountainCar with a trained DQN/PPO checkpoint")
    parser.add_argument("--model", required=True, help="Path to checkpoint saved by mcar_dqn.py or mcar_ppo.py")
    parser.add_argument("--algo", default="auto", choices=("auto", "dqn", "ppo"),
                        help="Algorithm to use. Default reads it from checkpoint metadata")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to run")
    parser.add_argument("--dev", default="cpu", help="Torch device, for example cpu or cuda")
    parser.add_argument("--render", action="store_true", help="Open an interactive game window")
    parser.add_argument("--record", help="Directory to store mp4 demo videos")
    parser.add_argument("--sample", action="store_true",
                        help="Sample PPO actions instead of taking the most likely action")
    parser.add_argument("--sleep", type=float, default=0.02,
                        help="Delay between rendered steps, default=0.02 seconds")
    args = parser.parse_args()

    device = torch.device(args.dev)
    model_path = Path(args.model)
    checkpoint, net = load_checkpoint(model_path, args.algo, device)
    algo = checkpoint["algo"]
    env_name = checkpoint.get("env_name", DEFAULT_ENV_NAME)

    env = make_env(env_name, args.render, args.record)
    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=episode)
            total_reward = 0.0
            steps = 0

            while True:
                if hasattr(net, "reset_noise"):
                    net.reset_noise()
                action = select_action(algo, net, obs, device, args.sample)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                if args.render and not args.record:
                    time.sleep(args.sleep)
                if terminated or truncated:
                    break

            print("Episode %d: reward=%.3f, steps=%d" % (episode, total_reward, steps))
    finally:
        env.close()

    if args.record:
        print("Video saved under %s" % Path(args.record).resolve())


if __name__ == "__main__":
    main()
