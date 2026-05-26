#!/usr/bin/env python3
import numpy as np
import ptan
import ptan.ignite as ptan_ignite
import gymnasium as gym
import argparse
import random
import torch
import torch.optim as optim
from pathlib import Path
from types import SimpleNamespace

from ignite.engine import Engine

from lib import common, dqn_extra


HYPERPARAMS = {
    'egreedy': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'egreedy',
        'replay_size':      100000,
        'replay_initial':   100,
        'target_net_sync':  100,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': False,
    }),
    'egreedy-long': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'egreedy-long',
        'replay_size':      100000,
        'replay_initial':   1000,
        'target_net_sync':  100,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': True,
    }),
    'noisynet': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'noisynet',
        'replay_size':      100000,
        'replay_initial':   1000,
        'target_net_sync':  1000,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': False,
    }),
    'counts': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'counts',
        'replay_size':      100000,
        'replay_initial':   1000,
        'target_net_sync':  1000,
        'learning_rate':    0.00005,
        'gamma':            0.99,
        'batch_size':       32,
        'counts_reward_scale': 0.5,
        'eps_decay_trigger': False,
    }),
    'shaped': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'shaped',
        'replay_size':      100000,
        'replay_initial':   1000,
        'target_net_sync':  500,
        'epsilon_frames':   50000,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': False,
        'shaped_position_scale': 1.0,
        'shaped_velocity_scale': 100.0,
        'shaped_goal_bonus': 100.0,
    }),
}

N_STEPS = 4


def counts_hash(obs: np.ndarray):
    r = obs.tolist()
    return tuple(map(lambda v: round(v, 3), r))


def make_checkpoint_path(save_dir: Path, params_name: str, run_name: str) -> Path:
    safe_run_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
    return save_dir / f"mcar_dqn_{params_name}_{safe_run_name}_best.dat"


def wrap_training_env(env: gym.Env, params_name: str, params: SimpleNamespace) -> gym.Env:
    if params_name == 'counts':
        return common.PseudoCountRewardWrapper(
            env, reward_scale=params.counts_reward_scale, hash_function=counts_hash)
    if params_name == 'shaped':
        return common.MountainCarShapedRewardWrapper(
            env,
            position_scale=params.shaped_position_scale,
            velocity_scale=params.shaped_velocity_scale,
            goal_bonus=params.shaped_goal_bonus,
        )
    return env


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-p", "--params", default='egreedy', choices=list(HYPERPARAMS.keys()),
                        help="Parameters, default=egreedy")
    parser.add_argument("--save-dir", default="saves", help="Directory to store best model checkpoints")
    args = parser.parse_args()

    params = HYPERPARAMS[args.params]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = make_checkpoint_path(save_dir, args.params, args.name)
    best_test_reward = float("-inf")

    env = wrap_training_env(gym.make(params.env_name), args.params, params)
    test_env = gym.make(params.env_name)
    if args.params.startswith("egreedy") or args.params in ('counts', 'shaped'):
        net = dqn_extra.MountainCarBaseDQN(env.observation_space.shape[0], env.action_space.n)
    elif args.params == 'noisynet':
        net = dqn_extra.MountainCarNoisyNetDQN(env.observation_space.shape[0], env.action_space.n)
    tgt_net = ptan.agent.TargetNet(net)
    print(net)

    if args.params.startswith('egreedy'):
        selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
        epsilon_tracker = common.EpsilonTracker(selector, params)
        training_enabled = not params.eps_decay_trigger
        epsilon_tracker_frame = 0
    else:
        selector = ptan.actions.ArgmaxActionSelector()
        training_enabled = True

    agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=N_STEPS,
        env_seed=common.SEED
    )
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        if not training_enabled:
            return {
                "loss": 0.0,
                "epsilon": selector.epsilon
            }

        optimizer.zero_grad()
        loss_v = common.calc_loss_double_dqn(batch, net, tgt_net.target_model,
                                             gamma=params.gamma**N_STEPS)
        loss_v.backward()
        optimizer.step()
        res = {
            "loss": loss_v.item(),
            "epsilon": 0.0,
        }
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()

        if args.params.startswith("egreedy") or args.params == 'shaped':
            epsilon_tracker.frame(engine.state.iteration - epsilon_tracker_frame)
            res['epsilon'] = selector.epsilon
        # reset noise every training step, this is fine in off-policy method
        if args.params == 'noisynet':
            net.reset_noise()
        return res

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, args.name, extra_metrics=(
        'test_reward', 'avg_test_reward', 'test_steps'))

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def check_reward_trigger(trainer: Engine):
        global training_enabled, epsilon_tracker_frame
        if training_enabled:
            return
        # check trigger condition to enable epsilon decay
        if trainer.state.episode_reward > -200:
            training_enabled = True
            epsilon_tracker_frame = trainer.state.iteration
            print("Epsilon decay triggered!")

    @engine.on(ptan_ignite.PeriodEvents.ITERS_1000_COMPLETED)
    def test_network(engine):
        net.train(False)
        obs, _ = test_env.reset()
        reward = 0.0
        steps = 0

        while True:
            acts, _ = agent([obs])
            obs, r, is_done, is_tr, _ = test_env.step(acts[0])
            reward += r
            steps += 1
            if is_done or is_tr:
                break
        test_reward_avg = getattr(engine.state, "test_reward_avg", None)
        if test_reward_avg is None:
            test_reward_avg = reward
        else:
            test_reward_avg = test_reward_avg * 0.95 + 0.05 * reward
        engine.state.test_reward_avg = test_reward_avg
        print("Test done: got %.3f reward after %d steps, avg reward %.3f" % (
            reward, steps, test_reward_avg
        ))
        engine.state.metrics['test_reward'] = reward
        engine.state.metrics['avg_test_reward'] = test_reward_avg
        engine.state.metrics['test_steps'] = steps
        global best_test_reward
        if test_reward_avg > best_test_reward:
            best_test_reward = test_reward_avg
            torch.save({
                "algo": "dqn",
                "params": args.params,
                "env_name": params.env_name,
                "obs_size": env.observation_space.shape[0],
                "n_actions": env.action_space.n,
                "state_dict": net.state_dict(),
                "best_avg_test_reward": best_test_reward,
                "test_reward": reward,
                "test_steps": steps,
                "iteration": engine.state.iteration,
            }, save_path)
            print("Saved best DQN checkpoint to %s, avg reward %.3f" % (
                save_path, best_test_reward
            ))

        if test_reward_avg > params.stop_test_reward:
            print("Reward boundary has crossed, stopping training. Contgrats!")
            engine.should_terminate = True
        net.train(True)

    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
