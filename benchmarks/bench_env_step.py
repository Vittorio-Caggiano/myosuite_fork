"""Benchmark environment step throughput."""

from __future__ import annotations

import myosuite  # noqa: F401 — registers envs
from myosuite.utils import gym


def _step_100(env_id: str) -> None:
    env = gym.make(env_id)
    env.reset()
    action = env.action_space.sample()
    for _ in range(100):
        env.step(action)
    env.close()


def test_step_throughput(benchmark, env_id: str) -> None:
    benchmark.pedantic(_step_100, args=(env_id,), rounds=5, iterations=1)
