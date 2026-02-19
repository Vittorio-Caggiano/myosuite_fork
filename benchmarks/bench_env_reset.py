"""Benchmark environment reset latency."""

from __future__ import annotations

import myosuite  # noqa: F401 — registers envs
from myosuite.utils import gym


def _reset(env_id: str) -> None:
    env = gym.make(env_id)
    env.reset()
    env.close()


def test_reset_latency(benchmark, env_id: str) -> None:
    benchmark.pedantic(_reset, args=(env_id,), rounds=20, iterations=1)
