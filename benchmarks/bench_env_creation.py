"""Benchmark environment creation (gym.make + close)."""

from __future__ import annotations

import myosuite  # noqa: F401 — registers envs
from myosuite.utils import gym


def _create_and_close(env_id: str) -> None:
    env = gym.make(env_id)
    env.close()


def test_env_creation(benchmark, env_id: str) -> None:
    benchmark.pedantic(_create_and_close, args=(env_id,), rounds=5, iterations=1)
