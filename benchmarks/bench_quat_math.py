"""Micro-benchmarks for quaternion math utilities."""

from __future__ import annotations

import numpy as np

from myosuite.utils.quat_math import euler2quat, mulQuat, quat2mat


def test_euler2quat(benchmark) -> None:
    euler = np.array([0.1, 0.2, 0.3])
    benchmark(euler2quat, euler)


def test_quat2mat(benchmark) -> None:
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    benchmark(quat2mat, quat)


def test_mulQuat(benchmark) -> None:
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.0, 1.0, 0.0, 0.0])
    benchmark(mulQuat, q1, q2)
