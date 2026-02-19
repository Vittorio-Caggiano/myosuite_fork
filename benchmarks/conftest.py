from __future__ import annotations

import pytest

ENV_IDS = [
    "myoFingerPoseRandom-v0",
    "myoElbowPose1D6MRandom-v0",
    "myoHandPoseRandom-v0",
    "myoLegWalk-v0",
]


@pytest.fixture(params=ENV_IDS)
def env_id(request: pytest.FixtureRequest) -> str:
    return request.param
