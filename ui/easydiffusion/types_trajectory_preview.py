# Temporary compile-check copy. This file will be removed after wiring TrajectoryData into types.py.
from typing import Any, List, Dict, Union

from pydantic import BaseModel
from easydiffusion.trajectory_types import TrajectoryData


class RenderTaskTrajectoryProbe(BaseModel):
    trajectory: TrajectoryData = TrajectoryData()
