from pydantic import BaseModel


class TrajectoryData(BaseModel):
    """Opt-in trajectory capture settings for FlexDiffusion."""

    enabled: bool = False
    capture_schedule: str = ""
    persistence_mode: str = "preview"
    output_root: str = None
    project_name: str = None
    preview_format: str = "jpeg"
    preview_quality: int = 75
    max_checkpoints: int = None
    storage_budget_mb: float = None
    writer_queue_size: int = 2
