from typing import Any, List, Dict, Union

from pydantic import BaseModel
from easydiffusion.trajectory_types import TrajectoryData


class GenerateImageRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""

    seed: int = 42
    width: int = 512
    height: int = 512

    num_outputs: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    distilled_guidance_scale: float = 3.5

    init_image: Any = None
    init_image_mask: Any = None
    ref_images: Any = None
    control_image: Any = None
    control_alpha: Union[float, List[float]] = None
    controlnet_filter: str = None
    prompt_strength: float = 0.8
    preserve_init_image_color_profile: bool = False
    strict_mask_border: bool = False

    sampler_name: str = None
    scheduler_name: str = None
    hypernetwork_strength: float = 0
    lora_alpha: Union[float, List[float]] = 0
    tiling: str = None


class FilterImageRequest(BaseModel):
    image: Any = None
    filter: Union[str, List[str]] = None
    filter_params: dict = {}


class ModelsData(BaseModel):
    model_paths: Dict[str, Union[str, None, List[str]]] = None
    model_params: Dict[str, Dict[str, Any]] = {}


class OutputFormatData(BaseModel):
    output_format: str = "jpeg"
    output_quality: int = 75
    output_lossless: bool = False


class SaveToDiskData(BaseModel):
    save_to_disk_path: str = None
    metadata_output_format: str = "txt"


class TaskData(BaseModel):
    request_id: str = None
    session_id: str = "session"


class RenderTaskData(TaskData):
    vram_usage_level: str = "balanced"

    use_face_correction: Union[str, List[str]] = None
    use_upscale: Union[str, List[str]] = None
    upscale_amount: int = 4
    latent_upscaler_steps: int = 10
    use_stable_diffusion_model: Union[str, List[str]] = "sd-v1-4"
    use_vae_model: Union[str, List[str]] = None
    use_text_encoder_model: Union[str, List[str]] = None
    use_hypernetwork_model: Union[str, List[str]] = None
    use_lora_model: Union[str, List[str]] = None
    use_controlnet_model: Union[str, List[str]] = None
    use_embeddings_model: Union[str, List[str]] = None
    filters: List[str] = []
    filter_params: Dict[str, Dict[str, Any]] = {}
    control_filter_to_apply: Union[str, List[str]] = None
    enable_vae_tiling: bool = True

    show_only_filtered_image: bool = False
    block_nsfw: bool = False
    stream_image_progress: bool = False
    stream_image_progress_interval: int = 5
    clip_skip: bool = False
    codeformer_upscale_faces: bool = False
    codeformer_fidelity: float = 0.5
    trajectory: TrajectoryData = TrajectoryData()


class MergeRequest(BaseModel):
    model0: str = None
    model1: str = None
    ratio: float = None
    out_path: str = "mix"
    use_fp16: bool = True
