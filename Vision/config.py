from dataclasses import dataclass, field
from typing import Dict, List

# ============================================================
# Camera Configuration
# ============================================================

@dataclass
class CameraConfig:
    device_index: int = 0          # USB camera index
    width: int = 640              # target width
    height: int = 480             # target height
    fps: int = 30                  # target fps
    fourcc: str = "MJPG"           # MJPG recommended for USB2.0 cameras
    backend: str = "v4l2"          # "auto", "dshow", "v4l2"


@dataclass
class ModelConfig:
    # model_path: str = "/home/turkeysquad/ME450/MainDesign/Vision/weights/turkey_1280.pt"
    model_path: str = "/home/turkeysquad/ME450/MainDesign/Vision/weights/turkey_640.hef"
    # model_path: str = "/home/turkeysquad/ME450/MainDesign/Vision/weights/turkey_640.pt"
    conf_thresh: float = 0.3
    iou_thresh: float = 0.50
    imgsz: int = 640
    # max_det: int = 10
    max_det: int = 5
    device: str = "cpu" # "hailo" #"cpu" #"hailo"  #"cuda:0" "cuda:0"   # "cpu" / "cuda:0"
    # half: bool = True
    half: bool = False

    classes_of_interest: List[str] = field(default_factory=lambda: [
        "turkey",
        "person",
        "vehicle",
    ])

@dataclass
class PositionConfig:
    hfov_deg = 96.0
    human_model_path = "/home/turkeysquad/ME450/MainDesign/Vision/weights/distance_human.pkl"
    turkey_model_path = "/home/turkeysquad/ME450/MainDesign/Vision/weights/distance_turkey.pkl"

# ============================================================
# Detection Filtering Configuration
# ============================================================

@dataclass
class FilterConfig:
    min_bbox_area: int = 100
    max_bbox_area: int = 1920 * 1080

    # Optional aspect ratio filtering
    enable_aspect_ratio_filter: bool = False
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0

# ============================================================
# Temporal Verification Configuration
# Example:
# In the last 5 frames, at least 3 frames must contain a valid object
# ============================================================

@dataclass
class TemporalConfig:
    init_window: int = 10
    init_min_hits: int = 5
    delete_missing_frames: int = 30

    # If True, detections are checked separately by class
    verify_per_class: bool = True


# ============================================================
# Output / Debug Configuration
# ============================================================

@dataclass
class DebugConfig:
    show_console_log: bool = True
    show_fps: bool = True
    save_log_to_file: bool = False
    log_file_path: str = "logs/runtime_log.txt"

    show_preview: bool = False
    preview_window_name: str = "Turkey CV Pipeline"


# ============================================================
# Class / Behavior Configuration
# This makes it easier to later assign different priorities
# ============================================================

@dataclass
class ClassConfig:
    class_name_map: Dict[int, str] = field(default_factory=lambda: {
        0: "turkey",
        1: "person",
        2: "vehicle",
    })

    # Higher number = higher priority
    class_priority: Dict[str, int] = field(default_factory=lambda: {
        "person": 3,
        "vehicle": 2,
        "turkey": 1,
    })


# ============================================================
# Whole App Configuration
# ============================================================

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    class_cfg: ClassConfig = field(default_factory=ClassConfig)


# Global config object
CONFIG = AppConfig()
