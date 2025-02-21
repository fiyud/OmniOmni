# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.50"

import os

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

from ultralytics_1.data.explorer.explorer import Explorer
from ultralytics_1.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from ultralytics_1.utils import ASSETS, SETTINGS
from ultralytics_1.utils.checks import check_yolo as checks
from ultralytics_1.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
