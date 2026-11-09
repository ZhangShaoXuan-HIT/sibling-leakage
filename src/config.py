"""Path configuration. Data are not distributed; see data/README.md."""
import os
from pathlib import Path

NEUSEG_ROOT = Path(os.environ.get("NEUSEG_ROOT", "data/NEU_Seg-main"))
STM_ROOT = Path(os.environ.get("STM_ROOT", "data/STM/Segmentation_data"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "out"))
OUT_ROOT.mkdir(parents=True, exist_ok=True)
