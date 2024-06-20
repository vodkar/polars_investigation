import tempfile
from pathlib import Path

DATASETS_PATH = Path(__file__).parent.parent / "datasets"
DATASET_SIZES = ["200MB", "400MB", "800MB", "1600MB", "3200MB", "6400MB"]
TRAIN_PARQUET_NAME = "train.parquet"
USERS_SESSION_PARQUET = DATASETS_PATH / "users_session.parquet"
USERS_PARQUET = "users.parquet"
MEMRAY_TRACK_FILE = Path(tempfile.gettempdir()) / "memray_track.bin"
