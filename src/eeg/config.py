from pathlib import Path


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data" / "physionet.org" / "files" / "eegmmidb" / "1.0.0"
MODEL_DIR = PROJECT_ROOT / "models"

EVENT_LABELS = {
    "T1": "imagined left fist",
    "T2": "imagined right fist",
}
DEFAULT_CHANNELS = ("C3", "Cz", "C4")

SELECTED_RUNS = ("R04", "R08", "R12")
PREPROCESSING_ARGS = {
    "channels": list(DEFAULT_CHANNELS),
    "l_freq": 8.0,
    "h_freq": 30.0,
    "notch_freq": 60.0,
    "tmin": 0.5,
    "tmax": 3.5,
    "resample_sfreq": None,
}
