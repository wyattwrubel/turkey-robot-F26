"""
config/paths.py: every file and folder the robot code reads or writes (DR2 plan, contract C0).

All paths come from the repo's own location, so the same code runs on the robot
(~/ME450/MainDesign) and on any laptop clone. Model files are not here: they live
outside git, and Vision/config.py says where.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Maps and routes
ZONE_CSV      = REPO_ROOT / "Navigation" / "Zone.csv"
WAYPOINTS_CSV = REPO_ROOT / "Navigation" / "Waypoints.csv"

# Run logs (git ignores new ones; see .gitignore)
NAV_LOG_DIR     = REPO_ROOT / "Navigation" / "logs"
ENCODER_LOG_DIR = REPO_ROOT / "DriveTrain" / "Encoder"
IMU_LOG_DIR     = REPO_ROOT / "Sensors" / "AccelReadings"

# Assets and dependencies
AUDIO_DIR  = REPO_ROOT / "Canine" / "AudioFiles"
CASADI_DIR = REPO_ROOT / "casadi"   # W26's CasADi checkout; its compiled build exists only on the robot
