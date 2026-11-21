"""PolERo: political evasion & clarity classification (Nitro NLP hackathon)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = REPO_ROOT / "train_data.csv"
TEST_CSV = REPO_ROOT / "test_data.csv"
SAMPLE_CSV = REPO_ROOT / "sample_output.csv"
ARTIFACTS = REPO_ROOT / "artifacts"
SUBMISSIONS = REPO_ROOT / "submissions"
CONFIGS = REPO_ROOT / "configs"
SEED = 42
