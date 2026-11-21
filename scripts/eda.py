"""Generate docs/EDA.md (T1.1). Recommendations are kept in docs/eda_recommendations.md and appended."""
from polero import REPO_ROOT
from polero.eda import main

rec = REPO_ROOT / "docs" / "eda_recommendations.md"
main(recommendations=rec.read_text() if rec.exists() else "")
print("wrote docs/EDA.md")
