"""
HISTORICAL (reproduction script): it verifies the exact backend files of the Phase 4-6 experiment.
The runtime has since changed (Query Router and Groq removed, execution moved to
backend/analytics_executor.py), so it refuses to build from the current working tree. The bundles it
made are in artifacts/, and they are the only copy of those backend files (that state was never
committed). Its frozen hashes are intentionally NOT updated.

Build the single zip the Phase 4 Colab notebook needs: the train/validation data, plan_format.py,
score.py, and the read-only backend modules + campaign CSV that validate_plan()/execute_plan() use.
Paths inside the zip mirror the repo, so plan_format.py finds backend/ and data/ unchanged.

    python experiments/tiny_model/notebooks/make_colab_bundle.py
    -> experiments/tiny_model/artifacts/marketingiq_colab_bundle.zip   (git-ignored)

The frozen evaluation set (eval_frozen.jsonl) is never included, and neither are .env files or
any other secrets: only the files listed below are packed. Nothing is modified, nothing is uploaded.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TINY = HERE.parent
REPO = TINY.parents[1]
OUT = TINY / "artifacts" / "marketingiq_colab_bundle.zip"

EXPECTED = {  # Phase 3 frozen hashes (dataset/train_manifest.json)
    "experiments/tiny_model/dataset/train.jsonl":
        "55452c8e687b9fb78c4ca33b466534cbf63e4039a0be40ba8814c628c9aca007",
    "experiments/tiny_model/dataset/validation.jsonl":
        "e247278e7ca147e9e2ea8786784d8409a930c2b2b357afa8d54ddbe672c35e73",
}
FILES = [
    *EXPECTED,
    "experiments/tiny_model/dataset/plan_format.py",
    "experiments/tiny_model/evaluation/score.py",
    # the modules score.py -> plan_format.backend() imports (checked: nothing else is loaded)
    *(f"backend/{m}.py" for m in ("query_planner", "data_tools", "campaign_repository", "chart_builder",
                                  "grounding", "query_router", "schema_catalog", "analytics_engine")),
    "data/tech_advertising_campaigns_dataset.csv",
]
FORBIDDEN = ("eval_frozen", "eval_manifest", ".env")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = {}
    for rel in FILES:
        if any(f in rel for f in FORBIDDEN):
            sys.exit(f"refusing to bundle {rel}")
        path = REPO / rel
        if not path.is_file():
            sys.exit(f"missing {rel}")
        manifest[rel] = sha256(path)
        if rel in EXPECTED and manifest[rel] != EXPECTED[rel]:
            sys.exit(f"{rel} does not match its frozen SHA-256; the Phase 3 data has changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixed_time = (2026, 9, 29, 0, 0, 0)  # stable bytes for identical inputs
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in FILES:
            z.writestr(zipfile.ZipInfo(rel, fixed_time), (REPO / rel).read_bytes(), zipfile.ZIP_DEFLATED)
        z.writestr(zipfile.ZipInfo("bundle_manifest.json", fixed_time), json.dumps(manifest, indent=2))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(FILES)} files)")
    print(f"zip SHA-256: {sha256(OUT)}")
    print("Upload it to Google Drive as MyDrive/marketingiq_tiny/marketingiq_colab_bundle.zip")


if __name__ == "__main__":
    main()
