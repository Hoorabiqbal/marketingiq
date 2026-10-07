"""
HISTORICAL (reproduction script): it verifies the exact backend files of the Phase 4-6 experiment.
The runtime has since changed (Query Router and Groq removed, execution moved to
backend/analytics_executor.py), so it refuses to build from the current working tree. The bundles it
made are in artifacts/, and they are the only copy of those backend files (that state was never
committed). Its frozen hashes are intentionally NOT updated.

Build the single zip the Phase 5 Colab notebook needs: the frozen evaluation set, the Phase 5
harness (evaluation/frozen_eval.py) and the exact plan format, scorer, backend modules and CSV
Phase 4 trained and validated against. Paths mirror the repo, like the Phase 4 bundle.

    python experiments/tiny_model/notebooks/make_eval_bundle.py
    -> experiments/tiny_model/artifacts/marketingiq_eval_bundle.zip   (git-ignored)

Refuses to build unless eval_frozen.jsonl is the frozen eval-v1 file and every shared file is
byte-identical to the Phase 4 bundle (a changed validator or executor would change what is being
measured). No training data, .env files or secrets are packed. Nothing is uploaded.
"""
import json
import sys
import zipfile
from pathlib import Path

from make_colab_bundle import REPO, TINY, sha256

OUT = TINY / "artifacts" / "marketingiq_eval_bundle.zip"

EXPECTED = {
    # eval-v1 (dataset/eval_manifest.json)
    "experiments/tiny_model/dataset/eval_frozen.jsonl":
        "51966831e505de8ac59fec49dad53cb1c2d4daa8744fa46d75794be831c7f2b1",
    # Phase 4 bundle (marketingiq_colab_bundle.zip bundle_manifest.json): what the model was validated with
    "experiments/tiny_model/dataset/plan_format.py": "fc4480906324b1769310cc9e53e85f33f153eb49601fd8b50296f2c4c067c120",
    "experiments/tiny_model/evaluation/score.py": "96c9ff545ef2e6987e776858e7404344386b31b2f55e3a249c957a0179abf339",
    "backend/query_planner.py": "b62cc0962b5e16b36b6c99fe33e2ebaa77938d881851d4fce3b6b33017d5fd94",
    "backend/data_tools.py": "52d6491464c69a6cd8bff59e2bade153c2794b46d57a2ac56b9db620aba61906",
    "backend/campaign_repository.py": "f7feaf18372349be58099826c3bac9d5294cd7eea016728fae79100c60946e41",
    "backend/chart_builder.py": "b46b5df25a2a8ca7d6e48458c0a2929419c3af6d7dbc36be5cb82a4f15af530e",
    "backend/grounding.py": "fa080489801c98567a8ddd6d6021fcfcf96e6d502a5eecb3dff1167224db70ba",
    "backend/query_router.py": "d4cee13fdbabd0e8900573c9f093137a21233964431d5ad3056dbd344cc1642e",
    "backend/schema_catalog.py": "7c14417cad6c4af672fba0a3c3fc3d6892b2f41b336f388d655487233597b7f6",
    "backend/analytics_engine.py": "64f7b312d1301ea996f8f8d1c75c268d0d8e3b4aa97330dbadfd3eb983d169f8",
    "data/tech_advertising_campaigns_dataset.csv": "ff250cba6e4176d3adc6950599998ca29143d7dadb3b424844e1c4b17e6d9417",
}
FILES = [*EXPECTED, "experiments/tiny_model/dataset/eval_manifest.json",
         "experiments/tiny_model/evaluation/frozen_eval.py"]
FORBIDDEN = ("train.jsonl", "validation.jsonl", ".env")


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
            sys.exit(f"{rel} differs from its frozen / Phase 4 SHA-256; refusing to build the evaluation bundle")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixed_time = (2026, 9, 29, 0, 0, 0)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in FILES:
            z.writestr(zipfile.ZipInfo(rel, fixed_time), (REPO / rel).read_bytes(), zipfile.ZIP_DEFLATED)
        z.writestr(zipfile.ZipInfo("bundle_manifest.json", fixed_time), json.dumps(manifest, indent=2))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(FILES)} files)")
    print(f"zip SHA-256: {sha256(OUT)}")
    print("Upload it to Google Drive as MyDrive/marketingiq_tiny/marketingiq_eval_bundle.zip")


if __name__ == "__main__":
    main()
