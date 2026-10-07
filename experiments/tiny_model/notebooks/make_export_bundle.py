"""
HISTORICAL (reproduction script): it verifies the exact backend files of the Phase 4-6 experiment.
The runtime has since changed (Query Router and Groq removed, execution moved to
backend/analytics_executor.py), so it refuses to build from the current working tree. The bundles it
made are in artifacts/, and they are the only copy of those backend files (that state was never
committed). Its frozen hashes are intentionally NOT updated.

Build the single zip the Phase 6 export notebook needs: the Phase 4 plan format, the Phase 2 scorer,
the Phase 5 harness (its prompt/generation/LoRA-check/scoring functions; not the frozen set), the
read-only backend + CSV for validate_plan()/execute_plan(), and the Phase 6 smoke/benchmark set and
llama-server client.

    python experiments/tiny_model/notebooks/make_export_bundle.py
    -> experiments/tiny_model/artifacts/marketingiq_export_bundle.zip   (git-ignored)

Refuses to build unless every shared file is byte-identical to what Phase 4 trained/validated with
and Phase 5 evaluated with. Never packs eval_frozen.jsonl, training data, .env files or secrets.
"""
import json
import sys
import zipfile

from make_colab_bundle import REPO, TINY, sha256
from make_eval_bundle import EXPECTED as PHASE5_EXPECTED

OUT = TINY / "artifacts" / "marketingiq_export_bundle.zip"

EXPECTED = {rel: h for rel, h in PHASE5_EXPECTED.items() if "eval_frozen" not in rel}
EXPECTED["experiments/tiny_model/evaluation/frozen_eval.py"] = \
    "d5ef070a98a020c5545a0d02264469b4c944f785941deacfecb9c37587d78ecd"   # Phase 5 eval bundle manifest
FILES = [*EXPECTED,
         "experiments/tiny_model/benchmark/llama_client.py",
         "experiments/tiny_model/benchmark/benchmark_questions.jsonl"]
FORBIDDEN = ("eval_frozen", "eval_manifest", "train.jsonl", "validation.jsonl", ".env")


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
            sys.exit(f"{rel} differs from its Phase 4/5 SHA-256; refusing to build the export bundle")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixed_time = (2026, 9, 29, 0, 0, 0)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in FILES:
            z.writestr(zipfile.ZipInfo(rel, fixed_time), (REPO / rel).read_bytes(), zipfile.ZIP_DEFLATED)
        z.writestr(zipfile.ZipInfo("bundle_manifest.json", fixed_time), json.dumps(manifest, indent=2))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(FILES)} files)")
    print(f"zip SHA-256: {sha256(OUT)}")
    print("Upload it to Google Drive as MyDrive/marketingiq_tiny/marketingiq_export_bundle.zip")


if __name__ == "__main__":
    main()
