import argparse
import json
import os
import sys


def load_manifest(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: str, manifest: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Rollback active model version using manifest.json.")
    parser.add_argument("--version", required=True, help="Target model version key in manifest models map.")
    parser.add_argument(
        "--manifest",
        default=os.getenv("MODEL_MANIFEST_PATH", "models/manifest.json"),
        help="Path to model manifest file.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    models = manifest.get("models", {})
    if args.version not in models:
        known = ", ".join(sorted(models.keys())) or "none"
        raise ValueError(f"Unknown version '{args.version}'. Available versions: {known}")

    selected = models[args.version]
    bundle_path = selected.get("bundle_path")
    columns_path = selected.get("columns_path")
    if not bundle_path or not columns_path:
        raise ValueError(f"Version '{args.version}' is missing bundle_path/columns_path.")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"bundle_path does not exist: {bundle_path}")
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"columns_path does not exist: {columns_path}")

    previous = manifest.get("active_version")
    manifest["active_version"] = args.version
    save_manifest(args.manifest, manifest)
    print(f"Rollback complete: active_version {previous} -> {args.version}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
