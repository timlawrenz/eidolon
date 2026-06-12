import argparse
import sys
import json
from pathlib import Path

def cmd_discover(args):
    from .identity import discover_identities, build_manifest, save_manifest
    root = Path(args.source)
    try:
        identities = discover_identities(root, min_sets=args.min_sets)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    manifest = build_manifest(root, identities, max_identities=args.max_identities)
    output = Path(args.dataset)
    save_manifest(manifest, output)
    print(f"Discovered {len(identities)} identities and saved manifest.")
    return 0

def cmd_extract_faces(args):
    from .face_extraction import extract_all
    dataset = Path(args.dataset)
    manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest not found in {dataset}")
        return 1
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    extract_all(manifest, dataset)
    return 0

def cmd_review_seed(args):
    from .review.seed import seed_from_extraction
    dataset = Path(args.dataset)
    db_path = dataset / "review.db"
    faces_dir = dataset / "faces"
    manifest_path = dataset / "manifest.json"
    try:
        inserted = seed_from_extraction(db_path, faces_dir, manifest_path)
        print(f"Inserted {inserted} face crops into review DB.")
        return 0
    except Exception as e:
        print(f"Error seeding DB: {e}")
        return 1

def cmd_review_ui(args):
    from .review.ui import create_app
    dataset = Path(args.dataset)
    db_path = dataset / "review.db"
    faces_dir = dataset / "faces"
    app = create_app(db_path, faces_dir)
    print(f"Review UI running at http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0

def cmd_enrich(args):
    from .enrichment import generate_approved_list, run_stratum_enrichment
    dataset = Path(args.dataset)
    print("Generating list of approved images for Stratum...")
    list_path = generate_approved_list(dataset)
    print(f"Approved list written to: {list_path}")
    print("Running Stratum enrichment...")
    run_stratum_enrichment(list_path, dataset)
    print("Stratum enrichment complete.")
    return 0

def main():
    parser = argparse.ArgumentParser(prog="hegre-dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    p_disc = sub.add_parser("discover")
    p_disc.add_argument("--source", required=True)
    p_disc.add_argument("--dataset", required=True)
    p_disc.add_argument("--min-sets", type=int, default=3)
    p_disc.add_argument("--max-identities", type=int)
    p_disc.set_defaults(func=cmd_discover)

    p_ext = sub.add_parser("extract-faces")
    p_ext.add_argument("--dataset", required=True)
    p_ext.set_defaults(func=cmd_extract_faces)

    p_rev = sub.add_parser("review")
    rsub = p_rev.add_subparsers(dest="review_command", required=True)
    
    p_seed = rsub.add_parser("seed")
    p_seed.add_argument("--dataset", required=True)
    p_seed.set_defaults(func=cmd_review_seed)
    
    p_ui = rsub.add_parser("ui")
    p_ui.add_argument("--dataset", required=True)
    p_ui.add_argument("--port", type=int, default=5101)
    p_ui.set_defaults(func=cmd_review_ui)

    p_enrich = sub.add_parser("enrich")
    p_enrich.add_argument("--dataset", required=True)
    p_enrich.set_defaults(func=cmd_enrich)

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
