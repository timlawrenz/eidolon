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
    import json
    from .face_extraction import extract_all
    dataset = Path(args.dataset)
    manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest not found in {dataset}")
        return 1
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    extract_all(manifest, dataset, device=args.device, max_workers=args.workers)
    return 0

def cmd_review_seed(args):
    from .review.seed import seed_from_extraction
    dataset = Path(args.dataset)
    db_path = dataset / "review.db"
    faces_dir = dataset / "faces"
    manifest_path = dataset / "manifest.json"
    try:
        inserted = seed_from_extraction(db_path, faces_dir, manifest_path, verbose=getattr(args, 'verbose', False))
        print(f"Inserted {inserted} face crops into review DB.")
        return 0
    except Exception as e:
        print(f"Error seeding DB: {e}")
        return 1

def cmd_review_ui(args):
    from .review.ui import create_app
    dataset = Path(args.dataset)
    db_path = dataset / "review.db"
    # image_path in DB is stored relative to the dataset root (e.g. 'faces/anna-l/...')
    # so we must pass dataset as the root, not dataset/"faces"
    app = create_app(db_path, dataset)
    print(f"Review UI running at http://{args.bind}:{args.port}")
    app.run(host=args.bind, port=args.port, debug=False)
    return 0

def cmd_enrich(args):
    from .enrichment import run_stratum_enrichment
    dataset = Path(args.dataset)
    db_path = dataset / "review.db"
    faces_dir = dataset  # image_path in DB is stored relative to dataset root (e.g. 'faces/anna-l/...')
    skip_stratum = getattr(args, "skip_stratum", False)
    status_filter = getattr(args, "status", "both")
    if skip_stratum:
        print(f"Running AuraFace-only enrichment (--skip-stratum) for {status_filter} images...")
    else:
        print(f"Running Stratum enrichment with passes: {args.passes} for {status_filter} images...")
    run_stratum_enrichment(dataset, db_path, faces_dir, passes=args.passes, skip_stratum=skip_stratum, status_filter=status_filter)
    print("Enrichment complete.")
    return 0

def cmd_review_compute_geometry(args):
    from .review.geometry import compute_zg_distances, compute_af_distances
    dataset = Path(args.dataset)
    db_path = dataset / "review.db"
    stratum_dir = dataset / "stratum"
    encoder_path = args.encoder
    persona = getattr(args, "persona", None)
    skip_3d = getattr(args, "skip_3d", False)
    metric = getattr(args, "metric", "both")

    rc = 0

    if metric in ("zg", "both"):
        if not stratum_dir.exists():
            print(f"Error: Stratum directory {stratum_dir} not found. Run enrichment first.")
            rc = 1
        else:
            zg_rc = compute_zg_distances(db_path, stratum_dir, encoder_path, persona, skip_3d, metric=metric)
            if zg_rc != 0:
                rc = zg_rc

    if metric in ("af", "both"):
        af_rc = compute_af_distances(db_path, dataset, persona)
        if af_rc != 0:
            rc = af_rc

    return rc

def main(args=None):
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
    p_ext.add_argument("--device", default="cuda:0", help="Device for MTCNN (default: cuda:0)")
    p_ext.add_argument("--workers", type=int, default=4, help="Thread pool workers")
    p_ext.set_defaults(func=cmd_extract_faces)

    p_rev = sub.add_parser("review")
    rsub = p_rev.add_subparsers(dest="review_command", required=True)
    
    p_seed = rsub.add_parser("seed")
    p_seed.add_argument("--dataset", required=True)
    p_seed.add_argument("-v", "--verbose", action="store_true", help="Print debug output for what is inserted/skipped")
    p_seed.set_defaults(func=cmd_review_seed)
    
    p_ui = rsub.add_parser("ui")
    p_ui.add_argument("--dataset", required=True)
    p_ui.add_argument("--bind", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=5101)
    p_ui.set_defaults(func=cmd_review_ui)
    
    p_geom = rsub.add_parser("compute-geometry", help="Compute zg_distances, af_distances, pixel averages, and 3D FLAME spins")
    p_geom.add_argument("--dataset", required=True)
    p_geom.add_argument("--encoder", required=True, help="Path to geometry_pca encoder_production.npz")
    p_geom.add_argument("--persona", type=str, help="Optional persona name (or ID) to limit computation")
    p_geom.add_argument("--skip-3d", action="store_true", help="Skip generating the 3D rotating FLAME mesh (PyRender can be slow)")
    p_geom.add_argument("--metric", choices=["zg", "af", "both"], default="both",
                        help="Which distance metric to compute: zg (DWPose geometry), af (AuraFace identity), or both (default)")
    p_geom.set_defaults(func=cmd_review_compute_geometry)
    
    p_enrich = sub.add_parser("enrich")
    p_enrich.add_argument("--dataset", required=True)
    p_enrich.add_argument("--passes", default="pose,seg,depth,normal,caption,t5", help="Comma-separated passes for Stratum")
    p_enrich.add_argument("--skip-stratum", action="store_true", help="Skip Stratum entirely, only extract missing AuraFace embeddings")
    p_enrich.add_argument("--status", choices=["approved", "unreviewed", "both"], default="both",
                          help="Which images to enrich: approved, unreviewed, or both (default: both)")
    p_enrich.set_defaults(func=cmd_enrich)

    args_parsed = parser.parse_args(args)
    return args_parsed.func(args_parsed)

if __name__ == "__main__":
    sys.exit(main())
