"""CLI entry point for the hegre dataset tool.

Usage: python -m tools.hegre_dataset <command> [args...]
"""
import argparse
import sys
from pathlib import Path


def cmd_discover(args):
    from .identity import discover_identities, build_manifest, save_manifest
    
    root = Path(args.source)
    try:
        identities = discover_identities(root, min_sets=args.min_sets)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
        
    print(f"Found {len(identities)} identities with ≥{args.min_sets} sets.")
    
    manifest = build_manifest(root, identities, max_identities=args.max_identities)
    total_images = sum(len(v) for v in manifest.values())
    print(f"Manifest: {len(manifest)} identities, {total_images} images total.")
    
    output = Path(args.dataset)
    path = save_manifest(manifest, output)
    print(f"Saved: {path}")
    return 0


def cmd_extract_faces(args):
    from .face_extraction import extract_faces
    dataset = Path(args.dataset)
    try:
        extract_faces(dataset)
    except Exception as e:
        print(f"Error extracting faces: {e}")
        return 1
    return 0


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="hegre-dataset",
        description="Create and manage hegre face datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_discover = sub.add_parser("discover", help="Discover identities from ground truth")
    p_discover.add_argument("--source", required=True, help="Path to ground truth directory")
    p_discover.add_argument("--dataset", required=True, help="Path to output dataset directory")
    p_discover.add_argument("--min-sets", type=int, default=3, help="Minimum number of sets per identity")
    p_discover.add_argument("--max-identities", type=int, help="Maximum number of identities to process")
    p_discover.set_defaults(func=cmd_discover)

    p_extract = sub.add_parser("extract-faces", help="Run MTCNN face detection")
    p_extract.add_argument("--dataset", required=True, help="Path to dataset directory")
    p_extract.set_defaults(func=cmd_extract_faces)

    sub.add_parser("review", help="Start the review UI")
    sub.add_parser("enrich", help="Run stratum-hq enrichment")
    sub.add_parser("export", help="Export gate-ready dataset")
    sub.add_parser("catalog", help="List/manage dataset versions")

    parsed_args = parser.parse_args(args)
    if hasattr(parsed_args, "func"):
        return parsed_args.func(parsed_args)
    else:
        print(f"Command '{parsed_args.command}' not yet implemented.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
