"""CLI entry point for the hegre dataset tool.

Usage: python -m tools.hegre_dataset <command> [args...]
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="hegre-dataset",
        description="Create and manage hegre face datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("discover", help="Discover identities from ground truth")
    sub.add_parser("extract-faces", help="Run MTCNN face detection")
    sub.add_parser("review", help="Start the review UI")
    sub.add_parser("enrich", help="Run stratum-hq enrichment")
    sub.add_parser("export", help="Export gate-ready dataset")
    sub.add_parser("catalog", help="List/manage dataset versions")

    args = parser.parse_args()
    print(f"Command '{args.command}' not yet implemented.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
