"""Identity discovery and image manifest construction.

Scans the hegre ground truth directory, groups images by identity
using suffix-aware keys, and produces a manifest of all images
eligible for face extraction.
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def idkey(slug: str) -> str:
    """Extract a suffix-aware identity key from a set slug.
    
    Examples:
        'darina-l' -> 'darina-l'  (suffix '-l' distinguishes from 'darina')
        'keity-climbing' -> 'keity'
        'muriel' -> 'muriel'
    """
    t = slug.split("-")
    k = t[0]
    if len(t) > 1 and len(t[1]) <= 2 and t[1].isalpha():
        k = f"{t[0]}-{t[1]}"
    return k


def discover_identities(root: Path, min_sets: int = 3) -> dict[str, list[str]]:
    """Scan ground truth for identities with at least min_sets photo shoots.
    
    Args:
        root: Path to ground truth directory (e.g., /mnt/.../hegre-14000px/).
        min_sets: Minimum number of distinct shoots an identity must have.
    
    Returns:
        {identity_key: [set_dir_name, ...]} sorted by set count descending.
    """
    by_id: dict[str, list[str]] = defaultdict(list)
    
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {root}")
        
    for d in sorted(os.listdir(root)):
        if not re.match(r'^\d+_', d):
            continue
        if not (root / d).is_dir():
            continue
        slug = d.split("_", 1)[1]
        by_id[idkey(slug)].append(d)
    
    ranked = sorted(by_id.items(), key=lambda kv: len(kv[1]), reverse=True)
    return {k: sets for k, sets in ranked if len(sets) >= min_sets}


def build_manifest(
    root: Path,
    identities: dict[str, list[str]],
    max_identities: int | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Build a manifest of all images for selected identities.
    
    Every image in every set for each identity is included.
    No round-robin sampling — all images are surfaced for review.
    
    Args:
        root: Path to ground truth directory.
        identities: {identity_key: [set_dir_name, ...]} from discover_identities.
        max_identities: If set, limit to top-N identities by set count.
    
    Returns:
        {identity_key: [{set_slug, image_path, filename}, ...]}
    """
    if max_identities:
        identities = dict(list(identities.items())[:max_identities])
    
    manifest: dict[str, list[dict[str, str]]] = {}
    
    for model, set_dirs in identities.items():
        entries: list[dict[str, str]] = []
        for s in set_dirs:
            set_path = root / s
            slug = s.split("_", 1)[1]
            for f in sorted(os.listdir(set_path)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("_"):
                    entries.append({
                        "set_slug": slug,
                        "filename": f,
                        "image_path": str(set_path / f),
                    })
        manifest[model] = entries
    
    return manifest


def save_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    """Write manifest.json to the dataset directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
