"""Read-only review-DB status census. Committed 2026-10-17 by the scheduled cron audit run."""
import sys
from pathlib import Path

sys.path.insert(0, "/home/tim/source/activity/eidolon")
from tools.hegre_dataset.dataset import HegreDataset

ds = HegreDataset(Path("/mnt/nas-ai-models/training-data/eidolon/hegre-faces/v1"))
rows = ds.db.execute(
    "SELECT status, COUNT(*) c FROM images GROUP BY status ORDER BY c DESC"
).fetchall()
tot = 0
for r in rows:
    print(f"{r['status']:35s} {r['c']}")
    tot += r["c"]
print(f"{'TOTAL':35s} {tot}")
