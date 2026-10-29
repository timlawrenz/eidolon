"""Read-only dump of this job's cron execution rows + registry state.

Usage: python cron_state.py [YYYY-MM-DD]   (default: last 4 days)
"""
import json
import sqlite3
import sys
from pathlib import Path

CRON = Path("/home/tim/.hermes/profiles/eidolon/cron")
since = sys.argv[1] if len(sys.argv) > 1 else "2026-10-26"

db = sqlite3.connect(f"file:{CRON/'executions.db'}?mode=ro", uri=True)
db.row_factory = sqlite3.Row
cols = [r[1] for r in db.execute("PRAGMA table_info(executions)")]
print("executions columns:", ",".join(cols))
rows = db.execute(
    "SELECT * FROM executions WHERE started_at >= ? ORDER BY started_at", (since,)
).fetchall()
print(f"\n--- executions since {since} ---")
for r in rows:
    d = dict(r)
    print({k: d.get(k) for k in
           ("job_id", "status", "started_at", "finished_at", "exit_code", "error")})

print("\n--- registry ---")
jobs = json.loads((CRON / "jobs.json").read_text())
entries = jobs if isinstance(jobs, list) else jobs.get("jobs", jobs)
if isinstance(entries, dict):
    entries = list(entries.values())
for j in entries:
    if "lda" not in json.dumps(j).lower():
        continue
    for k in ("id", "name", "schedule", "enabled", "state", "last_status",
              "last_error", "failure_streak", "repeat", "last_run_at",
              "next_run_at", "deliver"):
        if k in j:
            print(f"{k} = {j[k]}")
