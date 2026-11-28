"""Drop the runs whose final picture was moved into <collection>/final_views/excluded/.

After looking through final_views/ (written by collect_fold_demos.py), move the pictures of badly
folded runs into final_views/excluded/ and run

    python3 scripts/grasp/reject_fold_demos.py data/demonstrations/physical_fold/<collection>

For each excluded picture (seed<seed>_...png) the run folder of that seed is
  - renamed from <run time>_<tag> to <run time>_rejected, and
  - its high_level.jsonl renamed to high_level.jsonl.rejected, because the BC loader
    (scripts/learning/demonstration_dataset.py) finds episodes by that file, not by the folder name.
  - its row in collect_summary.csv changed to status=rejected, success=False, run_dir=<new folder>,
    so the success count (and --resume, which collects replacements) no longer counts it.
Nothing is deleted. Every rejection is appended to <collection>/rejected.csv. Running it again is safe.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

SEED = re.compile(r"^seed(\d+)_")


def read_summary(collection):
    with (collection / "collect_summary.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_summary(collection, fieldnames, rows):
    path = collection / "collect_summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def find_run_dir(recorded):
    # The CSV keeps the path the run had when it ended; it may since have been renamed to _rejected
    if recorded.is_dir():
        return recorded
    rejected = recorded.with_name(recorded.name.rsplit("_", 1)[0] + "_rejected")
    return rejected if rejected.is_dir() else None


def reject(collection):
    collection = collection.resolve()
    excluded = collection / "final_views" / "excluded"
    if not excluded.is_dir():
        sys.exit(f"No {excluded}: move the pictures of runs to drop into it first")
    fieldnames, rows = read_summary(collection)
    rows_by_seed = {int(r["seed"]): r for r in rows if r.get("run_dir")}
    runs = {seed: Path(r["run_dir"]) for seed, r in rows_by_seed.items()}
    log_path = collection / "rejected.csv"
    new_log = not log_path.is_file()
    changed = []
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        log = csv.writer(handle)
        if new_log:
            log.writerow(["seed", "picture", "run_dir", "rejected_at"])
        for picture in sorted(excluded.glob("*.png")):
            match = SEED.match(picture.name)
            if not match or int(match.group(1)) not in runs:
                print(f"[reject] skipped {picture.name}: no run with that seed in collect_summary.csv")
                continue
            seed = int(match.group(1))
            run_dir = find_run_dir(runs[seed])
            if run_dir is None:
                print(f"[reject] skipped {picture.name}: run folder not found ({runs[seed]})")
                continue
            if run_dir.name.endswith("_rejected"):
                continue  # already done
            target = run_dir.with_name(run_dir.name.rsplit("_", 1)[0] + "_rejected")
            run_dir.rename(target)
            data = target / "high_level.jsonl"
            if data.is_file():
                data.rename(target / "high_level.jsonl.rejected")
            rows_by_seed[seed].update(status="rejected", success="False", run_dir=str(target))
            log.writerow([seed, picture.name, str(target), datetime.now().isoformat(timespec="seconds")])
            changed.append(target)
            print(f"[reject] seed {seed}: {run_dir.name} -> {target.name}")
    if changed:
        write_summary(collection, fieldnames, rows)
    successes = sum(r["success"] == "True" for r in rows)
    print(f"[reject] {len(changed)} run(s) rejected; successes left {successes}/{len(rows)} runs; log: {log_path}")
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collection", type=Path, help="Collection folder made by collect_fold_demos.py")
    reject(parser.parse_args(argv).collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
