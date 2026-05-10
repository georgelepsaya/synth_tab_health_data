#!/usr/bin/env python3
"""
Convert printed pandas DataFrame text dumps into proper CSVs, mirroring
a source directory tree into a destination directory.

Pandas wraps wide DataFrames into several blocks separated by blank lines,
with each block's header line ending in a trailing backslash (except the last).
This script reassembles all blocks into a single CSV per input file.

Robust to:
  - varying number of blocks per file
  - varying number of metrics (rows) per file
  - metric names containing spaces (e.g. "privacy.distinct l-diversity.gt")
  - negative values, scientific notation

Usage:
    python converter.py raw_results csv_results

    Walks raw_results/ for *.txt files, writes a matching *.csv into
    csv_results/ keeping the same subfolder structure. Existing CSVs are
    overwritten.
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_printout(text: str):
    """Parse a pandas DataFrame text dump into {metric: {col: value}}.

    Returns (metrics_dict, ordered_column_list).
    """
    blocks = re.split(r"\n\s*\n", text.strip("\n"))

    metrics: dict[str, dict[str, str]] = {}
    column_order: list[str] = []

    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        # First line is the header; strip the trailing "\" pandas adds
        header = lines[0].rstrip().rstrip("\\").strip()
        cols = header.split()
        for c in cols:
            if c not in column_order:
                column_order.append(c)
        n = len(cols)
        if n == 0:
            continue

        for line in lines[1:]:
            tokens = line.split()
            if len(tokens) < n:
                continue  # malformed / continuation line, skip
            values = tokens[-n:]
            name = " ".join(tokens[:-n]).strip()
            if not name:
                continue
            metrics.setdefault(name, {})
            for col, val in zip(cols, values):
                metrics[name][col] = val

    return metrics, column_order


def write_csv(metrics, columns, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + columns)  # empty cell for the metric-name index
        for metric, vals in metrics.items():
            writer.writerow([metric] + [vals.get(c, "") for c in columns])


def convert_tree(src_root: Path, dst_root: Path):
    txt_files = sorted(src_root.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found under {src_root}")
        return

    for txt in txt_files:
        rel = txt.relative_to(src_root)
        out_path = dst_root / rel.with_suffix(".csv")
        try:
            text = txt.read_text()
            metrics, cols = parse_printout(text)
            write_csv(metrics, cols, out_path)
            print(f"  {rel} -> {out_path.relative_to(dst_root.parent) if dst_root.parent in out_path.parents else out_path}  ({len(metrics)} metrics, {len(cols)} cols)")
        except Exception as e:
            print(f"  FAILED: {rel}: {e}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("src", help="Source directory (e.g. raw_results)")
    p.add_argument("dst", help="Destination directory (e.g. csv_results)")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        sys.exit(f"Source not found or not a directory: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    convert_tree(src, dst)


if __name__ == "__main__":
    main()