#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage example:

    python scripts/build_ref_db.py \
        --input data/Attribute-based_QA.json data/Knowledge-based_QA.json/... \
        --merged-json data/ref_merged.json \
        --out-fasta data/ref.fasta \
        --mmseqs-db data/refDB

"""

import argparse
import json
import re
import os
from pathlib import Path
import subprocess
from typing import List, Any

from tqdm import tqdm


SEQ_RE = re.compile(r"<seq>(.*?)</seq>", flags=re.S | re.I)


def safe_json_load(file_path: str) -> List[Any]:

    data: List[Any] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    try:
        obj = json.loads("".join(lines))
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    for line in lines:
        line = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', line)
        if not line.strip():
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return data


def extract_seq_from_item(item: dict) -> str:
    try:
        user_txt = next(
            c["value"]
            for c in item.get("conversations", [])
            if c.get("from") == "user"
        )
    except StopIteration:
        return ""

    m = SEQ_RE.search(user_txt)
    if not m:
        return ""

    seq = m.group(1)
    seq = seq.replace(" ", "").replace("\n", "").strip()
    return seq


def run_mmseqs_createdb(fasta_path: Path, db_path: Path) -> None:
    cmd = [
        "mmseqs", "createdb",
        str(fasta_path),
        str(db_path)
    ]
    print(f"[MMSEQS] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("mmseqs createdb failed")
    print("[MMSEQS] Database created.")


def build_ref_db(
    input_paths: List[Path],
    merged_json: Path,
    out_fasta: Path,
    mmseqs_db: Path,
    skip_mmseqs: bool = False
) -> None:
    
    all_items = []
    for p in input_paths:
        print(f"[Load] {p}")
        items = safe_json_load(str(p))
        print(f"  -> {len(items)} items")
        all_items.extend(items)

    print(f"[Merge] Total items = {len(all_items)}")

    merged_json.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    mmseqs_db.parent.mkdir(parents=True, exist_ok=True)

    with merged_json.open("w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"[Write] Merged JSON saved to {merged_json}")

    skipped = 0
    with out_fasta.open("w", encoding="utf-8") as fa:
        for idx, item in enumerate(tqdm(all_items, desc="Writing FASTA")):
            seq = extract_seq_from_item(item)
            if not seq:
                skipped += 1
                continue
            fa.write(f">ref_{idx}\n{seq}\n")

    print(f"[Write] FASTA saved to {out_fasta}")
    print(f"[Info] Missing sequences skipped: {skipped}")

    if skip_mmseqs:
        print("[MMSEQS] Skipped database creation (skip_mmseqs=True).")
    else:
        run_mmseqs_createdb(out_fasta, mmseqs_db)

    print("\n=== ALL DONE ===")
    print(f"Final DB path: {mmseqs_db}")
    print(f"FASTA: {out_fasta}")
    print(f"Merged JSON: {merged_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reference JSON/FASTA and MMseqs DB from QA datasets with <seq> tags."
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="Input JSON/JSONL files containing conversations with <seq>...</seq> in user messages."
    )
    parser.add_argument(
        "--merged-json",
        type=str,
        default="data/ref_merged.json",
        help="Path to save merged reference JSON (default: data/ref_merged.json)."
    )
    parser.add_argument(
        "--out-fasta",
        type=str,
        default="data/ref.fasta",
        help="Path to save reference FASTA file (default: data/ref.fasta)."
    )
    parser.add_argument(
        "--mmseqs-db",
        type=str,
        default="data/refDB",
        help="Path to MMseqs2 database directory (default: data/refDB)."
    )
    parser.add_argument(
        "--skip-mmseqs",
        action="store_true",
        help="If set, do NOT run `mmseqs createdb` (only generate merged JSON and FASTA)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.input]
    merged_json = Path(args.merged_json)
    out_fasta = Path(args.out_fasta)
    mmseqs_db = Path(args.mmseqs_db)

    build_ref_db(
        input_paths=input_paths,
        merged_json=merged_json,
        out_fasta=out_fasta,
        mmseqs_db=mmseqs_db,
        skip_mmseqs=args.skip_mmseqs,
    )


if __name__ == "__main__":
    main()
