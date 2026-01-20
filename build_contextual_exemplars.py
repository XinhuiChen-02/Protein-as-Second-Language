#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage example:
"""

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

SEQ_RE = re.compile(r"<seq>(.*?)</seq>", flags=re.S | re.I)

def safe_json_load(file_path: str) -> List[Dict[str, Any]]:

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            print(f"[safe_json_load] Loaded list with {len(obj)} items from {file_path}")
            return obj
        elif isinstance(obj, dict):
            print(f"[safe_json_load] Loaded dict from {file_path}, wrapping into list[dict].")
            return [obj]
    except json.JSONDecodeError:
        print(f"[safe_json_load] Whole-file JSON decode failed, trying jsonl for {file_path} ...")

    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line_clean = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", line)
            if not line_clean.strip():
                continue
            try:
                data.append(json.loads(line_clean))
            except json.JSONDecodeError:
                print(f"[safe_json_load] Warning: line {lineno} is not valid JSON, skipped.")
                continue

    print(f"[safe_json_load] Loaded {len(data)} items (jsonl) from {file_path}")
    if not data:
        raise ValueError(f"[safe_json_load] No valid JSON objects found in {file_path}")
    return data


def extract_triple(item: Dict[str, Any]) -> Tuple[str, str, str]:

    user_txt = next(c["value"] for c in item["conversations"] if c.get("from") == "user")
    m = SEQ_RE.search(user_txt)
    seq = m.group(1).strip() if m else ""
    full_question = user_txt.strip()
    answer = next((c["value"] for c in item["conversations"] if c.get("from") == "assistant"), "")
    return seq, full_question, answer

def mmseqs_similarity(query_seq: str, mmseqs_db: Path, top_k: int = 10000) -> List[Tuple[int, float]]:

    if not query_seq.strip():
        return []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as fq:
        fq.write(">query\n" + query_seq.strip().replace(" ", "").replace("\n", "") + "\n")
        query_fa = fq.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".m8", delete=False) as fr:
        result_m8 = fr.name

    tmp_dir = tempfile.mkdtemp()

    cmd = [
        "mmseqs", "easy-search",
        query_fa,
        str(mmseqs_db),
        result_m8,
        tmp_dir,
        "--format-output", "query,target,pident",
        "--max-seqs", str(top_k),
        "--threads", "8",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[MMseqs] easy-search failed:")
        print(proc.stderr)
        raise RuntimeError(proc.stderr)

    hits: List[Tuple[int, float]] = []
    with open(result_m8, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            _, target, pident = parts
            try:
                ref_idx = int(target.split("_")[1])
            except Exception:
                continue
            score = float(pident) / 100.0
            hits.append((ref_idx, score))

    try:
        os.unlink(query_fa)
    except OSError:
        pass
    try:
        os.unlink(result_m8)
    except OSError:
        pass

    return hits



def build_ref_cache(ref_items: List[Dict[str, Any]], cache_path: Path):
    ref_triples = [extract_triple(r) for r in ref_items]
    qa_texts = [f"{q} {a}" for _, q, a in ref_triples]
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform(tqdm(qa_texts, desc="Build TF-IDF cache"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((vect, X, ref_items), cache_path)
    print(f"[Cache] TF-IDF saved to {cache_path}, shape={X.shape}")
    return vect, X, ref_items


def load_or_build_cache(ref_items: List[Dict[str, Any]], cache_path: Path):
    if cache_path.exists():
        vect, X, cached_items = joblib.load(cache_path)
        if cached_items == ref_items:
            print(f"[Cache] TF-IDF loaded from {cache_path}, shape={X.shape}")
            return vect, X, ref_items
        else:
            print("[Cache] ref_items changed, rebuilding TF-IDF cache ...")
    return build_ref_cache(ref_items, cache_path)



def compute_similarity(
    current_item: Dict[str, Any],
    ref_items: List[Dict[str, Any]],
    vect: TfidfVectorizer,
    ref_X,
    mmseqs_db: Path,
    w_seq: float = 0.5,
    debug: bool = False,
    nonself: bool = True,
    identity_threshold: float = 0.9999,
) -> np.ndarray:

    cur_seq, cur_q, _ = extract_triple(current_item)
    n_ref = ref_X.shape[0]

    mmseqs_hits = mmseqs_similarity(cur_seq, mmseqs_db=mmseqs_db, top_k=10000)

    if debug:
        print("\n========== DEBUG: MMseqs for current query ==========")
        print(f"Total raw hits: {len(mmseqs_hits)}")
        print("Raw hits sample (first 10):", mmseqs_hits[:10])
        print("n_ref =", n_ref)

    mmseqs_hits = [(idx, score) for idx, score in mmseqs_hits if 0 <= idx < n_ref]

    if debug:
        print(f"Hits after index filtering (within [0, {n_ref})):", len(mmseqs_hits))
        print("Filtered hits sample (first 10):", mmseqs_hits[:10])

    if nonself:
        self_like_indices = {idx for idx, score in mmseqs_hits if score >= identity_threshold}
        mmseqs_map = {idx: score for idx, score in mmseqs_hits if idx not in self_like_indices}
    else:
        self_like_indices = set()
        mmseqs_map = {idx: score for idx, score in mmseqs_hits}

    if debug:
        print("self_like_indices (sample):", list(self_like_indices)[:10])
        print("mmseqs_map keys (sample):", list(mmseqs_map.keys())[:10])
        print("max s_score:", max(mmseqs_map.values()) if mmseqs_map else 0.0)

    cur_vec = vect.transform([cur_q])
    qa_scores = cosine_similarity(cur_vec, ref_X).ravel()

    scores: List[float] = []
    for idx in range(n_ref):
        if nonself and idx in self_like_indices:
            scores.append(-1e9)
            continue
        s_score = mmseqs_map.get(idx, 0.0)
        qa_score = qa_scores[idx]
        scores.append(w_seq * s_score + (1.0 - w_seq) * qa_score)

    scores_arr = np.array(scores)

    if debug:
        top5_idx = np.argsort(scores_arr)[::-1][:5]
        print("Top5 indices by combined score:", top5_idx)
        print("Top5 scores:", scores_arr[top5_idx])
        print("Corresponding s_score and qa_score:")
        for i_idx in top5_idx:
            s = mmseqs_map.get(i_idx, 0.0)
            q = qa_scores[i_idx]
            print(f"  idx={i_idx}, s_seq={s:.4f}, s_text={q:.4f}, combined={scores_arr[i_idx]:.4f}")
        print("=====================================================\n")

    return scores_arr

def select_few_shot_exemplars(
    current_item: Dict[str, Any],
    reference_data: List[Dict[str, Any]],
    vect: TfidfVectorizer,
    ref_X,
    mmseqs_db: Path,
    num_shots: int = 2,
    w_seq: float = 0.5,
    sim_threshold: float = 0.25,
    debug: bool = False,
    nonself: bool = True,
    identity_threshold: float = 0.9999,
):

    scores = compute_similarity(
        current_item=current_item,
        ref_items=reference_data,
        vect=vect,
        ref_X=ref_X,
        mmseqs_db=mmseqs_db,
        w_seq=w_seq,
        debug=debug,
        nonself=nonself,
        identity_threshold=identity_threshold,
    )

    n_ref = len(reference_data)

 
    valid_idx = np.where(scores >= sim_threshold)[0]
    if len(valid_idx) == 0:
        return [], np.array([], dtype=int)

 
    top_idx = valid_idx[np.argsort(scores[valid_idx])[::-1][:num_shots]]

    if debug:
        print("[DEBUG] valid_idx size:", len(valid_idx))
        print("[DEBUG] selected top_idx:", top_idx)

    exemplars = []
    for idx in top_idx:
        seq, q, a = extract_triple(reference_data[idx])
        exemplars.append(
            {
                "ref_index": int(idx),
                "seq": seq,
                "user": q,
                "assistant": a,
                "score": float(scores[idx]),
            }
        )

    return exemplars, top_idx



def process_questions(
    question_data: List[Dict[str, Any]],
    reference_data: List[Dict[str, Any]],
    mmseqs_db: Path,
    tfidf_cache: Path,
    output_path: Path,
    num_shots: int = 4,
    w_seq: float = 0.5,
    max_ref: int = None,
    sim_threshold: float = 0.0,
    nonself: bool = True,
    identity_threshold: float = 0.9999,
    debug_first_n: int = 3,
):

    if max_ref is not None:
        reference_data = reference_data[:max_ref]
    print(f"[Info] Reference size: {len(reference_data)}")

    vect, ref_X, _ = load_or_build_cache(reference_data, cache_path=tfidf_cache)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "id",
            "system",
            "question",
            "answer",
            "few_shot_indices",
            "few_shot_examples",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, item in enumerate(tqdm(question_data), 1):
            _, q_text, ans_text = extract_triple(item)
            debug_flag = (i <= debug_first_n)

            exemplars, idxs = select_few_shot_exemplars(
                current_item=item,
                reference_data=reference_data,
                vect=vect,
                ref_X=ref_X,
                mmseqs_db=mmseqs_db,
                num_shots=num_shots,
                w_seq=w_seq,
                sim_threshold=sim_threshold,
                debug=debug_flag,
                nonself=nonself,
                identity_threshold=identity_threshold,
            )

            system_prompt = next(
                (c["value"] for c in item["conversations"] if c.get("from") == "system"),
                "",
            )

            writer.writerow(
                {
                    "id": item.get("id", str(i)),
                    "system": system_prompt,
                    "question": q_text,
                    "answer": ans_text,
                    "few_shot_indices": ",".join(str(int(x)) for x in idxs),
                    "few_shot_examples": json.dumps(exemplars, ensure_ascii=False),
                }
            )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build contextual exemplars (few-shot) for protein QA datasets using MMseqs2 + TF-IDF."
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to query/test JSON/JSONL file.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference JSON/JSONL file (aligned with ref.fasta / refDB).",
    )
    parser.add_argument(
        "--mmseqs-db",
        type=str,
        required=True,
        help="Path to MMseqs2 database directory (created from ref.fasta).",
    )
    parser.add_argument(
        "--tfidf-cache",
        type=str,
        default="data/ref_tfidf.joblib",
        help="Path to TF-IDF cache file (joblib).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file for contextual exemplars.",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=4,
        help="Number of exemplars per query (default: 4).",
    )
    parser.add_argument(
        "--w-seq",
        type=float,
        default=0.5,
        help="Weight of sequence similarity in combined score (default: 0.5).",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.0,
        help="Minimum combined similarity score to accept exemplars (default: 0.0).",
    )
    parser.add_argument(
        "--max-ref",
        type=int,
        default=None,
        help="Optional: truncate reference set to first N items (for debugging).",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.9999,
        help="Identity threshold for Non-Self filtering (default: 0.9999).",
    )
    parser.add_argument(
        "--allow-self",
        action="store_true",
        help="Disable Non-Self filtering (i.e., allow exemplars with identity ≥ threshold).",
    )
    parser.add_argument(
        "--debug-first-n",
        type=int,
        default=3,
        help="Number of first queries to print detailed MMseqs debug info (default: 3).",
    )
    return parser.parse_args()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()

    queries_path = Path(args.queries)
    reference_path = Path(args.reference)
    mmseqs_db = Path(args.mmseqs_db)
    tfidf_cache = Path(args.tfidf_cache)
    output_path = Path(args.output)

    q_data = safe_json_load(str(queries_path))
    r_data = safe_json_load(str(reference_path))

    print(f"[Info] Loaded {len(q_data)} queries from {queries_path}")
    print(f"[Info] Loaded {len(r_data)} reference items from {reference_path}")
    print(f"[Info] MMseqs DB: {mmseqs_db}")
    print(f"[Info] TF-IDF cache: {tfidf_cache}")
    print(f"[Info] Output CSV: {output_path}")

    nonself = not args.allow_self

    process_questions(
        question_data=q_data,
        reference_data=r_data,
        mmseqs_db=mmseqs_db,
        tfidf_cache=tfidf_cache,
        output_path=output_path,
        num_shots=args.num_shots,
        w_seq=args.w_seq,
        max_ref=args.max_ref,
        sim_threshold=args.sim_threshold,
        nonself=nonself,
        identity_threshold=args.identity_threshold,
        debug_first_n=args.debug_first_n,
    )


if __name__ == "__main__":
    main()
