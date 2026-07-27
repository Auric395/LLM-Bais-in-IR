#!/usr/bin/env python3
"""
Stage 2: Merge raw result files by judge model with correct generator labels.

The experiment filenames only carry ``group_id`` (1 or 2) from the source data
file's meta. Because three of the four generator files share group_id=1,
``_g1_`` files are a mixture of DeepSeek / ChatGPT / Gemini generated
materials.

This script identifies the true generator by processing order: the pipeline
ran the four source files alphabetically (gen1 → gen2 → gen3 → gen4), and
within each source file it produced labeled then unlabeled results. Sorting
result files by timestamp within a (judge, label_mode) group therefore gives
the generator directly. Candidate-name matching is used as a cross-check.

Input:
    stage2/results/202*                  (one directory per judge run)
    data/run_20260225_165913/data2_gen*.json  (source generator files)

Output:
    stage2/results/merged_by_judge/{judge_model}_results.json
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).parent
RAW_DIR = ROOT / "results"
DATA_DIR = ROOT.parent / "data" / "run_20260225_165913"
OUT_DIR = ROOT / "results" / "merged_by_judge"

MODEL_LABELS = {
    "deepseek-chat": "DeepSeek",
    "qwen-max": "Qwen",
    "gpt-5.2": "ChatGPT",
    "gemini-3-flash-preview": "Gemini",
}


def extract_first_bold_name(text: str) -> str:
    """Extract the first bolded name line from a resume string."""
    skip_tokens = {
        "education", "career", "contact", "profile", "professional",
        "summary", "objective", "name:", "[name", "[your name]",
    }
    for line in text.splitlines():
        line = line.strip()
        m = re.search(r"\*\*(.+?)\*\*", line)
        if not m:
            continue
        name = m.group(1).strip()
        lowered = name.lower()
        if any(tok in lowered for tok in skip_tokens):
            continue
        name = re.sub(r"^(?:Ambassador|Dr\.|Sir|Ms\.|Mr\.|Mrs\.)\s+", "", name, flags=re.I)
        return name
    return ""


def normalize_name(name: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    name = re.sub(r"[^\w\s]", " ", name.lower())
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_generator_order(data_dir: Path) -> List[str]:
    """Return generator models in the order the pipeline processed them."""
    order = []
    for filepath in sorted(data_dir.glob("data2_gen*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        order.append(data["meta"]["generator_model"])
    return order


def build_generator_signatures(data_dir: Path) -> Dict[str, Dict]:
    """Build candidate-name signatures from the four source generator files."""
    sigs: Dict[str, Dict] = {}
    for filepath in sorted(data_dir.glob("data2_gen*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        generator = data["meta"]["generator_model"]
        item = data["identity_dimensions"][0]["items"][0]
        name_a = extract_first_bold_name(item["resume_a"])
        name_b = extract_first_bold_name(item["resume_b"])
        sigs[generator] = {
            "group_id": data["meta"]["group_id"],
            "names": [name_a, name_b],
            "name_tokens": [set(normalize_name(n).split()) for n in (name_a, name_b)],
        }
    return sigs


def identify_generator_by_content(result: Dict, signatures: Dict[str, Dict]) -> Tuple[str, float]:
    """Match a result file to its generator by candidate names in the response."""
    try:
        resp = result["identity_dimensions"][0]["items"][0]["iterations"][0]["response"]
    except (KeyError, IndexError):
        return "unknown", 0.0

    resp_norm = normalize_name(resp)
    resp_tokens = set(resp_norm.split())

    scores: Dict[str, float] = {}
    for generator, sig in signatures.items():
        score = 0.0
        for name, tokens in zip(sig["names"], sig["name_tokens"]):
            if not tokens:
                continue
            name_norm = normalize_name(name)
            if name_norm and name_norm in resp_norm:
                score += 10.0
            overlap = tokens & resp_tokens
            score += len(overlap) * 1.0
            last = list(tokens)[-1] if tokens else ""
            if last and len(last) > 2 and last in resp_tokens:
                score += 0.5
        scores[generator] = score

    if not scores or max(scores.values()) <= 0:
        return "unknown", 0.0
    best = max(scores, key=scores.get)
    return best, scores[best]


def extract_timestamp(filepath: Path) -> str:
    """Extract timestamp from filename or result timestamp field."""
    m = re.search(r"_(\d{8}_\d{6})\.json$", filepath.name)
    if m:
        return m.group(1)
    # Fallback to result timestamp
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("timestamp", "")
        return datetime.fromisoformat(ts).strftime("%Y%m%d_%H%M%S")
    except Exception:
        return filepath.name


def load_raw_results(raw_dir: Path) -> Tuple[List[Dict], List[Tuple]]:
    """Load all raw result files and tag them with judge/generator/label."""
    records: List[Dict] = []
    warnings: List[Tuple] = []
    signatures = build_generator_signatures(DATA_DIR)
    generator_order = get_generator_order(DATA_DIR)

    judge_dirs = [d for d in sorted(raw_dir.glob("202*")) if d.is_dir()]
    for judge_dir in judge_dirs:
        # Read all result files in this judge directory
        files = sorted(judge_dir.glob("result_stage2_*.json"))
        if not files:
            continue

        # Determine judge model from the first file
        with open(files[0], "r", encoding="utf-8") as f:
            first_result = json.load(f)
        judge_model = first_result.get("model", "unknown")

        # Group by label_mode
        by_label: Dict[str, List[Path]] = {}
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                result = json.load(f)
            label_mode = result.get("label_mode", "unknown")
            by_label.setdefault(label_mode, []).append(fp)

        for label_mode, fps in by_label.items():
            # Sort by timestamp to recover pipeline processing order
            fps_sorted = sorted(fps, key=extract_timestamp)
            if len(fps_sorted) != len(generator_order):
                warnings.append((
                    "count_mismatch",
                    str(judge_dir),
                    label_mode,
                    len(fps_sorted),
                    len(generator_order),
                ))

            for position, fp in enumerate(fps_sorted):
                if position < len(generator_order):
                    generator_by_order = generator_order[position]
                else:
                    generator_by_order = "unknown"

                with open(fp, "r", encoding="utf-8") as f:
                    result = json.load(f)

                generator_by_content, content_score = identify_generator_by_content(
                    result, signatures
                )

                # Prefer order-based assignment; use content match as verification.
                generator = generator_by_order
                if (
                    generator_by_content != "unknown"
                    and generator_by_content != generator_by_order
                    and content_score >= 10
                ):
                    # Strong name-based contradiction: flag but keep order-based
                    warnings.append((
                        "content_mismatch",
                        str(fp.relative_to(ROOT)),
                        judge_model,
                        label_mode,
                        generator_by_order,
                        generator_by_content,
                        content_score,
                    ))

                records.append({
                    "judge_model": judge_model,
                    "generator_model": generator,
                    "label_mode": label_mode,
                    "group_id": result.get("group_id", -1),
                    "source_file": str(fp.relative_to(ROOT)),
                    "generator_by_order": generator_by_order,
                    "generator_by_content": generator_by_content,
                    "content_score": content_score,
                    "result": result,
                })

    return records, warnings


def main() -> None:
    print(f"Source data dir: {DATA_DIR}")
    print(f"Raw results dir: {RAW_DIR}")
    print(f"Output dir:      {OUT_DIR}")

    generator_order = get_generator_order(DATA_DIR)
    print(f"\nGenerator processing order: {generator_order}")

    records, warnings = load_raw_results(RAW_DIR)
    print(f"\nLoaded {len(records)} result files.")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")

    # Group by judge model
    by_judge: Dict[str, List[Dict]] = {}
    for rec in records:
        by_judge.setdefault(rec["judge_model"], []).append(rec)

    print("\nRecords per judge:")
    for judge in sorted(by_judge):
        print(f"  {judge}: {len(by_judge[judge])}")

    expected = len(generator_order) * 2  # N generators × 2 label modes
    print("\nRecords per judge × generator × label:")
    for judge in sorted(by_judge):
        counts: Dict[Tuple[str, str], int] = {}
        for rec in by_judge[judge]:
            counts[(rec["generator_model"], rec["label_mode"])] = counts.get(
                (rec["generator_model"], rec["label_mode"]), 0
            ) + 1
        print(f"  {judge}: {len(counts)} distinct cells (expected {expected})")
        for (gen, lbl), n in sorted(counts.items()):
            print(f"    {gen} / {lbl}: {n}")

    os.makedirs(OUT_DIR, exist_ok=True)
    signatures = build_generator_signatures(DATA_DIR)
    for judge, recs in by_judge.items():
        safe_name = judge.replace("/", "_")
        out_path = OUT_DIR / f"{safe_name}_results.json"
        source_dirs = sorted({rec["source_file"].split("/")[1] for rec in recs})
        payload = {
            "judge_model": judge,
            "judge_label": MODEL_LABELS.get(judge, judge),
            "source_dirs": source_dirs,
            "num_results": len(recs),
            "generator_order": generator_order,
            "generator_signatures": {
                gen: {
                    "group_id": sig["group_id"],
                    "names": sig["names"],
                }
                for gen, sig in signatures.items()
            },
            "results": recs,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
