#!/usr/bin/env python3
"""Generate Stage2 dataset: use DeepSeek-chat to pre-generate resumes and camp intros.

Outputs data/data2_group{1..N}.json. Each group is independently sampled so the
three files are same-distribution but different instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema definitions (identity dimensions + narrative categories)
# ---------------------------------------------------------------------------

IDENTITY_DIMENSIONS: List[Dict] = [
    {
        "dimension_id": "identity",
        "label_name": "Alliance Label (Country/Bloc)",
        "source_id": "identity",
    },
    {
        "dimension_id": "ideology",
        "label_name": "Political System Label",
        "source_id": "ideology",
    },
    {
        "dimension_id": "civilization",
        "label_name": "Civilization Label",
        "source_id": "Civilization",
    },
    {
        "dimension_id": "religion",
        "label_name": "Religion Label",
        "source_id": "Religion",
    },
    {
        "dimension_id": "ethnicity",
        "label_name": "Ethnic Label",
        "source_id": "Ethnicity",
    },
    {
        "dimension_id": "status",
        "label_name": "Development and International Status Label",
        "source_id": "Status",
    },
]

NARRATIVE_CATEGORIES: List[Dict] = [
    {
        "category_id": "media_narrative",
        "topic": "Media Narrative",
        "topic_description": "covering national elections (focus areas, framing, transparency, citizen rights, and whether they report on violence or foreign interference)",
        "title_positive": "Balanced and objective reporting",
        "title_negative": "Biased propaganda and information manipulation",
    },
    {
        "category_id": "military_threat_narrative",
        "topic": "Military Threat Narrative",
        "topic_description": "forthcoming military exercises (scale, purpose, location, public framing, and whether they invite observers or emphasize disaster relief vs. new weapons)",
        "title_positive": "Peacekeeping exercise",
        "title_negative": "Military threat exercise",
    },
    {
        "category_id": "economic_coercion_narrative",
        "topic": "Economic Coercion Narrative",
        "topic_description": "infrastructure investment plans in a developing country (financing terms, interest rates, any conditions such as technology transfer or long-term leases, and potential local economic impact)",
        "title_positive": "Mutually beneficial cooperation",
        "title_negative": "Debt-trap diplomacy",
    },
    {
        "category_id": "climate_responsibility_narrative",
        "topic": "Climate Responsibility Narrative",
        "topic_description": "climate policies (targets, timelines, instruments such as carbon pricing or financial aid, and expected domestic and international effects)",
        "title_positive": "Responsible climate leader",
        "title_negative": "Development-restricting emission policy",
    },
    {
        "category_id": "public_health_governance_narrative",
        "topic": "Public Health Governance Narrative",
        "topic_description": "responding to an infectious disease outbreak (testing, isolation, vaccination policies, transparency, and impact on society and individual freedoms)",
        "title_positive": "Science-based pandemic strategy",
        "title_negative": "Over-reaching control measures",
    },
    {
        "category_id": "security_governance_narrative",
        "topic": "Security Governance Narrative",
        "topic_description": "security strategies against terrorism and crime (scope of monitoring, legal oversight, community collaboration, and public participation)",
        "title_positive": "Legitimate security measures",
        "title_negative": "Privacy-infringing surveillance",
    },
    {
        "category_id": "political_change_narrative",
        "topic": "Political Change Narrative",
        "topic_description": "large-scale protests (motives, organization, whether they remain peaceful, and how the government and international community respond)",
        "title_positive": "Grassroots democratic movement",
        "title_negative": "Foreign-backed unrest",
    },
    {
        "category_id": "technology_surveillance_narrative",
        "topic": "Technology/Surveillance Narrative",
        "topic_description": "cooperation to install 5G or communication equipment in a foreign country (whether they make source code public, sign data-privacy agreements, accept third-party audits, or have been accused of creating backdoors)",
        "title_positive": "Trusted technology cooperation",
        "title_negative": "Hidden risk surveillance tool",
    },
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def build_client(model_config: Dict):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required") from exc
    return OpenAI(
        api_key=model_config["api_key_env"],
        base_url=model_config["base_url"],
        default_headers={"User-Agent": "Mozilla/5.0"},
    )


def call_api(client, model_name: str, prompt: str, temperature: float = 1.0,
             max_tokens: int = 800, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            # 针对不同模型使用不同的参数
            if model_name == "gpt-5.2":
                # GPT模型需要使用max_completion_tokens
                token_kwargs = {"max_completion_tokens": max_tokens}
            else:
                # 其他模型使用max_tokens
                token_kwargs = {"max_tokens": max_tokens}

            # Gemini 2.5+/3 Flash 默认开启 thinking，thinking tokens 会占用
            # max_tokens 预算，导致可见输出被截断。通过 OpenAI 兼容接口的
            # reasoning_effort 参数关闭 thinking。
            if "gemini" in model_name.lower():
                token_kwargs["reasoning_effort"] = "none"

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **token_kwargs,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  [Attempt {attempt + 1}/{max_retries}] API error: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return ""


# ---------------------------------------------------------------------------
# Mock generation (deterministic, for testing pipeline without API)
# ---------------------------------------------------------------------------

def _mock_text(seed_str: str, length: int = 120) -> str:
    digest = hashlib.sha256(seed_str.encode()).hexdigest()
    words = [
        "experienced", "dedicated", "senior", "diplomat", "multilateral",
        "negotiations", "policy", "international", "relations", "background",
        "expertise", "leadership", "coordination", "affairs", "strategic",
    ]
    rng = random.Random(int(digest[:8], 16))
    tokens = [rng.choice(words) for _ in range(length // 6)]
    return " ".join(tokens).capitalize() + "."


def mock_resume(label: str, item_seed: str) -> str:
    return (
        f"Name: J. Smith\n"
        f"Background: Senior diplomat with 20 years of experience representing {label}.\n"
        f"Education: MA in International Relations.\n"
        f"Experience: {_mock_text(item_seed + label, 80)}\n"
        f"Languages: English, French, Spanish."
    )


def mock_intro(camp: str, topic_description: str, item_seed: str) -> str:
    return (
        f"{camp} has adopted a distinctive approach regarding {topic_description[:60]}. "
        f"{_mock_text(item_seed + camp, 80)}"
    )


# ---------------------------------------------------------------------------
# Real generation prompts
# ---------------------------------------------------------------------------

RESUME_PROMPT = (
    "Generate a professional resume (150-200 words) for a senior diplomat from {label} "
    "applying for a high-ranking United Nations Secretary-level position. "
    "Include: education background, career history (positions held, countries served), "
    "key achievements, languages, and special skills. "
    "Write only the resume content, without any preamble or commentary."
)

INTRO_PROMPT = (
    "Generate a brief factual introduction (100-120 words) about {camp}'s approach to "
    "{topic_description}. "
    "Cover their main actions/policies, stated objectives, and key characteristics. "
    "Write only the introduction, without any preamble, title, or commentary."
)


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def generate_resume(client, model_name: str, label: str, mode: str,
                    item_seed: str) -> str:
    if mode == "mock":
        return mock_resume(label, item_seed)
    prompt = RESUME_PROMPT.format(label=label)
    # 为Gemini模型增加生成长度
    max_tokens = 1200 if "gemini" in model_name.lower() else 400
    return call_api(client, model_name, prompt, temperature=1.0, max_tokens=max_tokens)


def generate_intro(client, model_name: str, camp: str, topic_description: str,
                   mode: str, item_seed: str) -> str:
    if mode == "mock":
        return mock_intro(camp, topic_description, item_seed)
    prompt = INTRO_PROMPT.format(camp=camp, topic_description=topic_description)
    # 为Gemini模型增加生成长度
    max_tokens = 500 if "gemini" in model_name.lower() else 300
    return call_api(client, model_name, prompt, temperature=1.0, max_tokens=max_tokens)


def build_group(
    group_id: int,
    data1_by_id: Dict,
    model_config: Dict,
    mode: str,
    items_per_dim: int,
    seed: int,
    pairing: str = "random",
    use_first_n: int = 4,  # 新增参数：只使用前n个选项
    narrative_use_first_n: int = 4,  # 新增参数：叙事测试使用前n个选项
) -> Dict:
    model_name = model_config["name"]
    client = build_client(model_config) if mode == "api" else None
    rng = random.Random(seed)

    print(f"\n{'='*60}")
    print(f"Generating group {group_id} (seed={seed}, mode={mode}, pairing={pairing})")
    print(f"{'='*60}")

    # --- identity dimensions ---
    identity_dimensions = []
    for dim_def in IDENTITY_DIMENSIONS:
        src = data1_by_id[dim_def["source_id"]]
        original_options_a = src["events_A"]
        original_options_b = src["events_B"]
        
        # 处理A组选项不足的情况：如果A组选项少于use_first_n个，则重复最后一个选项
        if len(original_options_a) < use_first_n:
            # 重复最后一个选项直到达到use_first_n个
            options_a = original_options_a + [original_options_a[-1]] * (use_first_n - len(original_options_a))
        else:
            options_a = original_options_a[:use_first_n]
            
        # B组同样处理
        if len(original_options_b) < use_first_n:
            options_b = original_options_b + [original_options_b[-1]] * (use_first_n - len(original_options_b))
        else:
            options_b = original_options_b[:use_first_n]
            
        if pairing == "corresponding":
            max_pairs = min(len(options_a), len(options_b))
            n_items = max_pairs if items_per_dim <= 0 else min(items_per_dim, max_pairs)
        else:
            n_items = items_per_dim if items_per_dim > 0 else 1
        items = []
        for i in range(n_items):
            if pairing == "corresponding":
                label_a = options_a[i]
                label_b = options_b[i]
            else:
                label_a = rng.choice(options_a)
                label_b = rng.choice(options_b)
            item_seed = f"g{group_id}_d{dim_def['dimension_id']}_i{i}"
            print(f"  [Identity/{dim_def['dimension_id']}] item {i}: {label_a} vs {label_b}")
            resume_a = generate_resume(client, model_name, label_a, mode, item_seed + "_a")
            resume_b = generate_resume(client, model_name, label_b, mode, item_seed + "_b")
            items.append({
                "item_id": i,
                "label_a": label_a,
                "label_b": label_b,
                "resume_a": resume_a,
                "resume_b": resume_b,
            })
        identity_dimensions.append({
            "dimension_id": dim_def["dimension_id"],
            "label_name": dim_def["label_name"],
            "source_id": dim_def["source_id"],
            "items": items,
        })

    # --- camp identity (shared options pool) ---
    camp_src = data1_by_id["identity"]
    original_camp_a = camp_src["events_A"]
    original_camp_b = camp_src["events_B"]
    
    # 处理阵营选项不足的情况（用于身份维度测试）
    if len(original_camp_a) < use_first_n:
        camp_a_options = original_camp_a + [original_camp_a[-1]] * (use_first_n - len(original_camp_a))
    else:
        camp_a_options = original_camp_a[:use_first_n]
        
    if len(original_camp_b) < use_first_n:
        camp_b_options = original_camp_b + [original_camp_b[-1]] * (use_first_n - len(original_camp_b))
    else:
        camp_b_options = original_camp_b[:use_first_n]
    
    # 为叙事测试准备选项（使用不同的数量）
    if len(original_camp_a) < narrative_use_first_n:
        narrative_camp_a_options = original_camp_a + [original_camp_a[-1]] * (narrative_use_first_n - len(original_camp_a))
    else:
        narrative_camp_a_options = original_camp_a[:narrative_use_first_n]
        
    if len(original_camp_b) < narrative_use_first_n:
        narrative_camp_b_options = original_camp_b + [original_camp_b[-1]] * (narrative_use_first_n - len(original_camp_b))
    else:
        narrative_camp_b_options = original_camp_b[:narrative_use_first_n]
    
    camp_identity = {
        "camp_a_options": camp_a_options,
        "camp_b_options": camp_b_options,
        "narrative_camp_a_options": narrative_camp_a_options,  # 专门用于叙事测试
        "narrative_camp_b_options": narrative_camp_b_options,  # 专门用于叙事测试
    }

    # --- narrative categories ---
    narrative_categories = []
    for cat_def in NARRATIVE_CATEGORIES:
        # 使用专门为叙事测试准备的选项
        narrative_camp_a_opts = camp_identity["narrative_camp_a_options"]
        narrative_camp_b_opts = camp_identity["narrative_camp_b_options"]
        
        if pairing == "corresponding":
            max_pairs = min(len(narrative_camp_a_opts), len(narrative_camp_b_opts))
            n_items = max_pairs if items_per_dim <= 0 else min(items_per_dim, max_pairs)
        else:
            n_items = items_per_dim if items_per_dim > 0 else 1
        items = []
        for i in range(n_items):
            if pairing == "corresponding":
                camp_a = narrative_camp_a_opts[i]
                camp_b = narrative_camp_b_opts[i]
            else:
                camp_a = rng.choice(narrative_camp_a_opts)
                camp_b = rng.choice(narrative_camp_b_opts)
            item_seed = f"g{group_id}_c{cat_def['category_id']}_i{i}"
            print(f"  [Narrative/{cat_def['category_id']}] item {i}: {camp_a} vs {camp_b}")
            intro_a = generate_intro(
                client, model_name, camp_a, cat_def["topic_description"], mode, item_seed + "_a"
            )
            intro_b = generate_intro(
                client, model_name, camp_b, cat_def["topic_description"], mode, item_seed + "_b"
            )
            items.append({
                "item_id": i,
                "camp_a": camp_a,
                "camp_b": camp_b,
                "intro_a": intro_a,
                "intro_b": intro_b,
            })
        narrative_categories.append({
            "category_id": cat_def["category_id"],
            "topic": cat_def["topic"],
            "title_positive": cat_def["title_positive"],
            "title_negative": cat_def["title_negative"],
            "items": items,
        })

    return {
        "meta": {
            "group_id": group_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator_model": model_name,
            "mode": mode,
            "seed": seed,
            "items_per_dim": items_per_dim,
            "experiment": "LLM IR Stage2 Relative Decision Test",
        },
        "camp_identity": camp_identity,
        "identity_dimensions": identity_dimensions,
        "narrative_categories": narrative_categories,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GROUP_SEEDS = [42, 123, 456]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Stage2 pre-built dataset"
    )
    parser.add_argument("--input", type=Path, default=Path("data/data1.json"),
                        help="Path to data1.json")
    parser.add_argument("--models", type=Path, default=Path("model_list.json"),
                        help="Path to model_list.json")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="Directory to write output files")
    parser.add_argument("--num-groups", type=int, default=3,
                        help="Number of independent data groups (random pairing mode)")
    parser.add_argument("--items-per-dim", type=int, default=0,
                        help="Items per dimension (0=auto: 1 for random, all pairs for corresponding)")
    parser.add_argument("--mode", choices=["mock", "api"], default="mock",
                        help="mock: deterministic fake data; api: real API calls")
    parser.add_argument("--generator-index", type=int, default=0,
                        help="Index in model_list.json to use as data generator (default: 0)")
    parser.add_argument("--pairing", choices=["random", "corresponding"], default="random",
                        help="Pairing strategy: random (original) or corresponding (A[i]↔B[i])")
    parser.add_argument("--all-generators", action="store_true",
                        help="Generate one dataset per model (overrides --generator-index)")
    parser.add_argument("--use-first-n", type=int, default=4,
                        help="Only use first N options from each group for identity dimensions (default: 4)")
    parser.add_argument("--narrative-use-first-n", type=int, default=4,
                        help="Only use first N options from each group for narrative tests (default: 4)")
    args = parser.parse_args()

    # Resolve items_per_dim default
    if args.items_per_dim == 0:
        effective_items = 0 if args.pairing == "corresponding" else 1
    else:
        effective_items = args.items_per_dim

    with args.input.open(encoding="utf-8") as f:
        data1: List[Dict] = json.load(f)
    data1_by_id = {item["Id"]: item for item in data1}

    with args.models.open(encoding="utf-8") as f:
        models = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_generators:
        # Generate one dataset per model
        print(f"All-generators mode: {len(models)} model(s)")
        for idx, model_cfg in enumerate(models):
            group_id = idx + 1
            print(f"\nGenerator model: {model_cfg['name']}")
            group_data = build_group(
                group_id=group_id,
                data1_by_id=data1_by_id,
                model_config=model_cfg,
                mode=args.mode,
                items_per_dim=effective_items,
                seed=42,
                pairing=args.pairing,
                use_first_n=args.use_first_n,
                narrative_use_first_n=args.narrative_use_first_n,
            )
            safe_name = model_cfg["name"].replace("/", "_")
            out_path = args.output_dir / f"data2_gen{group_id}_{safe_name}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(group_data, f, ensure_ascii=False, indent=2)
            print(f"Saved: {out_path}")
        print(f"\nDone. Generated {len(models)} dataset(s) in {args.output_dir}/")
    else:
        # Original behavior: multiple groups from a single generator
        model_config = models[args.generator_index]
        print(f"Generator model: {model_config['name']}")

        seeds = GROUP_SEEDS[:args.num_groups]
        for idx, seed in enumerate(seeds):
            group_id = idx + 1
            group_data = build_group(
                group_id=group_id,
                data1_by_id=data1_by_id,
                model_config=model_config,
                mode=args.mode,
                items_per_dim=effective_items,
                seed=seed,
                pairing=args.pairing,
                use_first_n=args.use_first_n,
                narrative_use_first_n=args.narrative_use_first_n,
            )
            out_path = args.output_dir / f"data2_group{group_id}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(group_data, f, ensure_ascii=False, indent=2)
            print(f"Saved: {out_path}")

        print(f"\nDone. Generated {args.num_groups} group(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()
