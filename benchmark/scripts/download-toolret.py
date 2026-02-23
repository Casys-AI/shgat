#!/usr/bin/env python3
"""
Download ToolRet datasets from HuggingFace and save as JSON.
Embedding is handled separately by embed-toolret.ts (Deno/BGE-M3).

Usage:
    python3 scripts/download-toolret.py
    python3 scripts/download-toolret.py --subset web --task toolbench
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download ToolRet data")
    parser.add_argument("--subset", default="web", choices=["web", "code", "customized"])
    parser.add_argument("--task", default="toolbench")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Tools ---
    print(f"[download] Loading ToolRet-Tools (config={args.subset})...")
    ds_tools = load_dataset("mangopy/ToolRet-Tools", args.subset, split="tools")
    print(f"[download] {len(ds_tools)} tools")

    tools = []
    for item in ds_tools:
        doc_str = item["documentation"]
        category_name = None
        tool_name = None
        api_name = None
        try:
            doc = json.loads(doc_str)
            if isinstance(doc, dict):
                category_name = doc.get("category_name")
                tool_name = doc.get("tool_name")
                api_name = doc.get("api_name") or doc.get("name")
        except (json.JSONDecodeError, TypeError):
            pass

        tools.append({
            "id": item["id"],
            "documentation": doc_str,
            "category_name": category_name,
            "tool_name": tool_name,
            "api_name": api_name,
        })

    # --- Queries ---
    print(f"[download] Loading ToolRet-Queries (config={args.task})...")
    ds_queries = load_dataset("mangopy/ToolRet-Queries", args.task, split="queries")
    print(f"[download] {len(ds_queries)} queries")

    queries = []
    for item in ds_queries:
        labels = json.loads(item["labels"]) if isinstance(item["labels"], str) else item["labels"]
        queries.append({
            "id": item["id"],
            "query": item["query"],
            "labels": [{"id": l["id"], "relevance": l.get("relevance", 1)} for l in labels],
        })

    # --- Hierarchy ---
    # ToolRet has category_name but no tool_name/tool_server.
    # Build 2-level hierarchy: Category → tool IDs (leaf APIs).
    categories = {}  # category_name → [tool_id, ...]
    category_of_tool = {}  # tool_id → category_name
    for t in tools:
        cat = t.get("category_name")
        if cat:
            categories.setdefault(cat, set()).add(t["id"])
            category_of_tool[t["id"]] = cat

    hierarchy = {
        "categories": {k: sorted(v) for k, v in categories.items()},
        "category_of_tool": category_of_tool,
    }

    # --- Save ---
    tools_path = out_dir / "tools-raw.json"
    queries_path = out_dir / "queries-raw.json"
    hierarchy_path = out_dir / "hierarchy.json"

    with open(tools_path, "w") as f:
        json.dump(tools, f)
    print(f"[save] Tools → {tools_path} ({os.path.getsize(tools_path) / 1e6:.1f} MB)")

    with open(queries_path, "w") as f:
        json.dump(queries, f)
    print(f"[save] Queries → {queries_path} ({os.path.getsize(queries_path) / 1e6:.1f} MB)")

    with open(hierarchy_path, "w") as f:
        json.dump(hierarchy, f, indent=2)
    print(f"[save] Hierarchy → {hierarchy_path}")

    n_cats = len(hierarchy["categories"])
    n_with_cat = len(category_of_tool)
    print(f"\n=== Summary ===")
    print(f"Tools:      {len(tools)}")
    print(f"Queries:    {len(queries)}")
    print(f"Categories: {n_cats}")
    print(f"Tools with hierarchy: {n_with_cat}/{len(tools)}")
    print(f"\nNext: deno run --allow-all --config deno.json scripts/embed-toolret.ts")


if __name__ == "__main__":
    main()
