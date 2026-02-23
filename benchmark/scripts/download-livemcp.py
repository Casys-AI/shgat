#!/usr/bin/env python3
"""
Download LiveMCPBench data from GitHub and prepare for SHGAT benchmark.

Downloads:
- Tool catalogue (69 servers, 525 tools, 8 categories, 3-level hierarchy)
- Task queries (95 tasks with ground-truth tool annotations)

Usage:
    python3 scripts/download-livemcp.py
    python3 scripts/download-livemcp.py --output-dir data/livemcp
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

TOOLS_URL = "https://raw.githubusercontent.com/icip-cas/LiveMCPBench/main/tools/LiveMCPTool/tools.json"
ANNOTATIONS_URL = "https://raw.githubusercontent.com/icip-cas/LiveMCPBench/main/annotated_data/all_annotations.json"


def download_json(url: str) -> any:
    print(f"[download] {url.split('/')[-1]}...")
    resp = urllib.request.urlopen(url)
    return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Download LiveMCPBench data")
    parser.add_argument("--output-dir", default="data/livemcp")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Download ---
    raw_servers = download_json(TOOLS_URL)
    raw_annotations = download_json(ANNOTATIONS_URL)

    # --- Build tool catalogue with 3-level hierarchy ---
    # Level 0: Category (8)
    # Level 1: MCP Server (69)
    # Level 2: Tool (525)
    tools = []
    hierarchy = {
        "categories": {},      # category → [server_name, ...]
        "servers": {},         # server_name → [tool_id, ...]
        "server_category": {}, # server_name → category
        "tool_server": {},     # tool_id → server_name
    }

    for server_entry in raw_servers:
        server_name = server_entry["name"]
        category = server_entry.get("category", "unknown")
        server_desc = server_entry.get("description", "")

        hierarchy["categories"].setdefault(category, []).append(server_name)
        hierarchy["server_category"][server_name] = category

        for srv_key, srv_data in server_entry.get("tools", {}).items():
            for tool in srv_data.get("tools", []):
                tool_id = f"{srv_key}:{tool['name']}"
                # Build text for embedding: server context + tool description
                text = f"[Server: {server_name}] [Category: {category}] {tool['name']}: {tool.get('description', '')}"
                input_schema = tool.get("inputSchema", {})

                tools.append({
                    "id": tool_id,
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "text": text,
                    "server_name": srv_key,
                    "server_display_name": server_name,
                    "category": category,
                    "input_schema": input_schema,
                })

                hierarchy["servers"].setdefault(srv_key, []).append(tool_id)
                hierarchy["tool_server"][tool_id] = srv_key

    # --- Build queries from annotations ---
    queries = []
    for ann in raw_annotations:
        meta = ann.get("Annotator Metadata", {})
        tools_str = meta.get("Tools", "")
        ground_truth_tools = []
        for line in tools_str.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                tool_name = line.split(". ", 1)[-1].strip()
            else:
                tool_name = line.strip()
            if tool_name:
                ground_truth_tools.append(tool_name)

        queries.append({
            "id": ann["task_id"],
            "query": ann["Question"],
            "category": ann.get("category", ""),
            "ground_truth_tools": ground_truth_tools,
            "num_steps": int(meta.get("Number of steps", 0)),
            "num_tools": int(meta.get("Number of tools", 0)),
        })

    # --- Save ---
    tools_path = out_dir / "tools-raw.json"
    queries_path = out_dir / "queries-raw.json"
    hierarchy_path = out_dir / "hierarchy.json"

    with open(tools_path, "w") as f:
        json.dump(tools, f, indent=2)
    print(f"[save] Tools → {tools_path} ({os.path.getsize(tools_path) / 1e6:.1f} MB)")

    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"[save] Queries → {queries_path} ({os.path.getsize(queries_path) / 1e6:.1f} MB)")

    with open(hierarchy_path, "w") as f:
        json.dump(hierarchy, f, indent=2)
    print(f"[save] Hierarchy → {hierarchy_path}")

    # --- Summary ---
    print(f"\n=== LiveMCPBench Summary ===")
    print(f"Categories: {len(hierarchy['categories'])}")
    print(f"Servers:    {len(hierarchy['servers'])}")
    print(f"Tools:      {len(tools)}")
    print(f"Queries:    {len(queries)}")
    print(f"\nHierarchy (3 levels):")
    for cat, servers in sorted(hierarchy["categories"].items()):
        n_tools = sum(len(hierarchy["servers"].get(
            # Find the srv_key for each server display name
            next((t["server_name"] for t in tools if t["server_display_name"] == s), ""),
            []
        )) for s in servers)
        print(f"  {cat}: {len(servers)} servers, {n_tools} tools")

    print(f"\nNext: deno run --allow-all --config deno.json lib/shgat-tf/benchmark/scripts/embed-livemcp.ts")


if __name__ == "__main__":
    main()
