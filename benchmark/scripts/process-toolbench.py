#!/usr/bin/env python3
"""
Process ToolRet raw data into structured ToolBench format with 3-level hierarchy.

Reads: data/tools-raw.json, data/queries-raw.json, data/hierarchy.json
Writes: data/toolbench/tools-raw.json, data/toolbench/queries-raw.json, data/toolbench/hierarchy.json

Hierarchy: Category (49) → Collection (~3100) → API (~13800)

Usage:
    python3 lib/shgat-tf/benchmark/scripts/process-toolbench.py
"""

import json
import os
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
OUT_DIR = DATA_DIR / "toolbench"


def recover_hierarchy(tb_tools):
    """Recover tool collection groupings from API name prefixes."""
    # Parse docs and sort by category+name
    parsed = []
    for t in tb_tools:
        doc = json.loads(t['documentation'])
        parsed.append({
            'id': t['id'],
            'name': doc.get('name', ''),
            'category': doc.get('category_name', 'Unknown'),
            'description': doc.get('description', '') or '',
            'documentation': t['documentation'],
        })

    parsed.sort(key=lambda x: (x['category'], x['name']))

    tool_collections = {}  # coll_key → {category, apis: [id, ...]}
    tool_of_api = {}       # api_id → coll_key

    current_prefix = ""
    current_cat = ""
    current_apis = []

    def flush():
        nonlocal current_prefix, current_cat, current_apis
        if current_prefix and current_apis:
            coll_key = f"{current_cat}/{current_prefix}"
            tool_collections[coll_key] = {
                'category': current_cat,
                'collection_name': current_prefix,
                'apis': list(current_apis),
            }
            for aid in current_apis:
                tool_of_api[aid] = coll_key
        current_apis = []

    for p in parsed:
        name = p['name']
        cat = p['category']

        if cat != current_cat:
            flush()
            current_cat = cat
            current_prefix = name.split('_')[0] if '_' in name else name
            current_apis = [p['id']]
            continue

        # Check if shares prefix with current tool collection
        shared = ""
        for i in range(min(len(current_prefix), len(name))):
            if current_prefix[i] == name[i]:
                shared += name[i]
            else:
                break

        if '_' in shared:
            shared = shared[:shared.rfind('_')]

        if shared and len(shared) >= 3 and shared == current_prefix:
            current_apis.append(p['id'])
        else:
            flush()
            current_prefix = name.split('_')[0] if '_' in name else name
            if len(current_prefix) < 3:
                current_prefix = name
            current_apis = [p['id']]

    flush()
    return tool_collections, tool_of_api


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print("[load] Reading raw tools...")
    all_tools = json.load(open(DATA_DIR / "tools-raw.json"))
    print(f"[load] {len(all_tools)} total tools")

    print("[load] Reading raw queries...")
    all_queries = json.load(open(DATA_DIR / "queries-raw.json"))
    print(f"[load] {len(all_queries)} queries")

    # Filter to ToolBench tools only (have category_name in documentation)
    tb_tools = []
    for t in all_tools:
        doc = json.loads(t['documentation'])
        if doc.get('category_name'):
            tb_tools.append(t)

    print(f"[filter] {len(tb_tools)} ToolBench tools (with category)")

    # Recover hierarchy
    print("[hierarchy] Recovering tool collections from API name prefixes...")
    tool_collections, tool_of_api = recover_hierarchy(tb_tools)

    # Build 3-level hierarchy
    categories = {}  # cat → [coll_key, ...]
    collections = {}  # coll_key → [api_id, ...]
    api_to_collection = {}
    collection_to_category = {}

    for coll_key, coll_data in tool_collections.items():
        cat = coll_data['category']
        categories.setdefault(cat, []).append(coll_key)
        collections[coll_key] = coll_data['apis']
        collection_to_category[coll_key] = cat
        for api_id in coll_data['apis']:
            api_to_collection[api_id] = coll_key

    hierarchy = {
        'categories': categories,
        'collections': collections,
        'api_to_collection': api_to_collection,
        'collection_to_category': collection_to_category,
    }

    # Build tools output (only ToolBench tools with category)
    tb_id_set = set(t['id'] for t in tb_tools)
    tools_out = []
    for t in tb_tools:
        doc = json.loads(t['documentation'])
        cat = doc.get('category_name', 'Unknown')
        api_name = doc.get('name', '')
        description = doc.get('description', '') or ''
        coll_key = api_to_collection.get(t['id'], '')

        # Build text for embedding: category + collection + API description
        coll_name = coll_key.split('/')[-1] if coll_key else ''
        text = f"[Category: {cat}] [Tool: {coll_name}] {api_name}: {description}"

        tools_out.append({
            'id': t['id'],
            'name': api_name,
            'description': description,
            'text': text,
            'category': cat,
            'collection': coll_key,
            'documentation': t['documentation'],
        })

    # Build queries output (only queries that reference ToolBench tools)
    queries_out = []
    for q in all_queries:
        # Keep queries where at least one label references a ToolBench tool
        tb_labels = [l for l in q['labels'] if l['id'] in tb_id_set]
        if tb_labels:
            queries_out.append({
                'id': q['id'],
                'query': q['query'],
                'labels': tb_labels,
                'all_labels': q['labels'],  # keep all for reference
            })

    # Save
    tools_path = OUT_DIR / "tools-raw.json"
    queries_path = OUT_DIR / "queries-raw.json"
    hierarchy_path = OUT_DIR / "hierarchy.json"

    with open(tools_path, "w") as f:
        json.dump(tools_out, f)
    print(f"[save] Tools → {tools_path} ({os.path.getsize(tools_path) / 1e6:.1f} MB)")

    with open(queries_path, "w") as f:
        json.dump(queries_out, f)
    print(f"[save] Queries → {queries_path} ({os.path.getsize(queries_path) / 1e6:.1f} MB)")

    with open(hierarchy_path, "w") as f:
        json.dump(hierarchy, f, indent=2)
    print(f"[save] Hierarchy → {hierarchy_path}")

    # Stats
    non_singleton = sum(1 for v in collections.values() if len(v) > 1)
    sizes = [len(v) for v in collections.values()]

    print(f"\n=== ToolBench Hierarchy ===")
    print(f"Categories:       {len(categories)}")
    print(f"Collections:      {len(collections)} ({non_singleton} with 2+ APIs)")
    print(f"APIs:             {len(tools_out)}")
    print(f"Queries:          {len(queries_out)}")
    print(f"APIs/collection:  min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.1f}")
    print(f"\nHierarchy (3 levels): Category → Collection → API")

    # Per-category stats
    for cat in sorted(categories.keys()):
        colls = categories[cat]
        n_apis = sum(len(collections[c]) for c in colls)
        print(f"  {cat}: {len(colls)} collections, {n_apis} APIs")

    print(f"\nNext: deno run --allow-all --config deno.json lib/shgat-tf/benchmark/scripts/embed-toolbench.ts")


if __name__ == "__main__":
    main()
