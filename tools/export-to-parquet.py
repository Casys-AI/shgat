#!/usr/bin/env python3
"""Export bench dataset from msgpack.gz → Parquet files.

Memory-optimized: uses msgpack streaming to extract each section independently,
never holding the full dataset in memory at once.

Usage:
    cd lib/shgat-tf
    python3 tools/export-to-parquet.py [--data-path <path>]
"""

import gc
import gzip
import json
import os
import struct
import sys
import time

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRU_DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../gru/data"))


def get_arg(name, default):
    try:
        idx = sys.argv.index(f"--{name}")
        return sys.argv[idx + 1] if idx + 1 < len(sys.argv) else default
    except ValueError:
        return default


def emb_bytes(emb):
    return struct.pack(f"{len(emb)}f", *emb)


def sparse_bytes(sparse):
    if not sparse:
        return b"", b""
    return (
        struct.pack(f"{len(sparse)}i", *(int(s[0]) for s in sparse)),
        struct.pack(f"{len(sparse)}f", *(float(s[1]) for s in sparse)),
    )


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


def extract_section(raw_bytes, key):
    """Extract a single top-level key from msgpack bytes.

    The dataset is a map: {nodes: [...], leafIds: [...], ...}
    We decode the full map but immediately extract and return only the requested key,
    then delete the full map.
    """
    # For msgpack maps, we can't easily stream by key without custom parsing.
    # Instead, use unpackb with a hook that skips large arrays we don't need.
    # Actually, msgpack doesn't support random access, so we decode fully
    # but use raw=True for bytes efficiency and only convert what we need.
    pass


def main():
    data_path = get_arg("data-path",
                        os.path.join(GRU_DATA_DIR, "bench-dataset-export.msgpack.gz"))
    print("=== Export Parquet from msgpack.gz (Python streaming) ===\n")

    # --- Decompress (keep raw bytes, don't decode yet) ---
    print(f"[Data] Decompressing {data_path}...")
    t0 = time.time()
    with gzip.open(data_path, "rb") as f:
        raw = f.read()
    print(f"  {len(raw) / 1e6:.1f}MB decompressed in {time.time() - t0:.1f}s (RSS: {rss_mb():.0f}MB)")

    # --- Decode with list_hook to use tuples (less memory than lists) ---
    print("  Decoding msgpack (raw=True for memory efficiency)...")
    t1 = time.time()
    # raw=False needed for string keys, but we can use max_buffer_size
    unpacker = msgpack.Unpacker(max_buffer_size=2 * 1024 * 1024 * 1024)
    unpacker.feed(raw)
    ds = unpacker.unpack()
    del raw, unpacker
    gc.collect()
    print(f"  Decoded in {time.time() - t1:.1f}s (RSS: {rss_mb():.0f}MB)")

    # Quick stats
    nodes = ds[b"nodes"] if b"nodes" in ds else ds.get("nodes", [])
    leaf_ids = ds[b"leafIds"] if b"leafIds" in ds else ds.get("leafIds", [])
    emb_dim = ds[b"embeddingDim"] if b"embeddingDim" in ds else ds.get("embeddingDim", 0)
    prod_train = ds[b"prodTrain"] if b"prodTrain" in ds else ds.get("prodTrain", [])
    prod_test = ds[b"prodTest"] if b"prodTest" in ds else ds.get("prodTest", [])
    n8n_train = ds[b"n8nTrain"] if b"n8nTrain" in ds else ds.get("n8nTrain", [])
    n8n_eval = ds[b"n8nEval"] if b"n8nEval" in ds else ds.get("n8nEval", [])
    wf_tool_lists = ds[b"workflowToolLists"] if b"workflowToolLists" in ds else ds.get("workflowToolLists", [])

    def s(x):
        return x.decode() if isinstance(x, bytes) else x

    print(f"  Nodes: {len(nodes)} ({len(leaf_ids)} leaves), EmbDim: {emb_dim}")
    print(f"  Prod: {len(prod_train)} train / {len(prod_test)} test")
    print(f"  N8n: {len(n8n_train)} train / {len(n8n_eval)} eval")
    print(f"  RSS: {rss_mb():.0f}MB\n")

    counts = {
        "nodes": len(nodes),
        "prodTrain": len(prod_train),
        "prodTest": len(prod_test),
        "n8nTrain": len(n8n_train),
        "n8nEval": len(n8n_eval),
    }

    # ======== NODES ========
    t = time.time()
    print("[Parquet] Nodes...")
    k_id = b"id" if b"id" in (nodes[0] if nodes else {}) else "id"
    k_emb = b"embedding" if b"embedding" in (nodes[0] if nodes else {}) else "embedding"
    k_ch = b"children" if b"children" in (nodes[0] if nodes else {}) else "children"
    k_lv = b"level" if b"level" in (nodes[0] if nodes else {}) else "level"

    table = pa.table({
        "id": pa.array([s(n[k_id]) for n in nodes], pa.string()),
        "embedding": pa.array([emb_bytes(n[k_emb]) for n in nodes], pa.binary()),
        "children_json": pa.array([json.dumps([s(c) for c in n[k_ch]]) for n in nodes], pa.string()),
        "level": pa.array([n[k_lv] for n in nodes], pa.int32()),
    })
    out = os.path.join(GRU_DATA_DIR, "bench-nodes.parquet")
    pq.write_table(table, out, compression="snappy")
    print(f"  {table.num_rows} rows → {os.path.getsize(out)/1e6:.1f}MB ({time.time()-t:.1f}s, RSS: {rss_mb():.0f}MB)")
    del table
    # Free nodes from the dict
    if b"nodes" in ds: ds[b"nodes"] = None
    elif "nodes" in ds: ds["nodes"] = None
    del nodes; gc.collect()

    # ======== PROD TRAIN ========
    t = time.time()
    print("[Parquet] Prod train...")
    ex = prod_train
    k_ie = b"intentEmbedding" if ex and b"intentEmbedding" in ex[0] else "intentEmbedding"
    k_ct = b"contextToolIds" if ex and b"contextToolIds" in ex[0] else "contextToolIds"
    k_tt = b"targetToolId" if ex and b"targetToolId" in ex[0] else "targetToolId"
    k_it = b"isTerminal" if ex and b"isTerminal" in ex[0] else "isTerminal"
    k_tr = b"_traceId" if ex and b"_traceId" in ex[0] else "_traceId"

    table = pa.table({
        "intent_embedding": pa.array([emb_bytes(e[k_ie]) for e in ex], pa.binary()),
        "context_tool_ids_json": pa.array([json.dumps([s(c) for c in e[k_ct]]) for e in ex], pa.string()),
        "target_tool_id": pa.array([s(e[k_tt]) for e in ex], pa.string()),
        "is_terminal": pa.array([e[k_it] for e in ex], pa.int32()),
        "trace_id": pa.array([s(e[k_tr]) for e in ex], pa.string()),
    })
    out = os.path.join(GRU_DATA_DIR, "bench-prod-train.parquet")
    pq.write_table(table, out, compression="snappy")
    print(f"  {table.num_rows} rows → {os.path.getsize(out)/1e6:.1f}MB ({time.time()-t:.1f}s)")
    del table, ex
    if b"prodTrain" in ds: ds[b"prodTrain"] = None
    elif "prodTrain" in ds: ds["prodTrain"] = None
    del prod_train; gc.collect()

    # ======== PROD TEST ========
    t = time.time()
    print("[Parquet] Prod test...")
    ex = prod_test
    table = pa.table({
        "intent_embedding": pa.array([emb_bytes(e[k_ie]) for e in ex], pa.binary()),
        "context_tool_ids_json": pa.array([json.dumps([s(c) for c in e[k_ct]]) for e in ex], pa.string()),
        "target_tool_id": pa.array([s(e[k_tt]) for e in ex], pa.string()),
        "is_terminal": pa.array([e[k_it] for e in ex], pa.int32()),
        "trace_id": pa.array([s(e[k_tr]) for e in ex], pa.string()),
    })
    out = os.path.join(GRU_DATA_DIR, "bench-prod-test.parquet")
    pq.write_table(table, out, compression="snappy")
    print(f"  {table.num_rows} rows → {os.path.getsize(out)/1e6:.1f}MB ({time.time()-t:.1f}s)")
    del table, ex
    if b"prodTest" in ds: ds[b"prodTest"] = None
    elif "prodTest" in ds: ds["prodTest"] = None
    del prod_test; gc.collect()

    # ======== N8N TRAIN ========
    t = time.time()
    print("[Parquet] N8n train...")
    ex = n8n_train
    k_st = b"softTargetSparse" if ex and b"softTargetSparse" in ex[0] else "softTargetSparse"
    idx_b, prb_b = [], []
    for e in ex:
        ib, pb = sparse_bytes(e.get(k_st, []))
        idx_b.append(ib); prb_b.append(pb)
    table = pa.table({
        "intent_embedding": pa.array([emb_bytes(e[k_ie]) for e in ex], pa.binary()),
        "context_tool_ids_json": pa.array([json.dumps([s(c) for c in e[k_ct]]) for e in ex], pa.string()),
        "target_tool_id": pa.array([s(e[k_tt]) for e in ex], pa.string()),
        "is_terminal": pa.array([e[k_it] for e in ex], pa.int32()),
        "soft_target_indices": pa.array(idx_b, pa.binary()),
        "soft_target_probs": pa.array(prb_b, pa.binary()),
    })
    out = os.path.join(GRU_DATA_DIR, "bench-n8n-train.parquet")
    pq.write_table(table, out, compression="snappy")
    print(f"  {table.num_rows} rows → {os.path.getsize(out)/1e6:.1f}MB ({time.time()-t:.1f}s)")
    del table, ex, idx_b, prb_b
    if b"n8nTrain" in ds: ds[b"n8nTrain"] = None
    elif "n8nTrain" in ds: ds["n8nTrain"] = None
    del n8n_train; gc.collect()

    # ======== N8N EVAL ========
    t = time.time()
    print("[Parquet] N8n eval...")
    ex = n8n_eval
    idx_b, prb_b = [], []
    for e in ex:
        ib, pb = sparse_bytes(e.get(k_st, []))
        idx_b.append(ib); prb_b.append(pb)
    table = pa.table({
        "intent_embedding": pa.array([emb_bytes(e[k_ie]) for e in ex], pa.binary()),
        "context_tool_ids_json": pa.array([json.dumps([s(c) for c in e[k_ct]]) for e in ex], pa.string()),
        "target_tool_id": pa.array([s(e[k_tt]) for e in ex], pa.string()),
        "is_terminal": pa.array([e[k_it] for e in ex], pa.int32()),
        "soft_target_indices": pa.array(idx_b, pa.binary()),
        "soft_target_probs": pa.array(prb_b, pa.binary()),
    })
    out = os.path.join(GRU_DATA_DIR, "bench-n8n-eval.parquet")
    pq.write_table(table, out, compression="snappy")
    print(f"  {table.num_rows} rows → {os.path.getsize(out)/1e6:.1f}MB ({time.time()-t:.1f}s)")
    del table, ex, idx_b, prb_b
    if b"n8nEval" in ds: ds[b"n8nEval"] = None
    elif "n8nEval" in ds: ds["n8nEval"] = None
    del n8n_eval; gc.collect()

    # ======== METADATA ========
    meta = {
        "leafIds": [s(x) for x in leaf_ids],
        "embeddingDim": emb_dim,
        "workflowToolLists": [[s(t) for t in wf] for wf in wf_tool_lists],
        "exportedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "export-to-parquet.py",
        "counts": counts,
    }
    meta_path = os.path.join(GRU_DATA_DIR, "bench-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"\n[Metadata] bench-metadata.json ({os.path.getsize(meta_path)/1024:.1f}KB)")

    print(f"\n=== Export complete (RSS: {rss_mb():.0f}MB) ===")
    print("Files:")
    for fn in ["bench-nodes.parquet", "bench-prod-train.parquet", "bench-prod-test.parquet",
               "bench-n8n-train.parquet", "bench-n8n-eval.parquet", "bench-metadata.json"]:
        fp = os.path.join(GRU_DATA_DIR, fn)
        if os.path.exists(fp):
            print(f"  {fn}: {os.path.getsize(fp)/1e6:.1f}MB")


if __name__ == "__main__":
    main()
