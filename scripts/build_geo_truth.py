#!/usr/bin/env python
"""One Truth-layout Parquet per generator from the GDML-vertexed campaign shards delivered by the generator side
(data/geo_2026-11/HANDOFF.md: 8 campaigns per generator, species x current, one shard per grid job, with `gen_weight`
normalised per nuclide to POT_gen = 1.998e20). Shards are concatenated in campaign and job order, nothing is re-vertexed
or rescaled; row groups of --rows events so that scripts/surrogate_to_ntuple.py --range reads one slice at a time.
Usage: scripts/build_geo_truth.py nuwro|gibuu [--base data/geo_2026-11] [--rows 200000]"""
import argparse, glob, json, pathlib
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("generator", choices=["nuwro", "gibuu"]); ap.add_argument("--base", default="data/geo_2026-11"); ap.add_argument("--rows", type=int, default=200000); a = ap.parse_args()
base = pathlib.Path(a.base); camps = sorted(p for p in (base / "data").glob(f"{a.generator}_geo_*_2026-11") if (p / "truth").is_dir())
out = base / f"{a.generator}_geo_2026-11.truth.parquet"; writer = None; buf = []; nbuf = 0; summary = {}; total = 0; wsum = 0.0
def flush():
    global buf, nbuf, writer
    if not buf: return
    t = pa.concat_tables(buf); buf = []; nbuf = 0
    if writer is None: writer = pq.ParquetWriter(out, t.schema, compression="zstd")
    writer.write_table(t, row_group_size=a.rows)
for c in camps:
    man = json.load(open(c / "truth" / "truth_manifest.json")); shards = sorted((c / "truth").glob("*.truth.parquet")); n = 0; w = 0.0
    assert len(shards) == len(man["shards"]), f"{c.name}: {len(shards)} shards on disk, {len(man['shards'])} in the manifest"
    for s in shards:
        t = pq.read_table(s); n += t.num_rows; w += float(np.sum(t["gen_weight"].to_numpy())); buf.append(t); nbuf += t.num_rows
        if nbuf >= a.rows: flush()
    mn = sum(v["events"] for v in man["shards"].values()); mw = sum(v["sum_gen_weight"] for v in man["shards"].values())
    assert n == mn and abs(w - mw) < 1e-3 * max(1.0, mw), f"{c.name}: {n} events / {w:.1f} weight on disk, manifest {mn} / {mw:.1f}"
    summary[c.name] = {"shards": len(shards), "events": n, "sum_gen_weight": w, "pot_gen": man.get("pot_gen")}; total += n; wsum += w
    print(f"{c.name}: {len(shards)} shards, {n:,} events, sum gen_weight {w:,.1f} (manifest ok)", flush=True)
flush(); writer.close()
json.dump({"generator": a.generator, "campaigns": summary, "events": total, "sum_gen_weight": wsum, "rows_per_group": a.rows}, open(base / f"{a.generator}_geo_2026-11.json", "w"), indent=1)
print(f"{out}: {total:,} events, sum gen_weight {wsum:,.1f}, {pq.ParquetFile(out).metadata.num_row_groups} row groups")
