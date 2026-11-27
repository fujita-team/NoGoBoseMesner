"""
Experiment 1 -- Shrikhande vs 4x4 Rook's graph (both SRG(16,6,2,2)).

Train/test sets consist of independently, uniformly permuted copies of the two
graphs; every PE is recomputed from the *permuted* adjacency matrix, so the
experiment also tests permutation equivariance of the PE pipeline.

For every (model, PE) row we report:
  * whether the PE is inside the scope of Thm 3.2 / Thm 3.5,
  * an exact verdict: does (edge-vertex or all-pairs) 1-WL initialised with the
    PE colours separate the two graphs?  (graph-level, no learning involved)
  * train / test accuracy, mean ± std over seeds.

Theory predicts 50.0 ± 0.0 for every in-scope row. Sanity rows:
  * K4 counts (subgraph information outside the BM algebra) -> 100%
  * prism vs K_{3,3} (Lemma 3.1): no PE -> 50%, RWSE -> 100%
Out-of-scope rows (LapPE eigenvectors, random features) are reported as-is.

Usage:  python exp1_shrikhande_vs_rook.py [--seeds 5] [--epochs 150] [--ntrain 100] [--ntest 50]
"""
import argparse
import json
import os
import time

import numpy as np

from drg_common import (PE_REGISTRY, adj, k33_graph, make_split, prism_graph, rook_graph,
                        save_split, shrikhande_graph, train_eval, fmt_pm, wl_pe_distinguishes,
                        write_tables)


CACHE = "results/cache_{tag}.json"


def cached_run(tag, key, fn):
    """Per-row cache so long runs can be interrupted and resumed."""
    path = CACHE.format(tag=tag)
    os.makedirs("results", exist_ok=True)
    cache = json.load(open(path)) if os.path.exists(path) else {}
    if key not in cache:
        cache[key] = fn()
        json.dump(cache, open(path, "w"), indent=1)
    return cache[key]


ROWS = [  # (model, PE)
    ("GIN", "none"),
    ("GIN", "RWSE diag(T^k)"),
    ("GIN", "heat-kernel diag"),
    ("GIN", "probe exp(tA)1"),
    ("GINE", "RRWP [T^k]_uv"),
    ("GINE", "EPNN projectors"),
    ("GINE", "resistance+resolvent"),
    ("Transformer+RPE", "RRWP [T^k]_uv"),
    ("Transformer+RPE", "EPNN projectors"),
    ("Transformer+RPE", "resistance+resolvent"),
    ("GIN", "LapPE (eigvecs)"),
    ("GIN", "random features"),
    ("GIN", "K4 counts"),
]


def run_pair(G0, G1, rows, args, tag, data_dir="data"):
    TAG = tag
    rng = np.random.default_rng(12345)
    train = make_split([G0, G1], args.ntrain, rng)
    test = make_split([G0, G1], args.ntest, rng)
    save_split(f"{data_dir}/{tag}_train.npz", train)
    save_split(f"{data_dir}/{tag}_test.npz", test)
    A0, A1 = adj(G0), adj(G1)
    out = []
    for model, pe in rows:
        spec = PE_REGISTRY[pe]
        if spec.invariant:
            wl = wl_pe_distinguishes(A0, A1, spec, all_pairs=(model == "Transformer+RPE"))
            wl = "yes" if wl else "no"
        else:
            wl = "n/a (not invariant)"
        t = time.time()
        res = [cached_run(TAG, f"{model}|{pe}|seed{s}|ep{args.epochs}|n{args.ntrain}-{args.ntest}",
                          lambda s=s: train_eval(model, spec, train, test, seed=s, epochs=args.epochs))
               for s in range(args.seeds)]
        tr, te = [r[0] for r in res], [r[1] for r in res]
        print(f"  {model:16s} {pe:22s} WL:{wl:5s} train {fmt_pm(tr)}  test {fmt_pm(te)}  ({time.time()-t:.0f}s)")
        out.append([model, pe, spec.in_scope, wl, fmt_pm(tr), fmt_pm(te)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--ntrain", type=int, default=100, help="training graphs per class")
    ap.add_argument("--ntest", type=int, default=50, help="test graphs per class")
    args = ap.parse_args()
    header = ["Model", "PE", "In scope", "1-WL+PE separates", "Train acc (%)", "Test acc (%)"]

    print("== Shrikhande vs 4x4 Rook ==")
    rows = run_pair(shrikhande_graph(), rook_graph(), ROWS, args, "shrikhande_rook")
    write_tables(rows, header, "results/exp1_shrikhande_rook",
                 f"Shrikhande vs 4x4 Rook: {2*args.ntrain} train / {2*args.ntest} test permuted "
                 f"graphs, {args.seeds} seeds")

    print("\n== Sanity check (Lemma 3.1): triangular prism vs K_{3,3} ==")
    rows = run_pair(prism_graph(), k33_graph(), [("GIN", "none"), ("GIN", "RWSE diag(T^k)")],
                    args, "prism_k33")
    write_tables(rows, header, "results/exp1_prism_k33",
                 "Positive control (Lemma 3.1): prism vs K_{3,3}")


if __name__ == "__main__":
    main()
