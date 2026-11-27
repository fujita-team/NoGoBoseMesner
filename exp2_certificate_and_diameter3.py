"""
Experiment 2 -- beyond one SRG pair.

(A) Exact algebraic certificate (no learning). For three pairs of non-isomorphic
    DRGs with equal intersection arrays
        Shrikhande vs Rook(4x4)      SRG(16,6,2,2),   diameter 2
        Chang-1   vs T(8)            SRG(28,12,6,4),  diameter 2
        Doob D(1,1) vs Hamming H(3,4)  {9,6,3;1,2,3},  diameter 3
    and a list of spectral functions f(M), we measure
        spread_G, spread_F : max variation of [f(M)]_uv inside a distance class
                             (0  <=>  f(M) lies in the Bose-Mesner algebra)
        |alpha_G - alpha_F|: max difference of the distance-basis coefficients
    Several f are deliberately NOT holomorphic (step, |.|^0.5, pseudo-inverse),
    which shows that holomorphy is not what drives the result: only the fact that
    f is a function on the finite spectrum does.
    We also report 2-FWL and the K4-count control.

(B) Diameter-3 learning test: Doob vs H(3,4) (64 vertices), a regime with four
    distance classes where the pair-level claims of Thm 3.2 / 3.5 are non-trivial.

Usage: python exp2_certificate_and_diameter3.py [--seeds 3] [--epochs 100] [--skip-train]
"""
import argparse
import json
import os
import time

import numpy as np

from drg_common import (PE_REGISTRY, adj, bm_certificate, chang_graph, doob_graph, fmt_pm,
                        fwl2_distinguishes, hamming_graph, intersection_array, k4_counts,
                        make_split, rook_graph, save_split, shrikhande_graph, train_eval,
                        triangular_graph, wl_pe_distinguishes, write_tables)


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


PAIRS = {
    "Shrikhande / Rook4x4": (shrikhande_graph, rook_graph),
    "Chang-1 / T(8)": (lambda: chang_graph(0), triangular_graph),
    "Doob / H(3,4)": (doob_graph, lambda: hamming_graph(3, 4)),
}


def pinv_fn(w):
    safe = np.where(np.abs(w) > 1e-9, w, 1.0)
    return np.where(np.abs(w) > 1e-9, 1.0 / safe, 0.0)


FUNCS = [  # (label, operator, f, holomorphic?)
    ("exp(-L)  [heat kernel]", "L", lambda w: np.exp(-w), "yes"),
    ("(I+L)^-1 [resolvent]", "L", lambda w: 1 / (1 + w), "yes"),
    ("A^5", "A", lambda w: w ** 5, "yes"),
    ("tanh(3 Lnorm)", "Lnorm", lambda w: np.tanh(3 * w), "yes"),
    ("L^+  [pseudo-inverse]", "L", pinv_fn, "no (at 0)"),
    ("1[lambda < mean]  [low-pass step]", "A", lambda w: (w < w.mean()).astype(float), "no"),
    ("|A|^0.5", "A", lambda w: np.sqrt(np.abs(w)), "no"),
]


def certificate(args):
    rows = []
    for name, (g, f) in PAIRS.items():
        AG, AF = adj(g()), adj(f())
        ia = intersection_array(AG)
        assert ia is not None and ia == intersection_array(AF), name
        t = time.time()
        fwl = fwl2_distinguishes(AG, AF)
        k4 = sorted(set(k4_counts(AG))) != sorted(set(k4_counts(AF)))
        print(f"{name}: array {ia}, 2-FWL separates: {fwl}, K4 separates: {k4} ({time.time()-t:.1f}s)")
        for lab, op, fn, hol in FUNCS:
            sG, sF, dA = bm_certificate(AG, AF, op, fn)
            rows.append([name, lab, hol, f"{sG:.1e}", f"{sF:.1e}", f"{dA:.1e}",
                         "yes" if fwl else "no", "yes" if k4 else "no"])
    write_tables(rows, ["DRG pair", "f(M)", "Holomorphic", "Spread in class (G)",
                        "Spread in class (F)", "max |alpha_G - alpha_F|",
                        "2-FWL separates", "K4 count separates"],
                 "results/exp2_bm_certificate",
                 "Exact Bose-Mesner certificate: f(M) is constant on distance classes and has "
                 "identical coefficients on both graphs (values at machine precision)")


TRAIN_ROWS = [
    ("GIN", "none"),
    ("GIN", "RWSE diag(T^k)"),
    ("GIN", "heat-kernel diag"),
    ("GINE", "RRWP [T^k]_uv"),
    ("Transformer+RPE", "RRWP [T^k]_uv"),
    ("Transformer+RPE", "resistance+resolvent"),
    ("GIN", "LapPE (eigvecs)"),
    ("GIN", "random features"),
    ("GIN", "K4 counts"),
]


def diameter3(args):
    TAG = "doob_hamming"
    G0, G1 = doob_graph(), hamming_graph(3, 4)
    rng = np.random.default_rng(777)
    train = make_split([G0, G1], args.ntrain, rng)
    test = make_split([G0, G1], args.ntest, rng)
    save_split("data/doob_hamming_train.npz", train)
    save_split("data/doob_hamming_test.npz", test)
    A0, A1 = adj(G0), adj(G1)
    rows = []
    for model, pe in TRAIN_ROWS:
        spec = PE_REGISTRY[pe]
        wl = ("yes" if wl_pe_distinguishes(A0, A1, spec, all_pairs=(model == "Transformer+RPE"))
              else "no") if spec.invariant else "n/a (not invariant)"
        t = time.time()
        ep = args.heavy_epochs if model != "GIN" else args.epochs  # pair-feature models are O(n^2 d)
        res = [cached_run(TAG, f"{model}|{pe}|seed{s}|ep{ep}|n{args.ntrain}-{args.ntest}",
                          lambda s=s: train_eval(model, spec, train, test, seed=s, epochs=ep))
               for s in range(args.seeds)]
        tr, te = [r[0] for r in res], [r[1] for r in res]
        print(f"  {model:16s} {pe:22s} WL:{wl:5s} train {fmt_pm(tr)} test {fmt_pm(te)} ({time.time()-t:.0f}s)")
        rows.append([model, pe, spec.in_scope, wl, ep, fmt_pm(tr), fmt_pm(te)])
    write_tables(rows, ["Model", "PE", "In scope", "1-WL+PE separates", "Epochs", "Train acc (%)",
                  "Test acc (%)"],
                 "results/exp2_doob_hamming",
                 f"Doob vs H(3,4) (diameter 3, 64 vertices): {2*args.ntrain} train / "
                 f"{2*args.ntest} test permuted graphs, {args.seeds} seeds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--heavy-epochs", type=int, default=40,
                    help="epochs for GINE / Transformer rows (64-vertex pair tensors are slow on CPU)")
    ap.add_argument("--ntrain", type=int, default=100)
    ap.add_argument("--ntest", type=int, default=50)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-certificate", action="store_true")
    args = ap.parse_args()
    if not args.skip_certificate:
        certificate(args)
    if not args.skip_train:
        diameter3(args)
