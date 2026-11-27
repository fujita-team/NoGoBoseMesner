"""
Shared utilities for the no-go experiments on distance-regular graphs (DRGs).

Contents
--------
1. Graph constructions (Shrikhande, 4x4 Rook, Hamming H(d,q), Doob, T(8), Chang,
   prism / K_{3,3}) + intersection-array check.
2. Intrinsic spectral PEs  f(M)  (node-, pair- and probe-level) and a few
   *out-of-scope* encodings (LapPE eigenvectors, random node features, K4 counts).
3. Exact combinatorial verdicts: edge-vertex 1-WL with PE colours, all-pairs
   1-WL with PE colours, and 2-FWL (folklore WL on pairs).
4. Dense PyTorch models: GIN (node PE), GINE (node + edge PE),
   pair-biased graph transformer (node + all-pairs PE).
5. Dataset construction from independently permuted copies, training loop,
   and table writers (Markdown / CSV / LaTeX).

Everything is dense and in float64: the graphs are tiny, and float64 keeps
floating-point noise far below the scale of any genuine structural signal.
"""
from __future__ import annotations

import itertools
import math
import os
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
ROUND = 8  # decimals used to remove floating-point noise from invariant PEs


# =============================================================================
# 1. Graphs
# =============================================================================
def _relabel(G):
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def rook_graph(m: int = 4) -> nx.Graph:
    """K_m [] K_m (the m x m rook's graph); m=4 gives SRG(16,6,2,2)."""
    return _relabel(nx.cartesian_product(nx.complete_graph(m), nx.complete_graph(m)))


def shrikhande_graph() -> nx.Graph:
    """Cayley graph on Z4 x Z4 with connection set {±(0,1), ±(1,0), ±(1,1)}."""
    G = nx.Graph()
    gens = [(0, 1), (0, 3), (1, 0), (3, 0), (1, 1), (3, 3)]
    for i, j in itertools.product(range(4), repeat=2):
        for a, b in gens:
            G.add_edge((i, j), ((i + a) % 4, (j + b) % 4))
    return _relabel(G)


def hamming_graph(d: int, q: int) -> nx.Graph:
    G = nx.complete_graph(q)
    for _ in range(d - 1):
        G = nx.cartesian_product(G, nx.complete_graph(q))
    return _relabel(G)


def doob_graph() -> nx.Graph:
    """Doob graph D(1,1) = Shrikhande [] K4: same intersection array as H(3,4)."""
    return _relabel(nx.cartesian_product(shrikhande_graph(), nx.complete_graph(4)))


def triangular_graph(m: int = 8) -> nx.Graph:
    """T(m) = line graph of K_m; T(8) is SRG(28,12,6,4)."""
    return _relabel(nx.line_graph(nx.complete_graph(m)))


def chang_graph(which: int = 0) -> nx.Graph:
    """Chang graphs: Seidel switching of T(8) w.r.t. the edge set of
    (0) a perfect matching 4K2, (1) C3 + C5, (2) C8 in K8."""
    L = nx.line_graph(nx.complete_graph(8))
    if which == 0:
        S = [(0, 1), (2, 3), (4, 5), (6, 7)]
    elif which == 1:
        S = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (5, 6), (6, 7), (3, 7)]
    else:
        S = [(i, (i + 1) % 8) for i in range(8)]
    S = {tuple(sorted(e)) for e in S}
    nodes = list(L.nodes())
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v in itertools.combinations(nodes, 2):
        adj = L.has_edge(u, v)
        if (u in S) != (v in S):  # Seidel switching toggles S-to-complement pairs
            adj = not adj
        if adj:
            G.add_edge(u, v)
    return _relabel(G)


def prism_graph() -> nx.Graph:
    return _relabel(nx.circular_ladder_graph(3))  # K2 [] K3


def k33_graph() -> nx.Graph:
    return _relabel(nx.complete_bipartite_graph(3, 3))


def adj(G: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=float)


def distance_matrix(A: np.ndarray) -> np.ndarray:
    return np.asarray(nx.floyd_warshall_numpy(nx.from_numpy_array(A)))


def intersection_array(A: np.ndarray):
    """Return ((b0..b_{D-1}), (c1..c_D)) if the graph is distance-regular, else None."""
    Dist = distance_matrix(A)
    if not np.isfinite(Dist).all():
        return None
    D = int(Dist.max())
    b, c = [], []
    for i in range(D + 1):
        mask = Dist == i
        ci = ((Dist == i - 1).astype(float) @ A)[mask] if i > 0 else np.array([0.0])
        bi = ((Dist == i + 1).astype(float) @ A)[mask]
        if np.ptp(ci) > 0 or np.ptp(bi) > 0:
            return None
        c.append(int(ci[0]))
        b.append(int(bi[0]))
    return tuple(b[:D]), tuple(c[1:])


def permute(A: np.ndarray, rng: np.random.Generator):
    p = rng.permutation(A.shape[0])
    return A[np.ix_(p, p)], p


# =============================================================================
# 2. Positional encodings
# =============================================================================
def laplacian(A):
    return np.diag(A.sum(1)) - A


def norm_laplacian(A):
    d = A.sum(1)
    Dm = np.diag(1.0 / np.sqrt(d))
    return np.eye(len(A)) - Dm @ A @ Dm


def rw_matrix(A):
    return A / A.sum(1, keepdims=True)


def spectral_fn(M: np.ndarray, f) -> np.ndarray:
    """f(M) for symmetric M via the eigendecomposition. f only needs to be
    defined on the (finite) spectrum -- holomorphy is never used."""
    w, V = np.linalg.eigh(M)
    return (V * f(w)) @ V.T


def eigenprojectors(M: np.ndarray, tol: float = 1e-8):
    """Distinct eigenvalues and the orthogonal projectors onto their eigenspaces."""
    w, V = np.linalg.eigh(M)
    groups, cur = [], [0]
    for i in range(1, len(w)):
        if w[i] - w[cur[0]] > tol:
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    return [(float(w[g].mean()), V[:, g] @ V[:, g].T) for g in groups]


def k4_counts(A: np.ndarray) -> np.ndarray:
    """Number of 4-cliques through each vertex (a subgraph count, NOT in the BM algebra)."""
    n = len(A)
    cnt = np.zeros(n)
    N = [set(np.nonzero(A[v])[0]) for v in range(n)]
    for v in range(n):
        for a, b, c in itertools.combinations(sorted(N[v]), 3):
            if A[a, b] and A[a, c] and A[b, c]:
                cnt[v] += 1
    return cnt


# Each PE returns (node_feat [n, dn], pair_feat [n, n, dp] or None).
# 'invariant' = permutation-equivariant function of the graph (inside the paper's scope
# when it is a matrix function of a BM operator).
@dataclass
class PESpec:
    name: str
    fn: callable
    in_scope: str          # "yes" (covered by the theorems) / "no" (outside) / "control"
    invariant: bool = True
    resample: bool = False  # re-draw at every forward pass (random node features)


def pe_none(A, rng):
    return np.ones((len(A), 1)), None


def pe_rwse(A, rng, K=12):
    T = rw_matrix(A)
    P, out = np.eye(len(A)), []
    for _ in range(K):
        P = P @ T
        out.append(np.diag(P))
    return np.stack(out, 1), None


def pe_heat_diag(A, rng, ts=(0.1, 0.5, 1.0, 2.0, 4.0)):
    L = laplacian(A)
    return np.stack([np.diag(spectral_fn(L, lambda w, t=t: np.exp(-t * w))) for t in ts], 1), None


def pe_probe(A, rng, ts=(0.5, 1.0, 2.0)):
    """Probe-level f(A) 1 with a non-trivial filter on A (on L it is trivially f(0)1)."""
    one = np.ones(len(A))
    An = A / max(1.0, np.abs(np.linalg.eigvalsh(A)).max())
    return np.stack([spectral_fn(An, lambda w, t=t: np.exp(t * w)) @ one for t in ts], 1), None


def pe_rrwp(A, rng, K=8):
    """RRWP (GRIT): pair channels [T^k]_{uv}, k = 0..K-1; node channels = diagonal."""
    T = rw_matrix(A)
    P, mats = np.eye(len(A)), []
    for _ in range(K):
        mats.append(P)
        P = P @ T
    pair = np.stack(mats, -1)
    return np.stack([np.diag(m) for m in mats], 1), pair


def pe_epnn(A, rng, K=6):
    """EPNN-style: for each distinct Laplacian eigenvalue (sorted, padded to K)
    the pair channels (lambda, [E_lambda]_{uv}); node channels = diag(E_lambda)."""
    L = laplacian(A)
    proj = eigenprojectors(L)[:K]
    n = len(A)
    pair = np.zeros((n, n, 2 * K))
    node = np.zeros((n, K))
    for k, (lam, E) in enumerate(proj):
        pair[:, :, 2 * k] = lam
        pair[:, :, 2 * k + 1] = E
        node[:, k] = np.diag(E)
    return node, pair


def pe_resistance(A, rng, alphas=(0.5, 1.0, 2.0)):
    """Resistance distance (from L^+) and resolvents (I + a L)^{-1} as pair channels."""
    L = laplacian(A)
    Lp = spectral_fn(L, lambda w: np.where(np.abs(w) > 1e-9, 1.0 / np.where(np.abs(w) > 1e-9, w, 1.0), 0.0))
    d = np.diag(Lp)
    R = d[:, None] + d[None, :] - 2 * Lp
    chans = [R] + [spectral_fn(L, lambda w, a=a: 1.0 / (1.0 + a * w)) for a in alphas]
    pair = np.stack(chans, -1)
    return np.stack([np.diag(c) for c in chans[1:]], 1), pair


def pe_lappe(A, rng, k=4):
    """Laplacian eigenvectors 2..k+1 with random signs (basis-dependent: OUT of scope)."""
    w, V = np.linalg.eigh(laplacian(A))
    X = V[:, 1:k + 1] * rng.choice([-1.0, 1.0], size=k)
    return X, None


def pe_rni(A, rng, k=4):
    """Random node initialisation (re-drawn at every forward pass; OUT of scope)."""
    return rng.uniform(0, 1, size=(len(A), k)), None


def pe_k4(A, rng):
    """Positive control: 4-clique counts (subgraph information outside the BM algebra)."""
    return k4_counts(A)[:, None], None


PE_REGISTRY = {
    "none": PESpec("none", pe_none, "-"),
    "RWSE diag(T^k)": PESpec("RWSE diag(T^k)", pe_rwse, "yes"),
    "heat-kernel diag": PESpec("heat-kernel diag", pe_heat_diag, "yes"),
    "probe exp(tA)1": PESpec("probe exp(tA)1", pe_probe, "yes"),
    "RRWP [T^k]_uv": PESpec("RRWP [T^k]_uv", pe_rrwp, "yes"),
    "EPNN projectors": PESpec("EPNN projectors", pe_epnn, "yes"),
    "resistance+resolvent": PESpec("resistance+resolvent", pe_resistance, "yes"),
    "LapPE (eigvecs)": PESpec("LapPE (eigvecs)", pe_lappe, "no", invariant=False),
    "random features": PESpec("random features", pe_rni, "no", invariant=False, resample=True),
    "K4 counts": PESpec("K4 counts", pe_k4, "control"),
}


def compute_pe(spec: PESpec, A, rng):
    node, pair = spec.fn(A, rng)
    if spec.invariant:
        node = np.round(node, ROUND)
        pair = None if pair is None else np.round(pair, ROUND)
    return node, pair


# =============================================================================
# 3. Exact combinatorial verdicts
# =============================================================================
def _key(x):
    return tuple(np.round(np.atleast_1d(x), ROUND - 2).tolist())


def wl_pe_distinguishes(A1, A2, spec: PESpec, all_pairs: bool = False, max_iter: int = 50) -> bool:
    """Edge-vertex 1-WL (or all-pairs 1-WL if all_pairs) initialised with PE colours,
    run jointly on both graphs with a shared colour dictionary."""
    rng = np.random.default_rng(0)
    data = []
    for A in (A1, A2):
        node, pair = compute_pe(spec, A, rng)
        n = len(A)
        nbr = [list(range(n)) if all_pairs else list(np.nonzero(A[v])[0]) for v in range(n)]
        ecol = {}
        for v in range(n):
            for w in nbr[v]:
                pc = () if pair is None else _key(pair[v, w])
                ecol[v, w] = (int(A[v, w]), int(v == w), pc)
        data.append((n, nbr, ecol, [_key(node[v]) for v in range(n)]))
    cols = [d[3] for d in data]
    palette = {}
    cols = [[palette.setdefault(c, len(palette)) for c in cs] for cs in cols]
    for _ in range(max_iter):
        if Counter(cols[0]) != Counter(cols[1]):
            return True
        new_palette, new_cols = {}, []
        for (n, nbr, ecol, _), cs in zip(data, cols):
            sig = [(cs[v], tuple(sorted((cs[w], ecol[v, w]) for w in nbr[v]))) for v in range(n)]
            new_cols.append([new_palette.setdefault(s, len(new_palette)) for s in sig])
        if len(new_palette) == len(palette):
            return Counter(new_cols[0]) != Counter(new_cols[1])
        palette, cols = new_palette, new_cols
    return Counter(cols[0]) != Counter(cols[1])


def fwl2_distinguishes(A1, A2, max_iter: int = 20) -> bool:
    """2-FWL (= '2-WL' in Babai's convention, as powerful as 3-WL in the ML convention):
    colour of (u,v) <- (colour, multiset_w (c(u,w), c(w,v)))."""
    def init(A):
        return np.where(np.eye(len(A), dtype=bool), 2, A.astype(np.int64))
    C = [init(A1), init(A2)]
    ncol = 3
    for _ in range(max_iter):
        if Counter(C[0].ravel().tolist()) != Counter(C[1].ravel().tolist()):
            return True
        K = ncol
        sigs = []
        for Cg in C:
            codes = Cg[:, None, :] * K + Cg.T[None, :, :]   # [u, v, w] -> (c(u,w), c(w,v))
            codes.sort(axis=2)
            sigs.append(np.concatenate([Cg[:, :, None], codes], 2).reshape(-1, codes.shape[2] + 1))
        allsig = np.concatenate(sigs, 0)
        _, inv = np.unique(allsig, axis=0, return_inverse=True)
        inv = inv.ravel()
        m = sigs[0].shape[0]
        newC = [inv[:m].reshape(C[0].shape), inv[m:].reshape(C[1].shape)]
        new_ncol = int(inv.max()) + 1
        if new_ncol == ncol:
            return Counter(newC[0].ravel().tolist()) != Counter(newC[1].ravel().tolist())
        C, ncol = newC, new_ncol
    return Counter(C[0].ravel().tolist()) != Counter(C[1].ravel().tolist())


def bm_certificate(AG, AF, op: str, f):
    """Check f(M) in span{A_0..A_D} for both graphs and compare coefficients.
    Returns (max within-class spread on G, on F, max |alpha_G - alpha_F|)."""
    res = []
    for A in (AG, AF):
        M = {"A": A, "L": laplacian(A), "Lnorm": norm_laplacian(A)}[op]
        Fm = spectral_fn(M, f)
        Dist = distance_matrix(A)
        D = int(Dist.max())
        alpha, spread = [], 0.0
        for i in range(D + 1):
            vals = Fm[Dist == i]
            alpha.append(vals.mean())
            spread = max(spread, float(np.ptp(vals)))
        res.append((np.array(alpha), spread))
    return res[0][1], res[1][1], float(np.abs(res[0][0] - res[1][0]).max())


# =============================================================================
# 4. Models (dense, batched)
# =============================================================================
def mlp(din, dh, dout):
    return nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Linear(dh, dout))


class GIN(nn.Module):
    """GIN on node PEs. With use_edge=True it is GINE: messages relu(h_w + W e_vw)
    with e_vw the pair PE restricted to edges (edge-vertex coloured MPNN)."""

    def __init__(self, dn, dp, hid=64, layers=3, use_edge=False):
        super().__init__()
        self.use_edge = use_edge
        self.inp = nn.Linear(dn, hid)
        self.convs = nn.ModuleList([mlp(hid, hid, hid) for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))
        if use_edge:
            self.edge = nn.ModuleList([nn.Linear(dp, hid) for _ in range(layers)])
        self.out = mlp(hid, hid, 2)

    def forward(self, X, A, P):
        h = self.inp(X)
        for l, conv in enumerate(self.convs):
            if self.use_edge:
                m = (A[..., None] * F.relu(h[:, None, :, :] + self.edge[l](P))).sum(2)
            else:
                m = A @ h
            h = F.relu(conv((1 + self.eps[l]) * h + m))
        return self.out(h.mean(1))


class PairTransformer(nn.Module):
    """Graph transformer with all-pairs attention and an additive pair-PE bias
    (Graphormer/GRIT style). The adjacency is always included as a pair channel."""

    def __init__(self, dn, dp, hid=64, layers=3, heads=4):
        super().__init__()
        self.h, self.dk = heads, hid // heads
        self.inp = nn.Linear(dn, hid)
        self.qkv = nn.ModuleList([nn.Linear(hid, 3 * hid) for _ in range(layers)])
        self.bias = nn.ModuleList([mlp(dp + 1, hid, heads) for _ in range(layers)])
        self.ff = nn.ModuleList([mlp(hid, 2 * hid, hid) for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])
        self.out = mlp(hid, hid, 2)

    def forward(self, X, A, P):
        B, n, _ = X.shape
        h = self.inp(X)
        Pa = torch.cat([P, A[..., None]], -1)
        for l in range(len(self.qkv)):
            q, k, v = self.qkv[l](self.n1[l](h)).view(B, n, 3, self.h, self.dk).unbind(2)
            s = torch.einsum("bvhd,bwhd->bhvw", q, k) / math.sqrt(self.dk)
            s = s + self.bias[l](Pa).permute(0, 3, 1, 2)
            att = s.softmax(-1)
            h = h + torch.einsum("bhvw,bwhd->bvhd", att, v).reshape(B, n, -1)
            h = h + self.ff[l](self.n2[l](h))
        return self.out(h.mean(1))


MODELS = {"GIN": dict(cls=GIN, use_edge=False),
          "GINE": dict(cls=GIN, use_edge=True),
          "Transformer+RPE": dict(cls=PairTransformer)}


# =============================================================================
# 5. Data, training, tables
# =============================================================================
@dataclass
class Split:
    A: np.ndarray            # [N, n, n] permuted adjacency matrices
    y: np.ndarray            # [N] labels
    perm: np.ndarray         # [N, n] permutation applied to the canonical graph


def make_split(graphs, n_per_class: int, rng) -> Split:
    """Independently permuted copies of each canonical graph (balanced)."""
    As, ys, ps = [], [], []
    for label, G in enumerate(graphs):
        A0 = adj(G) if isinstance(G, nx.Graph) else G
        for _ in range(n_per_class):
            Ap, p = permute(A0, rng)
            As.append(Ap); ys.append(label); ps.append(p)
    idx = rng.permutation(len(As))
    return Split(np.stack(As)[idx], np.array(ys)[idx], np.stack(ps)[idx])


def make_split_from_samplers(samplers, n_per_class: int, rng) -> Split:
    """Like make_split, but each class is a callable rng -> adjacency (e.g. random perturbations)."""
    As, ys, ps = [], [], []
    for label, s in enumerate(samplers):
        for _ in range(n_per_class):
            Ap, p = permute(s(rng), rng)
            As.append(Ap); ys.append(label); ps.append(p)
    idx = rng.permutation(len(As))
    return Split(np.stack(As)[idx], np.array(ys)[idx], np.stack(ps)[idx])


def save_split(path, split: Split):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, A=split.A.astype(np.int8), y=split.y, perm=split.perm)


def featurize(split: Split, spec: PESpec, rng):
    nodes, pairs = [], []
    for A in split.A:
        nd, pr = compute_pe(spec, A, rng)
        nodes.append(nd)
        pairs.append(np.zeros((len(A), len(A), 1)) if pr is None else pr)
    return np.stack(nodes), np.stack(pairs)


class Standardizer:
    """Per-channel standardisation fitted on the training set (std=0 -> left unscaled)."""

    def __init__(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mu = flat.mean(0)
        sd = flat.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)

    def __call__(self, X):
        return (X - self.mu) / self.sd


def train_eval(model_name, spec: PESpec, train: Split, test: Split, seed: int,
               epochs=150, lr=1e-3, bs=32, hid=64, layers=3, standardize=True):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Xtr, Ptr = featurize(train, spec, rng)
    Xte, Pte = featurize(test, spec, rng)
    if standardize and not spec.resample:
        sx, sp = Standardizer(Xtr), Standardizer(Ptr)
        Xtr, Xte, Ptr, Pte = sx(Xtr), sx(Xte), sp(Ptr), sp(Pte)
    cfg = MODELS[model_name]
    kw = {"use_edge": cfg["use_edge"]} if "use_edge" in cfg else {}
    model = cfg["cls"](Xtr.shape[-1], Ptr.shape[-1], hid=hid, layers=layers, **kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    Atr, Ate = T(train.A.astype(float)), T(test.A.astype(float))
    Xtr, Ptr, Xte, Pte = map(T, (Xtr, Ptr, Xte, Pte))
    ytr, yte = T(train.y).long(), T(test.y).long()

    def fresh_x(Aset, X):
        if not spec.resample:
            return X
        return T(np.stack([spec.fn(A, rng)[0] for A in Aset]))

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(ytr))
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            Xb = fresh_x(train.A[b.numpy()], Xtr[b])
            loss = F.cross_entropy(model(Xb, Atr[b], Ptr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def acc(Aset, A, X, P, y):
        model.eval()
        pred = model(fresh_x(Aset, X), A, P).argmax(1)
        return (pred == y).double().mean().item()

    return acc(train.A, Atr, Xtr, Ptr, ytr), acc(test.A, Ate, Xte, Pte, yte)


def fmt_pm(xs):
    xs = 100 * np.asarray(xs)
    return f"{xs.mean():.1f} ± {xs.std():.1f}"


def write_tables(rows, header, stem, caption):
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    with open(stem + ".md", "w") as f:
        f.write(f"**{caption}**\n\n" + "\n".join(md) + "\n")
    with open(stem + ".csv", "w") as f:
        f.write(",".join(header) + "\n")
        f.writelines(",".join(f'"{c}"' for c in r) + "\n" for r in rows)
    esc = lambda s: str(s).replace("±", r"$\pm$").replace("%", r"\%").replace("_", r"\_").replace("^", r"\^{}")
    tex = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}",
           r"\begin{tabular}{" + "l" * len(header) + "}", r"\hline",
           " & ".join(r"\textbf{" + esc(h) + "}" for h in header) + r" \\", r"\hline"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(stem + ".tex", "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))
