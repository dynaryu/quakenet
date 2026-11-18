"""Sampling on the clique tree of a network whose edges carry components in series (the Kahramanmaraş network):
one function per clique and direction.

This is cliques.py extended; cliques.py itself, the tree of one bridge per edge under SA (EMA, Hebei, toy-ca),
is left as it was, and trees.py sends a BN here only when cliques.py cannot take it. The tree is the one in the
app's Clique tree tab. On a network whose edges each carry one bridge it is

    clique (SA_i, SAObs_i), one per edge i    factor  P(SAObs_i | SA_i), the measurement of SA_i
    clique (SA_i, D_i, E_i), one per edge i   factors P(D_i | SA_i) P(E_i | D_i)
    clique (SA, E)                            factor  P(SA): ln SA Gaussian, by site or correlated
    clique (E)                                no factor; the root
    clique (E, S_k), one per system k         factor  f(S_k | E)

Clique (SA_i, SAObs_i) hangs off clique (SA_i, D_i, E_i), sharing SA_i. Evidence ev is
{variable: value}, by the BN's variable names: a state index for D, E and S (EdgeOpen 0 closed,
1 open; SystemState 0 Fail, 1 Survival), a value in g for SAObs. SA is observed only through SAObs.
An observed SAObs_i is entered in clique (SA_i, SAObs_i), an observed D_i or E_i in clique
(SA_i, D_i, E_i), an observed S_k in clique (E, S_k).

An edge may instead carry several components in series: bridges, and ground-failure components, each a chain
PGD -> RoadOpen R read from PGV and PGA at the component (a fault crossing's from no field). Its BN node holds
P(E_i | D_1, ..., R_1, ...) as a composite-state table (Byun & Song 2021; kernels.edge.EdgeOpen): n + 1 disjoint
rows in road order, row k saying components 1..k-1 are open and k is closed, the rest in any state (E_i closed),
and row n + 1 that all are open (E_i open). The tree holds that table as it is, in a clique of its own, and gives
each component its own clique below it:

    clique (X_i, D_i..., R_i..., E_i), the edge clique   factor  P(E_i | D, R)
    clique (SA_j, D_j), one per bridge                   factor  P(D_j | SA_j)
    clique (PGV_j, PGA_j, PGD_j, R_j), one per liquefaction or landslide component
                                                         factors P(PGD_j | PGV_j, PGA_j) P(R_j | PGD_j)
    clique (PGD_j, R_j), one per fault crossing          factors P(PGD_j) P(R_j | PGD_j)

X_i is the field at the edge's components (SA at its bridges, PGV and PGA at its ground-failure ones). Each
component's clique hangs off the edge clique, sharing the field it reads and its own state: (SA_j, D_j) at a
bridge, (PGV_j, PGA_j, R_j) at a liquefaction or landslide component, R_j alone at a fault crossing. A bridge's
measurement clique (SA_j, SAObs_j) hangs off its bridge's clique, sharing SA_j. The edge clique hangs off
clique (SA, E), now (X, E): the hub holds every field the network has (SA, PGV, PGA), each its own Gaussian,
independent of the others. Summing the edge clique over (D, R) needs only each component's probability of its
evidence and of being open, which its clique sends up: the table's rows are disjoint, so the sum is over n + 1
rows, not every combination of states. The table reads a bridge's D_j only through whether it leaves the bridge
open, so going down the edge clique draws that, and the bridge's clique draws D_j given it. An edge of one ground-failure component has a single clique, (PGV_i, PGA_i, PGD_i, R_i, E_i) or
(PGD_i, R_i, E_i), with P(E_i | R_i), and hangs off the hub itself. A network whose edges each carry one bridge
has exactly the tree above.

sample() runs n_chain chains of sweeps, each sweep one pass through the tree with clique (E) as the
root. The leaves' messages are exact functions, built once: obs_up, then edge_up (or series_up, over the
rows of the edge's table) for each edge, and system_up for each system. In each sweep:

    collect     sa_e_up: P(E, evidence | X) at each chain's current field X. Summed over X instead, the
                message has no closed form on a correlated field, so clique (E) conditions on X.
    root        root_absorb: E moves one step by certificate MCMC, given that message and the
                observed systems' certificates; with no system observed, it is drawn from that
                message directly.
    distribute  sa_e_down: the new E; X moves one step from its previous value by elliptical slice
                sampling. Any move that leaves P(X | E, evidence) invariant would do.

The sweeps form a Metropolis-within-Gibbs chain over (X, E). After the burn-in, each kept sweep
is passed on down the tree: edge_down (or series_down, then comp_down), obs_down, system_down. With
nothing observed, every up message is 1 and no chain is needed: one pass down the tree draws
independent rows (_forward).

Every function returns its result and n_ops, the count of its dominant operation (in its docstring);
sample() times each call, so every up and down step has its own cost. What each message delivers is
also declared beside its function (@message, @root), and describe() hands those declarations to the
app, which draws them on the tree.

The certificate MCMC comes from rsr.mcmc (rsr-duco 0.3.1 or later, from PyPI). It is imported where it is
used, so the app can read describe() without it.
"""

from __future__ import annotations

import math
import re
import time

import torch
from src.cancellation import check_cancelled

from src.builder.kernels.hazard import GaussianField
from src.builder.kernels.observation import ObservedIntensity
from . import cliques

# ---- what each message delivers, for the app ----------------------------------------------------------
#
# A function that passes a message is marked @message(from, to, carries, delivers, toolkit): the kinds of
# clique it goes between; what it carries, m_up(separator) or m_down(separator) = a probability over the
# variables the two share, "evidence" meaning the evidence on the sender's side going up, towards the root, and
# on the far side going down; how it is computed or delivered, line by line; and the documentation of what does
# the work. m_up^X or m_down^X names its sender, X: SO (SA, SAObs), Dam (SA, D, E), Sa (SA, E), E (E),
# Sys (E, S), Edge (X_i, D, R, E_i), Br (SA_j, D_j), a bridge's clique on a series edge, and Comp (a ground-failure
# component's clique); the app writes up or down as a subscript and
# X as a superscript. The app shows carries on the tree and the rest in a window. A down message's lines give
# what is sent, then, after a line "---" (a rule in the window), the distribution the receiving clique draws its
# own variables from. The root's own step is marked @root.
#
# The hub's text names the network's fields: "{X}" stands for them ("SA", or "SA, PGV, PGA") and "{H}" for the
# hub's own name ("Sa" on SA alone, else "IM"). A message may also give series=(carries, delivers), its text on
# a network with series edges; describe() fills these in for the network it is asked about.

KINDS = {"sa_obs": "(SA_i, SAObs_i)", "sa_d_e": "(SA_i, D_i, E_i)", "sa_e": "({X}, E)", "e": "(E)", "e_s": "(E, S_k)",
         # an edge of several components in series: its clique, with P(E_i | D, R), and its components' cliques,
         # a bridge's with its damage and a ground-failure component's with its PGD
         "x_dr_e": "(X_i, D_i…, R_i…, E_i)", "sa_d": "(SA_j, D_j)", "gf_r": "(PGV_j, PGA_j, PGD_j, R_j)",
         "fault_r": "(PGD_j, R_j)",
         # an edge whose one component is a ground-failure component: its one clique
         "gf_e": "(PGV_i, PGA_i, PGD_i, R_i, E_i)", "fault_e": "(PGD_i, R_i, E_i)"}
BASE = ["sa_obs", "sa_d_e", "sa_e", "e", "e_s"]          # every network's: the tree of one bridge per edge
SERIES = [k for k in KINDS if k not in BASE]
TOOLKITS = {
    "rsr": ["rsr-mcmc", "https://jieunbyun.github.io/rsr/"],
    "rsr_rules": ["rsr", "https://jieunbyun.github.io/rsr/"],
    "ndtools": ["ndtools", "https://pypi.org/project/ndtools-duco/"],
    "exact": ["exact inference", None],
    "normal": ["PyTorch Normal", "https://pytorch.org/docs/stable/distributions.html#normal"],
    "multinomial": ["torch.multinomial", "https://pytorch.org/docs/stable/generated/torch.multinomial.html"],
    "ess": ["elliptical slice sampling", "https://proceedings.mlr.press/v9/murray10a.html"],
}
_MESSAGES, _ROOT = [], {}


def message(source, target, carries, delivers, toolkit=None, series=None):
    """Declare the decorated function the sender of the message from clique kind source to target (either may
    be a tuple of kinds, one message each). toolkit: a key of TOOLKITS, or a list of them. series: (carries,
    delivers) on a network with series edges, if it differs."""
    def mark(f):
        for s in (source if isinstance(source, tuple) else (source,)):
            for t in (target if isinstance(target, tuple) else (target,)):
                _MESSAGES.append({"function": f.__name__, "from": s, "to": t, "carries": carries,
                                  "delivers": delivers, "toolkit": toolkit, "series": series})
        return f
    return mark


def root(kind, belief, does, toolkit=None, series=None):
    """Declare the decorated function the root's own step, at clique kind kind: the belief it holds, from
    the messages it receives, and how it moves towards it, line by line as a message's."""
    def mark(f):
        _ROOT.update(function=f.__name__, clique=kind, belief=belief, does=does, toolkit=toolkit, series=series)
        return f
    return mark


def describe(fields=("SA",), kinds=()):
    """The message passing as this module does it, for a network with these fields (short names, e.g. "SA",
    "PGV") and these clique kinds beyond BASE (layout()): the root, every message between kinds it has, and the
    toolkits they use. The defaults are a network of one bridge per edge on SA."""
    fields, kinds = list(fields), set(kinds)
    have, series = set(BASE) | kinds, bool(kinds & set(SERIES))
    sub = {"{X}": ", ".join(fields), "{H}": "Sa" if fields == ["SA"] else "IM"}

    def fill(s):
        for k, v in sub.items():
            s = s.replace(k, v)
        return s

    def text(d):
        carries, delivers = d["series"] if series and d["series"] else (d.get("carries"), d.get("delivers"))
        return carries, delivers

    messages = []
    for m in _MESSAGES:
        if m["from"] in have and m["to"] in have:
            carries, delivers = text(m)
            messages.append({"function": m["function"], "from": m["from"], "to": m["to"], "carries": fill(carries),
                             "delivers": [fill(l) for l in delivers], "toolkit": m["toolkit"]})
    belief, does = _ROOT["series"] if series and _ROOT["series"] else (_ROOT["belief"], _ROOT["does"])
    rt = {"function": _ROOT["function"], "clique": _ROOT["clique"], "belief": fill(belief),
          "does": [fill(l) for l in does], "toolkit": _ROOT["toolkit"]}
    return {"root": rt, "messages": messages, "kinds": {k: fill(v) for k, v in KINDS.items() if k in have},
            "toolkits": TOOLKITS}


def is_field(function):
    """Whether a hazard function draws a Gaussian field of intensity (not a ground-failure component's PGD)."""
    return re.match(r"^hazard\.(shakemap|gmm)", function) is not None


def layout(nodes):
    """What describe() needs from a BN's nodes ({"id", "quantity", "function", "parents"}, as compiled.nodes()
    gives them): the fields' short names in order, and the clique kinds beyond BASE its edges call for."""
    by = {n["id"]: n for n in nodes}
    fields = []
    for n in nodes:
        name = re.sub(r"\(.*", "", n["quantity"])
        if is_field(n["function"]) and name not in fields:
            fields.append(name)
    kinds = set()
    for n in nodes:
        if n["function"] != "edge.open":
            continue
        ps = n["parents"]
        if len(ps) == 1 and by[ps[0]]["function"] == "fragility.bridge":
            continue                                                       # clique (SA_i, D_i, E_i)
        suffix = "_r" if len(ps) > 1 else "_e"
        if len(ps) > 1:
            kinds.add("x_dr_e")
        for p in ps:
            if by[p]["function"] == "fragility.bridge":
                kinds.add("sa_d")                                          # one bridge of several: its own clique
            else:
                pgd = [q for q in by[p]["parents"] if not is_field(by[q]["function"])]
                reads = any(by[q]["parents"] for q in pgd)
                kinds.add(("gf" if reads else "fault") + suffix)
    return {"fields": fields, "kinds": sorted(kinds)}


def drawing(nodes, tables=None):
    """The tree for the Clique tree tab, as cliques.drawing() gives it, with series edges: the edge clique
    (X_i, D_i..., R_i..., E_i) holds the edge's own table P(E_i | D, R), and each component's clique hangs off it
    sharing its field and its state: a bridge's (SA_j, D_j), with P(D_j | SA_j), and a ground-failure component's,
    (PGV_j, PGA_j, PGD_j, R_j) or (PGD_j, R_j). An edge of one bridge is clique (SA_i, D_i, E_i); an edge of one
    ground-failure component its one clique. tables: each edge's table (server.edge_tables), whose columns give its
    components in road order."""
    from . import cliques

    by = {n["id"]: n for n in nodes}

    def reach(starts):
        """What a component reads: the variables between it and the field (its PGD), parents first, and the
        field values it reads (none at a fault crossing)."""
        chain, field, todo = [], [], list(starts)
        while todo:
            i = todo.pop(0)
            if cliques._is_field(by[i]):
                field += [i] if i not in field else []
            elif i not in chain:
                chain.append(i)
                todo += by[i]["parents"]
        return chain[::-1], field

    local, comp = [], []
    for n in nodes:
        if cliques._role(n) != "edge":
            continue
        e = n["id"]
        parts = (tables or {}).get(e, {}).get("columns", [e])[1:] or n["parents"]          # road order
        if len(parts) == 1 and by[parts[0]]["function"] == "fragility.bridge":            # (SA_i, D_i, E_i)
            (d,) = parts
            sa = by[d]["parents"]
            local.append({"label": n["row"], "members": sa + [d, e], "separator": sa + [e], "kind": "sa_d_e",
                          "factors": [cliques._given(by, d), cliques._p([e], [d])]})
            continue
        own = []
        for p in parts:
            chain, field = reach(by[p]["parents"])
            kind = "bridge" if by[p]["function"] == "fragility.bridge" else "gf" if field else "fault"
            own.append({"p": p, "chain": chain, "field": field, "kind": kind})
        if len(own) == 1:                                                    # one ground-failure component
            c = own[0]
            members = c["field"] + c["chain"] + [c["p"], e]
            local.append({"label": n["row"], "members": members, "separator": c["field"] + [e], "kind": f"{c['kind']}_e",
                          "factors": [cliques._given(by, v) for v in c["chain"] + [c["p"]]] + [cliques._p([e], [c["p"]])]})
            continue
        x = list(dict.fromkeys(f for c in own for f in c["field"]))           # X_i
        local.append({"label": n["row"], "members": x + parts + [e], "separator": x + [e], "kind": "x_dr_e",
                      "factors": [cliques._p([e], parts)]})                    # the table
        for c in own:                        # a bridge's chain is empty: (SA_j, D_j), sharing both
            comp.append({"label": by[c["p"]]["row"], "members": c["field"] + c["chain"] + [c["p"]],
                         "separator": c["field"] + [c["p"]], "edge": len(local) - 1,
                         "kind": "sa_d" if c["kind"] == "bridge" else f"{c['kind']}_r",
                         "factors": [cliques._given(by, v) for v in c["chain"] + [c["p"]]]})
    return cliques._rest(nodes, local, comp)


# ---- the tree's pieces ----------------------------------------------------------------------------------
#
# An edge is "lone" if it carries one bridge: clique (SA_i, D_i, E_i), with the keys D, SA, Obs, obs, frag,
# to_e and site. Any other edge carries its components in series, in road order (the order of its table's
# rows): bridges (D, read from SA at the bridge) and ground-failure components, each a chain
# PGD -> R: PGD read from PGV and PGA at the component (a fault crossing's reads no field), 0 unless the ground
# fails (evidence on whether it failed is entered on PGD as 1 or 0). Each component's terms in the sum over the
# table's rows are its probability over the states its evidence allows, and over those that keep it open (_component),
# PGD summed out in closed form (kernels.ground: a lognormal PGD through a lognormal roadway curve). x is the
# hazard field's values at every site, in t["field_sites"] order: each edge's components' sites in edge order,
# so a network of lone edges has its field in edge order.


def tree(probs, corr=None):
    """The pieces each clique needs, from tbnpy's probability objects (define_model()[1]).
    The edges follow the system kernels' edge order, which every E vector uses.
    The hazard is every Gaussian field of ln IM (kernels.hazard.GaussianField: a GMM or a ShakeMap): SA at
    the bridges, and PGV and PGA at ground-failure components if the BN has them, independent of each other.
    Only the sites some component reads are kept. corr: the correlation R of ln IM, in the fields' own site
    order, to use instead of the model's; None for the model's own."""
    fields = [p for p in probs.values() if isinstance(p, GaussianField)]
    if not fields:
        raise ValueError("the clique tree needs a hazard field (hazard.gmm or hazard.shakemap)")
    native = [v.name for p in fields for v in p.childs]
    native_mu = torch.cat([p.ln_median for p in fields])
    native_sd = torch.cat([p.sigma for p in fields])
    native_corr = torch.block_diag(*[p.corr for p in fields]) if corr is None else corr
    systems = {k.split("::")[1]: p for k, p in probs.items() if k.startswith("SystemState::")}
    observed = {p.parents[0].name: p for p in probs.values() if isinstance(p, ObservedIntensity)}
    edges = []
    for e in next(iter(systems.values())).edge_ids:
        edge = probs[f"EdgeOpen::{e}"]
        nb = getattr(edge, "n_bridges", len(edge.parents))
        comps = []
        for v in edge.parents[:nb]:
            frag = probs[v.name]
            obs = observed.get(frag.parents[0].name)
            comps.append({"kind": "bridge", "D": v.name, "frag": frag, "SA": frag.parents[0].name,
                          "sites": [frag.parents[0].name], "obs": obs, "Obs": obs and obs.childs[0].name,
                          "open_if": edge.open_if})
        for v in edge.parents[nb:]:
            road = probs[v.name]
            pgd = probs[road.parents[0].name]               # one kernel: whether the ground fails, and how far
            # Evidence on whether the ground failed is entered on PGD itself, as 1 (PGD > 0) or 0 (PGD = 0): "OCC".
            comps.append({"kind": "ground", "R": v.name, "PGD": pgd.childs[0].name, "OCC": pgd.childs[0].name,
                          "road": road, "disp": pgd, "occ": pgd, "sites": [p.name for p in pgd.parents]})
        comps = [comps[j] for j in getattr(edge, "order", range(len(comps)))]          # road order: the table's rows
        c = {"E": edge.childs[0].name, "comps": comps, "lone": False, "D": None, "SA": None, "Obs": None, "obs": None}
        if len(comps) == 1 and comps[0]["kind"] == "bridge":               # clique (SA_i, D_i, E_i)
            b = comps[0]
            c.update(lone=True, D=b["D"], SA=b["SA"], Obs=b["Obs"], obs=b["obs"], frag=b["frag"],
                     to_e=torch.nn.functional.one_hot(edge.open_if.long(), 2).double())   # (K, 2)
        edges.append(c)
    order = []
    for c in edges:
        for comp in c["comps"]:
            order += [s for s in comp["sites"] if s not in order]
    index = {s: i for i, s in enumerate(order)}
    perm = torch.tensor([native.index(s) for s in order], dtype=torch.long)
    for c in edges:
        for comp in c["comps"]:
            comp["idx"] = [index[s] for s in comp["sites"]]
        c["site"] = index[c["SA"]] if c["SA"] else None
    return {"edges": edges, "ln_median": native_mu[perm], "sigma": native_sd[perm],
            "corr": native_corr[perm][:, perm], "systems": systems, "field_sites": order, "sites": order,
            "series": True}                                   # sampled by this module (trees.module)


def field(t):
    """The factor of the hub: ln IM ~ N(ln_median, D R D) over t["field_sites"], with D = diag(sigma)
    and R = corr (the identity if None). Returns the mean (n,) and a square root A (n, n) of the
    covariance, A A' = D R D: its Cholesky factor, or from its eigenvectors when co-located sites make
    it singular."""
    mu, sd = t["ln_median"], t["sigma"]
    R = torch.eye(len(mu), dtype=torch.float64) if t["corr"] is None else t["corr"]
    cov = sd[:, None] * R * sd[None, :]
    try:
        return mu, torch.linalg.cholesky(cov)
    except RuntimeError:
        values, vectors = torch.linalg.eigh(cov)
        return mu, vectors * values.clamp_min(0).sqrt()


class _Fixed:
    """A damage state set by do: the same state whatever SA, in place of the bridge's fragility."""

    def __init__(self, frag, state):
        self.states, self.state = frag.states, state

    def distribution(self, sa):
        return torch.nn.functional.one_hot(torch.full((len(sa),), self.state), len(self.states)).double()


def intervene(t, do):
    """The tree under do: an edge set to a state keeps it whatever its components; a damage state or a
    road state set keeps it whatever the shaking, and its edge follows it. Intensities (SA, PGV, PGA) and
    PGD cannot be set: set, they would cut the field. System states and SAObs set are written into the
    rows by sample()."""
    if any(c["SA"] in do for c in t["edges"] if c["lone"]):
        raise ValueError("SA cannot be set by do")
    if any(s in do for s in t["field_sites"]) or any(comp.get(k) in do for c in t["edges"] for comp in c["comps"]
                                                      for k in ("PGD", "OCC")):
        raise ValueError("ground motion, ground-failure occurrence and ground displacement cannot be set by do")
    edges = []
    for c in t["edges"]:
        c = dict(c)
        comps = []
        for comp in c["comps"]:
            comp = dict(comp)
            if comp["kind"] == "bridge" and comp["D"] in do:
                comp["frag"] = _Fixed(comp["frag"], int(do[comp["D"]]))
            if comp["kind"] == "ground" and comp["R"] in do:
                comp["fixed"] = int(do[comp["R"]])
            comps.append(comp)
        c["comps"] = comps
        if c["lone"]:
            c["frag"] = comps[0]["frag"]
        if c["E"] in do:
            c["do_e"] = int(do[c["E"]])
            if c["lone"]:
                c["to_e"] = torch.nn.functional.one_hot(torch.full((len(c["to_e"]),), int(do[c["E"]])), 2).double()
        edges.append(c)
    return {**t, "edges": edges}


def field_draw(fld, N, generator=None):
    """N independent draws of the intensities (N, n) from field()'s Gaussian (g, or cm/s for PGV)."""
    mean, chol = fld
    return torch.exp(mean + torch.randn(N, len(mean), dtype=torch.float64, generator=generator) @ chol.T)


def _lam(ev, name, k):
    """The evidence on a discrete variable over its k states: 1 at the observed state only, or 1 everywhere."""
    if name not in ev:
        return torch.ones(k, dtype=torch.float64)
    return torch.nn.functional.one_hot(torch.tensor(ev[name]), k).double()


def _ground_failure(comp, x):
    """A ground-failure component's P(PGD > 0): P(the ground fails) where it can then move, 0 where it cannot
    (ln median = -inf, e.g. a slope whose critical acceleration exceeds PGA); and its PGD given PGD > 0 as
    (ln median, sigma), at the field values x (N, n)."""
    k = comp["disp"]
    if comp["idx"]:
        pga = x[:, comp["idx"][1]]
        p = k.probability(x[:, comp["idx"][0]], pga)
        ln_med, sigma = k.given_occurrence(pga)
    else:                                                      # a fault crossing reads no field
        p = k.probability(len(x))
        ln_med, sigma = k.given_occurrence()
    ln_med = torch.as_tensor(ln_med, dtype=torch.float64).expand(len(x))
    return p.double() * torch.isfinite(ln_med), ln_med, float(sigma)


def _ground_weights(comp, x, ev):
    """(no, occurs and open, occurs and closed) (N,) each: the component's chain summed over PGD, with the
    evidence on its occurrence entered."""
    p, ln_med, sigma = _ground_failure(comp, x)
    lo = _lam(ev, comp["OCC"], 2)
    pcg = comp["road"].closed_given(ln_med, sigma)              # P(closed | it occurs)
    return lo[0] * (1 - p), lo[1] * p * (1 - pcg), lo[1] * p * pcg


def _component(comp, x, ev):
    """(total, open) (N,): the component's probability summed over its states allowed by the evidence, and
    over those that keep its edge open."""
    if comp["kind"] == "bridge":
        pmf = comp["frag"].distribution(x[:, comp["idx"][0]]) * _lam(ev, comp["D"], len(comp["open_if"]))
        return pmf.sum(1), (pmf * comp["open_if"]).sum(1)
    no, yes_open, yes_closed = _ground_weights(comp, x, ev)
    lam = _lam(ev, comp["R"], 2)
    if "fixed" in comp:                                          # R set by do: its chain above still varies
        whole = (no + yes_open + yes_closed) * lam[comp["fixed"]]
        return whole, whole * comp["fixed"]
    return lam[0] * yes_closed + lam[1] * (no + yes_open), lam[1] * (no + yes_open)


def _lone_support(c, ev):
    """(2,) whether the evidence on a lone edge's bridge and the edge allows E = closed, E = open."""
    return (_lam(ev, c["D"], len(c["to_e"]))[:, None] * c["to_e"] * _lam(ev, c["E"], 2)).sum(0) > 0


def _allowed(c, ev):
    """(2,) whether the evidence on a series edge's components and the edge allows E = closed, E = open."""
    if "do_e" in c:
        return torch.nn.functional.one_hot(torch.tensor(c["do_e"]), 2).double() > 0
    can_open, can_close = True, False
    for comp in c["comps"]:
        if comp["kind"] == "bridge":
            lam = _lam(ev, comp["D"], len(comp["open_if"])) > 0
            if isinstance(comp["frag"], _Fixed):
                lam = lam & (torch.arange(len(lam)) == comp["frag"].state)
            o, cl = (lam & (comp["open_if"] > 0)).any(), (lam & (comp["open_if"] == 0)).any()
        else:
            lam = _lam(ev, comp["R"], 2) > 0
            if "fixed" in comp:
                lam = lam & (torch.arange(2) == comp["fixed"])
            elif _lam(ev, comp["OCC"], 2)[1] == 0:                 # it did not occur: the road cannot close
                lam = lam & (torch.arange(2) == 1)
            o, cl = bool(lam[1]), bool(lam[0])
        can_open, can_close = can_open and bool(o), can_close or bool(cl)
    return torch.tensor([can_close, can_open]) & (_lam(ev, c["E"], 2) > 0)


# ---- clique (SA_i, SAObs_i), one per bridge with a measurement -------------------------------------------

@message("sa_obs", "sa_d", "m_up^SO(SA) = P(evidence | SA)",
         ["= N(SAObs = x; SA, (ε·SA)²) if SAObs is in the evidence;", "= 1 otherwise"], "normal")
@message("sa_obs", "sa_d_e", "m_up^SO(SA) = P(evidence | SA)",
         ["= N(SAObs = x; SA, (ε·SA)²) if SAObs is in the evidence;", "= 1 otherwise"], "normal")
def obs_up(t, i, ev, j=None):
    """Message to the clique holding edge i's bridges (component j's; every bridge on the edge if None), a function
    of the field values x (N, n): None (the constant 1) if no SAObs among them is observed, else the
    measurements' likelihood l(x) = prod N(o; sa, (epsilon sa)^2) at each observed SAObs = o. l.log is
    log l, finite where l underflows. n_ops = 0."""
    terms = []
    for jj, comp in enumerate(t["edges"][i]["comps"]):
        if (j is None or jj == j) and comp["kind"] == "bridge" and comp["obs"] is not None and comp["Obs"] in ev:
            terms.append((comp["obs"], float(ev[comp["Obs"]]), comp["idx"][0]))
    if not terms:
        return None, 0

    def log(x):
        out = None
        for kernel, o, k in terms:
            term = kernel.distribution(x[:, k]).log_prob(torch.full((len(x),), o, dtype=torch.float64))
            out = term if out is None else out + term
        return out

    def l(x):
        return log(x).exp()

    l.log = log
    return l, 0


@message("sa_d", "sa_obs", "m_down^Br(SA) = P(SA | evidence)",
         ["sent as a sample of SA", "---", "P(SAObs | SA, evidence) = N(SA, (ε·SA)²) if SAObs is not in the evidence;",
          "= δ(SAObs − x) if it is, as x"], "normal")
@message("sa_d_e", "sa_obs", "m_down^Dam(SA) = P(SA | evidence)",
         ["sent as a sample of SA", "---", "P(SAObs | SA, evidence) = N(SA, (ε·SA)²) if SAObs is not in the evidence;",
          "= δ(SAObs − x) if it is, as x"], "normal")
def obs_down(t, i, sa, ev, generator=None, j=None):
    """SAObs of edge i's bridge (component j; the edge's only bridge if None) given each row's SA (N,): a
    draw from the measurement kernel's distribution, by its inverse CDF. An observed SAObs keeps its
    value. Returns SAObs (N,) in g; n_ops = draws."""
    comps = t["edges"][i]["comps"]
    comp = comps[j] if j is not None else next(cc for cc in comps if cc["kind"] == "bridge")
    if comp["Obs"] in ev:
        return torch.full((len(sa),), float(ev[comp["Obs"]]), dtype=torch.float64), 0
    u = torch.rand(len(sa), dtype=torch.float64, generator=generator)
    return comp["obs"].distribution(sa).icdf(u), len(sa)


# ---- clique (SA_i, D_i, E_i), one per lone edge i -----------------------------------------------------

@message("sa_d_e", "sa_e", "m_up^Dam(SA, E) = P(E, evidence | SA)",
         ["= Σ_D P(D | SA) P(E | D) · m_up^SO(SA)   if neither D nor E is in the evidence",
          "= P(D = d | SA) P(E | D = d) · m_up^SO(SA)   if only D is, as d",
          "= Σ_D P(D | SA) P(E = e | D) · m_up^SO(SA) if E = e, 0 otherwise   if only E is, as e",
          "= P(D = d | SA) P(E = e | D = d) · m_up^SO(SA) if E = e, 0 otherwise   if both are"], "exact")
def edge_up(t, i, ev, l=None):
    """Message to the hub from lone edge i, a function of (SA_i, E_i), with l absorbed, the message from
    clique (SA_i, SAObs_i) (None for the constant 1):
        m(sa, e) = l(sa) sum_d P(D_i = d | sa) 1[open(d) = e] lambda_D(d) lambda_E(e).
    Returns m, mapping the field values x (N, n) to an (N, 2) table over E_i, with m.held, m without l: as
    l(sa) is the same for both states of E_i, m.held is the message to clique (E) while the field is held,
    and it never vanishes where l(sa) underflows; and m.log, log m, finite there too. n_ops = 0: nothing
    is evaluated yet."""
    c = t["edges"][i]
    s = c["site"]
    w = _lam(ev, c["D"], len(c["to_e"]))[:, None] * c["to_e"] * _lam(ev, c["E"], 2)          # (K, 2)

    def held(x):
        return c["frag"].distribution(x[:, s]) @ w

    def m(x):
        return held(x) if l is None else held(x) * l(x)[:, None]

    def log(x):
        return held(x).log() if l is None else held(x).log() + l.log(x)[:, None]

    m.held, m.log = held, log
    return m, 0


@message("sa_e", "sa_d_e", "m_down^{H}(SA, E) = P(SA, E | evidence)",
         ["sent as a sample of (SA, E)", "---", "P(D | SA, E, evidence) ∝ P(D | SA) P(E | D) if D is not in the evidence,",
          "over the damage states that give E;", "= δ(D − d) if it is, as d"], "multinomial")
def edge_down(t, i, sa, e, ev, generator=None):
    """D_i of lone edge i given each row's SA_i and E_i (N,): a categorical over the damage states with
    weights P(D_i = d | sa) 1[open(d) = e] lambda_D(d). An observed D_i keeps its value (its only weight).
    Returns D_i (N,) state indices; n_ops = fragility evaluations."""
    c = t["edges"][i]
    w = c["frag"].distribution(sa) * _lam(ev, c["D"], len(c["to_e"])) * c["to_e"][:, e].T      # (N, K)
    return torch.multinomial(w, 1, generator=generator)[:, 0], len(sa)


# ---- a series edge: the edge clique, with P(E_i | D, R), and its ground-failure components' cliques ------------

# Each kind of ground-failure component reads its own field, and a fault crossing none: what its clique shares
# with the edge clique (with E_i in its place on an edge of one component) and what its PGD is drawn given.
_READS = {"gf": "PGV_j, PGA_j", "fault": ""}
_PGD_SUMMED = "PGD summed out in closed form: a lognormal PGD through a lognormal roadway curve"
_PGD_EXACT = "drawn exactly: ln PGD and the road's lognormal capacity are jointly Gaussian"


def _given(x, *rest):
    """' | x, rest' or ' | rest' or '': the conditioning bar of a probability, on what there is."""
    on = ", ".join(v for v in (x, *rest) if v)
    return f" | {on}" if on else ""


def _sep(x, *rest):
    return ", ".join(v for v in (x, *rest) if v)


def _comp_up_text(kind):
    x = _READS[kind]
    return (f"m_up^Comp({_sep(x, 'R_j')}) = P(R_j, evidence_j{_given(x)})",
            [f"= Σ_PGD P(PGD_j{_given(x)}) P(R_j | PGD_j) · λ(evidence_j)", _PGD_SUMMED])


def _comp_down_text(kind):
    x = _READS[kind]
    return (f"m_down^Edge({_sep(x, 'R_j')}) = P({_sep(x, 'R_j')} | evidence)",
            [f"sent as a sample of ({_sep(x, 'R_j')})", "---",
             f"P(PGD_j{_given(x, 'R_j', 'evidence')}) ∝ P(PGD_j{_given(x)}) P(R_j | PGD_j), {_PGD_EXACT}"])


_BRIDGE_UP = ("m_up^Br(SA_j, D_j) = P(D_j, evidence_j | SA_j)",
              ["= P(D_j | SA_j) · m_up^SO(SA_j)   if D_j is not in the evidence",
               "= P(D_j = d | SA_j) · m_up^SO(SA_j) if D_j = d, 0 otherwise   if it is, as d"])
_BRIDGE_DOWN = ("m_down^Edge(SA_j, D_j) = P(SA_j, D_j | evidence)",
                ["sent as a sample of SA_j and of whether bridge j is open,",
                 "all that P(E_i | D, R) reads of D_j", "---",
                 "P(D_j | SA_j, evidence) ∝ P(D_j | SA_j) if D_j is not in the evidence,",
                 "over the damage states that give that state;", "= δ(D_j − d) if it is, as d"])


def _lone_up_text(kind):
    x = _READS[kind].replace("_j", "_i")
    return (f"m_up^Comp({_sep(x, 'E_i')}) = P(E_i, evidence{_given(x)})",
            [f"= P(R_i open, evidence{_given(x)})   if E_i is open",
             f"= P(R_i closed, evidence{_given(x)})   if E_i is closed", _PGD_SUMMED])


def _lone_down_text(kind):
    x = _READS[kind].replace("_j", "_i")
    return (f"m_down^{{H}}({_sep(x, 'E_i')}) = P({_sep(x, 'E_i')} | evidence)",
            [f"sent as a sample of ({_sep(x, 'E_i')})", "---", "R_i = E_i;",
             f"P(PGD_i{_given(x, 'R_i', 'evidence')}) ∝ P(PGD_i{_given(x)}) P(R_i | PGD_i), {_PGD_EXACT}"])


@message("sa_d", "x_dr_e", *_BRIDGE_UP, "exact")
@message("gf_r", "x_dr_e", *_comp_up_text("gf"), "exact")
@message("fault_r", "x_dr_e", *_comp_up_text("fault"), "exact")
def comp_up(t, i, j, ev):
    """Component j of edge i's terms in the sum over the edge's table, as a function of the field values x (N, n):
    (total, open) (N,) each, its probability over the states its evidence allows and over those that keep it
    open, from its clique's message up: at a ground-failure component m(x, R_j) at R_j open (open) and closed
    (total - open); at a bridge m(SA_j, D_j) = P(D_j | SA_j) lambda(D_j) summed over its damage states, all of them
    (total) and those that leave it open (open). A bridge's measurement, constant in D_j, series_up multiplies in
    once for the edge. n_ops = 0: nothing is evaluated yet."""
    comp = t["edges"][i]["comps"][j]
    return (lambda x: _component(comp, x, ev)), 0


@message("gf_e", "sa_e", *_lone_up_text("gf"), "exact")
@message("fault_e", "sa_e", *_lone_up_text("fault"), "exact")
@message("x_dr_e", "sa_e", "m_up^Edge(X_i, E_i) = P(E_i, evidence | X_i)",
         ["= Σ_{D, R} P(E_i | D, R) Π_bridges m_up^Br(SA_j, D_j) Π_ground m_up^Comp(·, R_j)",
          "E_i is open exactly when every component j on the edge is open, so",
          "= Π_j P(j open, evidence_j | X_i)   if E_i is open",
          "= Π_j P(evidence_j | X_i) − Π_j P(j open, evidence_j | X_i)   if E_i is closed",
          "and 0 at the state of E_i that is not the evidence, if E_i is in it"],
         "exact")
def series_up(t, i, ev, l=None):
    """Message to the hub from series edge i's edge clique (from its one component's clique, if it has one),
    a function of the field values x (N, n) and E_i: P(E_i | D, R) times the bridges' P(D_j | SA_j) and the
    ground-failure components' messages, summed over (D, R) by the table's rows (each component's terms from
    comp_up), with l absorbed, the message from the edge's measurements (None for the constant 1):
        m(x, e) = l(x) lambda_E(e) [prod_j total_j(x) - prod_j open_j(x), prod_j open_j(x)][e]
    With E set by do, m(x, e) = l(x) prod_j total_j(x) at that e only. Returns m, mapping x (N, n) to an
    (N, 2) table over E_i, with m.held, m without l (the message to clique (E) while x is held, which never
    vanishes where l(x) underflows), and m.log, log m. n_ops = 0: nothing is evaluated yet."""
    c = t["edges"][i]
    lam_e = _lam(ev, c["E"], 2)
    parts = [comp_up(t, i, j, ev)[0] for j in range(len(c["comps"]))]

    def held(x):
        total = torch.ones(len(x), dtype=torch.float64)
        opened = torch.ones(len(x), dtype=torch.float64)
        for part in parts:
            tot, op = part(x)
            total, opened = total * tot, opened * op
        if "do_e" in c:
            out = torch.zeros(len(x), 2, dtype=torch.float64)
            out[:, c["do_e"]] = total
            return out
        return torch.stack([(total - opened).clamp_min(0), opened], dim=1) * lam_e

    def m(x):
        return held(x) if l is None else held(x) * l(x)[:, None]

    def log(x):
        return held(x).log() if l is None else held(x).log() + l.log(x)[:, None]

    m.held, m.log = held, log
    return m, 0


@message("sa_e", "x_dr_e", "m_down^{H}(X_i, E_i) = P(X_i, E_i | evidence)",
         ["sent as a sample of (X_i, E_i)", "---",
          "the row of P(E_i | D, R) that holds: row n + 1 (all open)   if E_i is open",
          "P(row k | X_i, E_i, evidence) ∝ Π_{j<k} P(j open, evidence_j) · P(k closed, evidence_k) · Π_{j>k} P(evidence_j)   if E_i is closed",
          "then whether each component is open, in its cell of that row: open, closed, or from",
          "P(j open | X_j, evidence) if any; each component's clique draws its own variables given it"],
         "multinomial")
def series_down(t, i, x, e, ev, generator=None):
    """The row of series edge i's table P(E_i | D, R) that holds, given each row's field values x (N, n) and
    E_i (N,), 0-based: k < n says component k is the first closed one, n that all are open. E_i open gives n;
    E_i closed a row k < n with weight prod_{j<k} open_j * closed_k * prod_{j>k} total_j; with E_i set by do,
    the components are drawn from their own law and the row is the one they fall in. The edge clique's own draw
    of (D, R) given (X_i, E_i) is this row, then each component in its cell (comp_down). Returns (the row (N,),
    the components' (total, open) tables) for comp_down; n_ops = component evaluations."""
    c = t["edges"][i]
    comps, n = c["comps"], len(x)
    k = len(comps)
    tables = [_component(comp, x, ev) for comp in comps]
    w_open = torch.stack([op for _, op in tables], dim=1)                       # (N, k)
    w_closed = torch.stack([(tot - op).clamp_min(0) for tot, op in tables], dim=1)
    if "do_e" in c:                                             # E is set: the components follow their own law
        opens = torch.rand(n, k, dtype=torch.float64, generator=generator) * (w_open + w_closed).clamp_min(1e-300) < w_open
        f = torch.where(opens.all(dim=1), k, (~opens).int().argmax(dim=1))
        return (f, (w_open, w_closed)), n * k
    with torch.no_grad():
        lo, lc, lt = (w.clamp_min(0).log() for w in (w_open, w_closed, w_open + w_closed))
        zero = torch.zeros(n, 1, dtype=torch.float64)
        # exclusive sums by shifting, not by subtracting (-inf - -inf would be nan)
        before = torch.cat([zero, torch.cumsum(lo, dim=1)[:, :-1]], dim=1)                     # sum_{j<k} log open_j
        after = torch.cat([torch.flip(torch.cumsum(torch.flip(lt, [1]), dim=1), [1])[:, 1:], zero], dim=1)  # sum_{j>k}
        lrow = before + lc + after                                                  # (N, k)
    closed = ~e.bool()
    feasible = torch.isfinite(lrow).any(dim=1)
    f = torch.full((n,), k, dtype=torch.long)                   # k: no closed component (all open)
    pick = closed & feasible                                    # an infeasible closed edge falls back to all open
    if pick.any():
        lr = lrow[pick]
        f[pick] = torch.multinomial((lr - lr.max(dim=1, keepdim=True).values).exp(), 1, generator=generator)[:, 0]
    return (f, (w_open, w_closed)), n * k


def _lognormal_given(ln_med, sigma, road, closed, generator):
    """PGD (N,) from LogNormal(ln_med, sigma) given whether the road closed (N,) bool, drawn exactly; from
    LogNormal(ln_med, sigma) itself if closed is None; a point mass at exp(ln_med) if sigma = 0.

    The roadway curve is lognormal, P(closed | PGD) = Phi((ln PGD - ln theta) / beta): the road closes when
    ln PGD > ln C, its capacity C ~ LogNormal(ln theta, beta). So D = ln PGD - ln C ~ N(ln_med - ln theta,
    sigma^2 + beta^2) is drawn truncated to D > 0 (closed) or D < 0 (open), from the tail that holds that
    side so a side of tiny probability keeps its precision; then ln PGD | D is Gaussian, mean
    ln_med + sigma^2 / (sigma^2 + beta^2) (D - E D), variance sigma^2 beta^2 / (sigma^2 + beta^2)."""
    ln_med = ln_med.clamp_min(-700)                                            # -inf: none; masked by the caller
    s2 = float(sigma) ** 2
    if s2 <= 0:
        return torch.exp(ln_med)
    if closed is None:
        return torch.exp(ln_med + math.sqrt(s2) * torch.randn(len(ln_med), dtype=torch.float64, generator=generator))
    b2 = road.beta ** 2
    sd = math.sqrt(s2 + b2)
    z0 = (ln_med - math.log(road.median)) / sd                                # E D / sd; P(closed) = Phi(z0)
    side = torch.where(closed, z0, -z0)                                       # Phi(side): the drawn side's probability
    u = torch.rand(len(ln_med), dtype=torch.float64, generator=generator)
    t = torch.special.ndtri((u * torch.special.ndtr(side)).clamp_min(1e-300))  # N(0, 1) truncated to t < side
    d = sd * torch.where(closed, -t, t)                                       # D - E D: > -E D closed, < -E D open
    m = ln_med + s2 / (s2 + b2) * d
    return torch.exp(m + math.sqrt(s2 * b2 / (s2 + b2)) * torch.randn(len(ln_med), dtype=torch.float64,
                                                                       generator=generator))


def _component_down(comp, x, open_, ev, generator):
    """The component's own variables given whether it is open (N,) bool: a bridge's D; a ground-failure
    component's R and PGD. Returns {variable: (N,)}."""
    if comp["kind"] == "bridge":
        pmf = comp["frag"].distribution(x[:, comp["idx"][0]]) * _lam(ev, comp["D"], len(comp["open_if"]))
        mask = torch.where(open_[:, None], comp["open_if"][None, :], 1 - comp["open_if"][None, :])
        return {comp["D"]: torch.multinomial((pmf * mask).clamp_min(0) + 1e-300 * (pmf * mask).sum(1, keepdim=True).eq(0),
                                             1, generator=generator)[:, 0]}
    n = len(x)
    p, ln_med, sigma = _ground_failure(comp, x)
    road, lo = comp["road"], _lam(ev, comp["OCC"], 2)
    moves = torch.isfinite(ln_med)
    pcg = road.closed_given(ln_med, sigma)                                     # P(closed | it occurs)
    u = torch.rand(n, dtype=torch.float64, generator=generator)
    if "fixed" in comp:                       # R set by do: the occurrence and PGD follow their own law
        w_no, w_yes = lo[0] * (1 - p), lo[1] * p
        occurs = u * (w_no + w_yes).clamp_min(1e-300) >= w_no
        pgd_occurs = _lognormal_given(ln_med, sigma, road, None, generator)
    else:
        # open: it does not occur, or occurs and the road stays open; closed: it occurs and closes
        w_no, w_yes_open = lo[0] * (1 - p), lo[1] * p * (1 - pcg)
        occurs = torch.where(open_, u * (w_no + w_yes_open).clamp_min(1e-300) >= w_no, torch.ones(n, dtype=torch.bool))
        pgd_occurs = _lognormal_given(ln_med, sigma, road, ~open_, generator)
    pgd = torch.where(occurs & moves, pgd_occurs, torch.zeros(n, dtype=torch.float64))
    return {comp["R"]: open_.long(), comp["PGD"]: pgd}         # whether it occurred is PGD > 0


@message("sa_e", "gf_e", *_lone_down_text("gf"), "multinomial")
@message("sa_e", "fault_e", *_lone_down_text("fault"), "multinomial")
@message("x_dr_e", "gf_r", *_comp_down_text("gf"), "multinomial")
@message("x_dr_e", "fault_r", *_comp_down_text("fault"), "multinomial")
@message("x_dr_e", "sa_d", *_BRIDGE_DOWN, "multinomial")
def comp_down(t, i, x, f, tables, ev, generator=None):
    """The components of series edge i given each row's field values x (N, n) and its table's row f (N,) from
    series_down, with its tables: in their cells of that row, those before f open, f closed and the rest ("any")
    drawn from their own weights; then the message down to each component's clique, whether it is open, from
    which it draws its own variables (_component_down): a bridge's D_j, a ground-failure component's R_j and PGD.
    Returns {variable: (N,)}; n_ops = component evaluations."""
    comps, n = t["edges"][i]["comps"], len(x)
    w_open, w_closed = tables
    k = len(comps)
    j = torch.arange(k)[None, :]
    rest = torch.rand(n, k, dtype=torch.float64, generator=generator) * (w_open + w_closed).clamp_min(1e-300) < w_open
    opens = torch.where(j < f[:, None], True, torch.where(j == f[:, None], False, rest))
    out = {}
    for jj, comp in enumerate(comps):
        out.update(_component_down(comp, x, opens[:, jj], ev, generator))
    return out, n * k


class _Batch:
    """Every edge's message at once, m(x, e) for all edges i, the same function edge_up and series_up build
    edge by edge: the components evaluated as arrays (bridges by their lognormal curves, ground-failure
    components by the formulas kernels.ground shares) and multiplied into their edges by summing logs. Built
    only on a network with series edges, when every component allows it; otherwise sample() uses the
    per-edge messages."""

    def __init__(self, t, ev):
        edges = t["edges"]
        self.n_edges = len(edges)
        self.lam_e = torch.stack([_lam(ev, c["E"], 2) for c in edges])                       # (n_edges, 2)
        self.do = [(i, c["do_e"]) for i, c in enumerate(edges) if "do_e" in c]
        B = [(i, comp) for i, c in enumerate(edges) for comp in c["comps"] if comp["kind"] == "bridge"]
        G = [(i, comp) for i, c in enumerate(edges) for comp in c["comps"] if comp["kind"] == "ground"]
        self.obs = [(i, comp["obs"], float(ev[comp["Obs"]]), comp["idx"][0]) for i, comp in B
                    if comp["obs"] is not None and comp["Obs"] in ev]
        if any(isinstance(comp["frag"], _Fixed) or getattr(comp["frag"], "curve", None) != "lognormal" for _, comp in B):
            raise NotImplementedError("a damage state set by do, or a curve that is not lognormal")

        def T(v):
            return torch.as_tensor(v, dtype=torch.float64)

        self.b_edge = torch.tensor([i for i, _ in B], dtype=torch.long)
        self.b_site = torch.tensor([comp["idx"][0] for _, comp in B], dtype=torch.long)
        if B:
            self.b_med = torch.stack([comp["frag"].medians for _, comp in B]).log()           # (nb, K-1)
            self.b_beta = torch.stack([comp["frag"].betas for _, comp in B])
            self.b_lam = torch.stack([_lam(ev, comp["D"], len(comp["open_if"])) for _, comp in B])
            self.b_open = torch.stack([comp["open_if"] for _, comp in B])
        kinds = {"LiquefactionPGD": "liq", "LandslidePGD": "ls", "FaultPGD": "fault"}
        groups = {"liq": [], "ls": [], "fault": []}
        for i, comp in G:
            name = type(comp["occ"]).__name__
            if name not in kinds:
                raise NotImplementedError(f"ground-failure kernel {name}")
            groups[kinds[name]].append((i, comp))
        self.groups = []
        for kind, members in groups.items():
            if not members:
                continue
            g = {"kind": kind, "edge": torch.tensor([i for i, _ in members], dtype=torch.long),
                 "lam": torch.stack([_lam(ev, comp["R"], 2) for _, comp in members]),
                 "lam_o": torch.stack([_lam(ev, comp["OCC"], 2) for _, comp in members]),
                 "fixed": torch.tensor([comp.get("fixed", -1) for _, comp in members], dtype=torch.long),
                 "ln_theta": T([math.log(comp["road"].median) for _, comp in members]),
                 "beta": T([comp["road"].beta for _, comp in members])}
            occ = [comp["occ"] for _, comp in members]
            disp = [comp["disp"] for _, comp in members]
            if kind == "fault":
                g["p"] = T([k.q for k in occ])
                g["pcg"] = torch.stack([comp["road"].closed_given(T([comp["disp"].ln_med]), comp["disp"].sigma)[0]
                                        for _, comp in members])
            else:
                g["pgv"] = torch.tensor([comp["idx"][0] for _, comp in members], dtype=torch.long)
                g["pga"] = torch.tensor([comp["idx"][1] for _, comp in members], dtype=torch.long)
                g["length"] = T([k.length_m for k in occ])
                g["x0"] = T([k.x0 for k in occ])
                o, d = occ[0], disp[0]
                if kind == "liq":
                    if any(k.zhu != o.zhu or k.patch_m != o.patch_m or k.msf != o.msf for k in occ):
                        raise NotImplementedError("liquefaction components on different models")
                    # K_delta per component: 0 where the stretch has no free face, so it only settles
                    g.update(pga_t=T([k.pga_t for k in disp]), settle=T([k.settle_m for k in disp]),
                             kdelta=T([k.kdelta for k in disp]), msf=o.msf, patch=o.patch_m, zhu=o.zhu, sigma=0.0)
                else:
                    if any(k.jes != o.jes or k.patch_m != o.patch_m for k in occ) or any(k.jib != d.jib for k in disp):
                        raise NotImplementedError("landslide components on different models")
                    g.update(slope=T([k.slope for k in occ]), ac=T([k.ac for k in disp]), patch=o.patch_m, jes=o.jes,
                             jib=d.jib, sigma=d.jib["sigma_log10"] * math.log(10.0))
            self.groups.append(g)

    def _weights(self, g, x):
        """(no, occurs and open, occurs and closed) (N, n_g) for one group of ground-failure components, the
        evidence on their occurrence entered: _ground_weights for them all at once."""
        from src.builder.kernels import ground

        if g["kind"] == "fault":
            p, pcg = g["p"].expand(len(x), -1), g["pcg"].expand(len(x), -1)
        else:
            pgv, pga = x[:, g["pgv"]], x[:, g["pga"]]
            if g["kind"] == "liq":
                p = ground.liquefaction_probability(pgv, pga, g["x0"], g["msf"], g["length"], g["patch"], g["zhu"])
                ln_med = ground.liquefaction_ln_displacement(pga, g["pga_t"], g["settle"], g["kdelta"])
            else:
                p = ground.landslide_probability(pgv, pga, g["x0"], g["slope"], g["length"], g["patch"], g["jes"])
                ln_med = ground.landslide_ln_displacement(pga, g["ac"], g["jib"])
            p = p * torch.isfinite(ln_med)                        # P(PGD > 0), as _ground_failure
            s = torch.sqrt(g["beta"] ** 2 + g["sigma"] ** 2)
            pf = torch.special.ndtr((ln_med - g["ln_theta"]) / s)
            pcg = torch.where(torch.isinf(ln_med) & (ln_med < 0), torch.zeros_like(pf), pf)
        lo = g["lam_o"]
        return lo[:, 0] * (1 - p), lo[:, 1] * p * (1 - pcg), lo[:, 1] * p * pcg

    def held(self, x):
        """(N, n_edges, 2): every edge's message without the measurements, at the field values x (N, n)."""
        N = len(x)
        log_total = torch.zeros(N, self.n_edges, dtype=torch.float64)
        log_open = torch.zeros(N, self.n_edges, dtype=torch.float64)
        if len(self.b_edge):
            z = (x[:, self.b_site].log()[:, :, None] - self.b_med[None]) / self.b_beta[None]   # (N, nb, K-1)
            exceed = torch.cummin(torch.special.ndtr(z), dim=2).values
            ones, zeros = torch.ones_like(exceed[..., :1]), torch.zeros_like(exceed[..., :1])
            pmf = (torch.cat([ones, exceed], 2) - torch.cat([exceed, zeros], 2)).clamp_min(0) * self.b_lam[None]
            log_total.index_add_(1, self.b_edge, pmf.sum(2).log())
            log_open.index_add_(1, self.b_edge, (pmf * self.b_open[None]).sum(2).log())
        for g in self.groups:
            no, yes_open, yes_closed = self._weights(g, x)
            lam, fixed = g["lam"], g["fixed"]
            total = lam[:, 0] * yes_closed + lam[:, 1] * (no + yes_open)
            opened = lam[:, 1] * (no + yes_open)
            if (fixed >= 0).any():                              # R set by do: its chain above still varies
                whole = (no + yes_open + yes_closed) * lam.gather(1, fixed.clamp_min(0)[:, None])[:, 0]
                total = torch.where(fixed >= 0, whole, total)
                opened = torch.where(fixed >= 0, whole * fixed.clamp_min(0), opened)
            log_total.index_add_(1, g["edge"], total.log())
            log_open.index_add_(1, g["edge"], opened.log())
        total, opened = log_total.exp(), log_open.exp()
        out = torch.stack([(total - opened).clamp_min(0), opened], dim=2) * self.lam_e[None]
        for i, state in self.do:
            out[:, i] = 0.0
            out[:, i, state] = total[:, i]
        return out

    def log(self, x):
        """(N, n_edges, 2): log of every edge's message, the measurements included."""
        out = self.held(x).log()
        for i, kernel, o, k in self.obs:
            out[:, i] = out[:, i] + kernel.distribution(x[:, k]).log_prob(
                torch.full((len(x),), o, dtype=torch.float64))[:, None]
        return out


# ---- clique (SA, E), the hub: (X, E) over every field the network has -----------------------------------

@message("sa_e", "e", "m_up^{H}(E) = P(E, evidence | {X})",
         ["= Π_i m_up^Dam(SA_i, E_i)"], "exact",
         series=("m_up^{H}(E) = P(E, evidence | {X})",
                 ["= Π_i m_up^Dam(SA_i, E_i) · Π_i m_up^Edge(X_i, E_i)",
                  "over the edges of one bridge, and the others"]))
def sa_e_up(t, ms, sa):
    """Message to clique (E), the field held at each chain's current values sa (C, n):
    q[c, i, e] = m_i(sa[c], e), up to the factor l_i(sa[c]), constant in E (m_i.held). Given the field
    the edges are independent, so q (C, n_edges, 2) is the whole message, one prior per chain. ms is the
    list of per-edge messages, or a _Batch of them all. n_ops = component evaluations."""
    if isinstance(ms, _Batch):
        return ms.held(sa), sa.numel()
    return torch.stack([m.held(sa) for m in ms], dim=1), sa.numel()


@message("e", "sa_e", "m_down^E(E) = P(E | evidence)",
         ["sent as a sample of E", "---",
          "P(ln SA | E, evidence) ∝ N(ln SA; μ, Σ) Π_i m_up^Dam(SA_i, E_i);",
          "SA is not observed directly, but only through SAObs"],
         "ess",
         series=("m_down^E(E) = P(E | evidence)",
                 ["sent as a sample of E", "---",
                  "P(ln X | E, evidence) ∝ N(ln X; μ, Σ) Π_i m_up(X_i, E_i), X = ({X}) at every component,",
                  "each field its own Gaussian, independent of the others;",
                  "ground motion is not observed directly, SA only through SAObs"]))
def sa_e_down(t, ms, fld, sa, e, generator=None):
    """Message from clique (E): the new E (C, n_edges). The field moves one step from its current values sa
    (C, n) towards N(ln x; field()) prod_i m_i(x, e_i): one elliptical slice sampling update (Murray,
    Adams and MacKay 2010) of ln x. Each chain shrinks its own angle bracket until its proposal is
    above its slice. Returns sa (C, n); n_ops = component evaluations."""
    mean, chol = fld
    C, n = sa.shape

    def loglik(s, rows):                                            # s (len(rows), n)
        r = torch.arange(len(rows))
        if isinstance(ms, _Batch):
            return ms.log(s).gather(2, e[rows][:, :, None])[:, :, 0].sum(1)
        return sum(m.log(s)[r, e[rows, i]] for i, m in enumerate(ms))

    f0 = sa.log() - mean                                            # current point, centred
    nu = torch.randn(C, n, dtype=torch.float64, generator=generator) @ chol.T
    log_y = loglik(sa, torch.arange(C)) + torch.rand(C, dtype=torch.float64, generator=generator).log()
    theta = 2 * math.pi * torch.rand(C, dtype=torch.float64, generator=generator)
    lo, hi = theta - 2 * math.pi, theta.clone()
    out, todo, n_ops = sa.clone(), torch.arange(C), C * n
    while len(todo):
        check_cancelled()
        th = theta[todo, None]
        s = torch.exp(mean + f0[todo] * torch.cos(th) + nu[todo] * torch.sin(th))
        ok = loglik(s, todo) > log_y[todo]
        n_ops += len(todo) * n
        out[todo[ok]] = s[ok]
        todo = todo[~ok]                                            # shrink towards the current point
        neg = theta[todo] < 0
        lo[todo] = torch.where(neg, theta[todo], lo[todo])
        hi[todo] = torch.where(neg, hi[todo], theta[todo])
        theta[todo] = lo[todo] + (hi[todo] - lo[todo]) * torch.rand(len(todo), dtype=torch.float64,
                                                                      generator=generator)
    return out, n_ops


# ---- clique (E), the root -----------------------------------------------------------------------------

@root("e", "P(E | {X}, evidence)",
      ["∝ m_up^{H}(E) · Π_k m_up^Sys(E)", "---",
       "E takes one step of RSR-MCMC   if any S is in the evidence",
       "E is drawn directly from m_up^{H}(E)   otherwise"], ["rsr", "multinomial"])
def root_absorb(q, mdl, state, bridge=8, generator=None):
    """Clique (E) absorbs q, the message from the hub, (C, n, 2) one prior per chain, and the
    observed systems' boxes, and moves E one step towards its belief, for C chains.
    With boxes (mdl, the rsr.mcmc model of the boxes from _cut): one rsr.mcmc.step under the prior q
    of each chain from state, an rsr.mcmc.State (E as state indices (C, n), the labels and their
    memberships): a sweep over the observed systems then, unless bridge is None or 0, a tuple jump
    with bridge partner tuples. The step leaves P(E | X, evidence) invariant, as the Gibbs sweep needs. Without
    boxes (mdl None): a direct draw from q, as each chain's E given the field is then independent by edge.
    Returns the new state and n_ops = E vectors drawn."""
    from rsr import mcmc

    C = len(state.x)
    p = q / q.sum(dim=-1, keepdim=True)
    if mdl is None:
        e = torch.multinomial(p.expand(C, -1, -1).reshape(-1, 2), 1, generator=generator).view(C, -1)
        return mcmc.State(e, None, None), C
    n_partner = bridge or 0
    state, _ = mcmc.step(mcmc.with_probs(mdl, p.expand(C, -1, -1)), state, generator, n_partner=n_partner)
    return state, C * len(mdl.systems) + C * n_partner


def _cut(certs, support):
    """Each observed system's boxes cut to the support (n, 2) of the evidence on E, so that evidence
    holds in every box; boxes left empty are dropped."""
    refs_list = []
    for refs in certs:
        cut = refs * support.int()
        refs_list.append(cut[(cut.sum(dim=2) > 0).all(dim=1)])
    return refs_list


# ---- clique (E, S_k), one per system k ----------------------------------------------------------------

@message("e_s", "e", "m_up^Sys(E) = P(evidence | E)",
         ["= 1[E ∈ the lower references]   if S = 0 is in the evidence",
          "= 1[E ∈ the upper references]   if S = 1 is in the evidence",
          "= 1   if S is not in the evidence"], "rsr_rules")
def system_up(t, k, ev):
    """Message to clique (E): None (the constant 1) if S_k is not observed, else the certificates of
    its observed state, the stored rsr rules: the failure reference states for Fail, the survival
    ones for Survival. Their boxes cover the event only if the rules are complete at the system's
    threshold. n_ops = 0."""
    s = ev.get(f"SystemState::{k}")
    if s is None:
        return None, 0
    if t["systems"][k].rules is None:
        raise ValueError(f"SystemState::{k} cannot be observed: its system has no stored rsr rules")
    upper, lower = t["systems"][k].rules
    return (lower if s == 0 else upper), 0


@message("e", "e_s", "m_down^E(E) = P(E | evidence)",
         ["sent as a sample of E", "---", "P(S | E, evidence) = δ(S − f(E)) if S is not in the evidence;",
          "= δ(S − x) if it is;", "f(E) by the rsr rules: S = 1 in a survival box, 0 in a failure box"],
         ["rsr_rules", "ndtools"])
def system_down(t, k, e, ev, opened=None):
    """S_k for each E row (N, n), or with each of the opened edges opened as well (N, len(opened)): the same
    message as cliques.py's, by the system's rsr rules, learnt further from the rows they leave undecided.
    n_ops = system function calls to learn."""
    return cliques.system_down(t, k, e, ev, opened)


# ---- a whole run --------------------------------------------------------------------------------------

def _timer(cost):
    """timed(step, f, ...) calls f, which returns (result, n_ops), and adds its seconds and n_ops to cost[step]."""
    def timed(step, f, *args, **kwargs):
        check_cancelled()
        t0 = time.perf_counter()
        out, n_ops = f(*args, **kwargs)
        check_cancelled()
        c = cost.setdefault(step, {"sec": 0.0, "n_ops": 0})
        c["sec"] += time.perf_counter() - t0
        c["n_ops"] += n_ops
        return out
    return timed


def _forward(t, n, generator):
    """n independent draws straight down the tree: the field, then each edge's components from their own
    laws and the edge from them (series); a lone edge's damage from its fragility and the edge from it.
    Exact when nothing is observed. Returns (x (n, n_sites), E (n, n_edges), {component variable: (n,)});
    n_ops = rows."""
    x = field_draw(field(t), n, generator)
    e = torch.empty(n, len(t["edges"]), dtype=torch.long)
    own = {}
    for i, c in enumerate(t["edges"]):
        check_cancelled()
        if c["lone"]:
            d = torch.multinomial(c["frag"].distribution(x[:, c["site"]]), 1, generator=generator)[:, 0]
            own[c["D"]] = d
            e[:, i] = c["to_e"][d].argmax(dim=1)
            continue
        comps = c["comps"]
        if not comps:
            e[:, i] = 1 if "do_e" not in c else c["do_e"]
            continue
        tables = [_component(comp, x, {}) for comp in comps]
        opens = torch.stack([torch.rand(n, dtype=torch.float64, generator=generator) * tot.clamp_min(1e-300) < op
                             for tot, op in tables], dim=1)
        for j, comp in enumerate(comps):
            own.update(_component_down(comp, x, opens[:, j], {}, generator))
        e[:, i] = opens.all(dim=1).long() if "do_e" not in c else c["do_e"]
    return (x, e, own), n


def _feasible(t, ms, fld, sa, generator, tries=1000):
    """The chains' starting field sa (C, n), each chain whose field makes the evidence impossible on some edge
    (its message is 0 in both states of E, e.g. a slide observed where PGA is below the slope's critical
    acceleration) redrawn from field() until it allows it. Elliptical slice sampling then keeps it where the
    evidence is possible. n_ops = component evaluations."""
    sa, n_ops = sa.clone(), 0
    for _ in range(tries):
        q, ops = sa_e_up(t, ms, sa)
        n_ops += ops
        bad = ~(q.sum(dim=-1) > 0).all(dim=1)
        if not bad.any():
            return sa, n_ops
        sa[bad] = field_draw(fld, int(bad.sum()), generator)
        check_cancelled()
    raise ValueError(f"the evidence is not possible under the shaking field: no field drawn in {tries} tries "
                     f"allows it on every edge")


def sample(t, ev, n_chain=100, n_burn=200, n_keep=10_000, bridge=8, seed=0, sa0=None, do=None, e0=None):
    """Rows from P(everything | ev) under do (intervene()), N = n_keep * n_chain of them.

    Nothing observed: every up message is 1, so one pass down the tree is exact: N independent rows
    (_forward), no chains. Otherwise n_burn + n_keep sweeps of n_chain chains, each sweep sa_e_up,
    root_absorb and sa_e_down (see the module's docstring), each kept sweep then passed on down; row r
    from chain r % n_chain. The chains start from the field drawn from field(), or from sa0 (n_chain, n)
    in t["field_sites"] order if given; and E from e0 (n_chain, n_edges) where it lies in a box of every
    observed system's certificates, else from rsr.mcmc's sequential draw (mcmc.start). bridge: the partner
    tuples of the tuple jump that ends each step at the root (None or 0 for none). Returns rows
    {variable: (N,)}, each edge's field values, E, components and readings in turn, and cost
    {step: {"sec", "n_ops"}}, summed over the cliques of each kind."""
    check_cancelled()
    series = not all(c["lone"] for c in t["edges"])
    if direct := [s for s in t["field_sites"] if s in ev]:
        raise ValueError(f"{'ground motion' if series else 'SA'} is observed only through SAObs, not directly: {direct}")
    # Evidence on a PGD is whether the ground failed, PGD > 0 (1) or not (0); its displacement is not observed.
    if pgd := [comp["PGD"] for c in t["edges"] for comp in c["comps"] if comp.get("PGD") in ev and ev[comp["PGD"]] not in (0, 1)]:
        raise ValueError(f"evidence on ground displacement is whether the ground failed, 1 (PGD > 0) or 0: {pgd}")
    do = dict(do or {})
    if do:
        t = intervene(t, do)
    g, cost = torch.Generator().manual_seed(seed), {}
    timed = _timer(cost)
    rows = {}

    def put(x, e, draw):
        """Each edge in turn: the field at its components, E, its components (draw(i, c)), its readings."""
        for i, c in enumerate(t["edges"]):
            for comp in c["comps"]:
                for s, k in zip(comp["sites"], comp["idx"]):
                    if s not in rows:
                        rows[s] = x[:, k]
            rows[c["E"]] = e[:, i]
            draw(i, c)
            for j, comp in enumerate(c["comps"]):
                if comp["kind"] == "bridge" and comp["obs"] is not None:
                    rows[comp["Obs"]] = timed("obs_down", obs_down, t, i, x[:, comp["idx"][0]], ev, generator=g, j=j)

    if not ev:
        x, e, own = timed("forward", _forward, t, n_chain * n_keep, g)

        def drawn(i, c):
            for comp in c["comps"]:
                for v in ((comp["D"],) if comp["kind"] == "bridge" else (comp["R"], comp["PGD"])):
                    rows[v] = own[v]

        put(x, e, drawn)
    else:
        from rsr import mcmc

        # the leaves' messages, built once
        ls = [timed("obs_up", obs_up, t, i, ev) for i in range(len(t["edges"]))]
        ms = [timed("edge_up", edge_up, t, i, ev, l) if c["lone"] else timed("series_up", series_up, t, i, ev, l)
              for (i, c), l in zip(enumerate(t["edges"]), ls)]
        if series:
            try:
                ms = _Batch(t, ev)                 # the same messages, every edge at once
            except NotImplementedError:
                pass
        certs = [timed("system_up", system_up, t, k, ev) for k in t["systems"]]
        fld = field(t)
        sa = field_draw(fld, n_chain, g) if sa0 is None else sa0.clone()
        if series:
            sa = timed("sa_e_up", _feasible, t, ms, fld, sa, g)
        q = timed("sa_e_up", sa_e_up, t, ms, sa)
        support = torch.stack([_lone_support(c, ev) if c["lone"] else _allowed(c, ev)
                               for c in t["edges"]])                     # (n, 2), the E the evidence allows
        refs_list = _cut([r for r in certs if r is not None], support)
        if refs_list:
            # the boxes are built once; each step swaps in the chains' current prior
            mdl = mcmc.model(refs_list, (q / q.sum(dim=-1, keepdim=True)).expand(n_chain, -1, -1))
            state = mcmc.start(mdl, n_chain, g, x0=e0)   # warm start: E where it already is, if in the boxes
        else:
            mdl, state = None, mcmc.State(torch.zeros(n_chain, len(t["edges"]), dtype=torch.long), None, None)
        kept_sa, kept_e = [], []
        for sweep in range(n_burn + n_keep):
            check_cancelled()
            if sweep:
                q = timed("sa_e_up", sa_e_up, t, ms, sa)
            state = timed("root_absorb", root_absorb, q, mdl, state, bridge, generator=g)
            e = state.x
            sa = timed("sa_e_down", sa_e_down, t, ms, fld, sa, e, generator=g)
            if sweep >= n_burn:
                kept_sa.append(sa)
                kept_e.append(e)
        # each kept sweep passed on down
        x, e = torch.cat(kept_sa), torch.cat(kept_e)

        def down(i, c):
            if c["lone"]:
                rows[c["D"]] = timed("edge_down", edge_down, t, i, x[:, c["site"]], e[:, i], ev, generator=g)
            elif c["comps"]:
                f, tables = timed("series_down", series_down, t, i, x, e[:, i], ev, generator=g)
                rows.update(timed("comp_down", comp_down, t, i, x, f, tables, ev, generator=g))

        put(x, e, down)
    # the systems, each distinct edge vector once (most rows repeat a few)
    unique, inverse = torch.unique(e, dim=0, return_inverse=True)
    for k in t["systems"]:
        if f"SystemState::{k}" not in do:                                     # a forced one is set below
            rows[f"SystemState::{k}"] = timed("system_down", system_down, t, k, unique, ev)[inverse]
    for name, value in do.items():                  # a system state or a reading set by do: as set
        if name.startswith("SystemState::"):
            rows[name] = torch.full((len(e),), int(value))
        elif name in rows and name.startswith("SAObs"):
            rows[name] = torch.full_like(rows[name], value)
    check_cancelled()
    return rows, cost
