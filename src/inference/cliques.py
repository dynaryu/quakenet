"""Sampling on the clique tree: one function per clique and direction.

The tree is the one in the app's Clique tree tab, for a network whose edges each carry one bridge:

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

sample() runs n_chain chains of sweeps, each sweep one pass through the tree with clique (E) as the
root. The leaves' messages are exact functions, built once: obs_up, then edge_up for each edge, and
system_up for each system. In each sweep:

    collect     sa_e_up: P(E, evidence | SA) at each chain's current SA. Summed over SA instead, the
                message has no closed form on a correlated field, so clique (E) conditions on SA.
    root        root_absorb: E moves one step by certificate MCMC, given that message and the
                observed systems' certificates; with no system observed, it is drawn from that
                message directly.
    distribute  sa_e_down: the new E; SA moves one step from its previous value by elliptical slice
                sampling. Any move that leaves P(SA | E, evidence) invariant would do.

The sweeps form a Metropolis-within-Gibbs chain over (SA, E). After the burn-in, each kept sweep
is passed on down the tree: edge_down, obs_down, system_down. With nothing observed, every up
message is 1 and no chain is needed: one pass down the tree draws independent rows (_forward).

Every function returns its result and n_ops, the count of its dominant operation (in its docstring);
sample() times each call, so every up and down step has its own cost. What each message delivers is
also declared beside its function (@message, @root), and describe() hands those declarations to the
app, which draws them on the tree.

The certificate MCMC comes from rsr.mcmc (rsr-duco 0.3.1 or later, from PyPI). It is imported where it is
used, so the app can read describe() without it.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import tempfile
import time
from pathlib import Path

import torch
from src.cancellation import check_cancelled

from src.builder.kernels.hazard import GaussianField
from src.builder.kernels.observation import ObservedIntensity

# ---- what each message delivers, for the app ----------------------------------------------------------
#
# A function that passes a message is marked @message(from, to, carries, delivers, toolkit): the kinds of
# clique it goes between; what it carries, m_up(separator) or m_down(separator) = a probability over the
# variables the two share, "evidence" meaning the evidence on the sender's side going up, towards the root, and
# on the far side going down; how it is computed or delivered, line by line; and the documentation of what does
# the work. m_up^X or m_down^X names its sender, X: SO (SA, SAObs), Dam (SA, D, E), Sa (SA, E), E (E) and
# Sys (E, S); the app writes up or down as a subscript and X as a superscript. The app
# shows carries on the tree and the rest in a window. A down message's lines give what is sent, then, after a
# line "---" (a rule in the window), the distribution the receiving clique draws its own variables from. The
# root's own step is marked @root.

KINDS = {"sa_obs": "(SA_i, SAObs_i)", "sa_d_e": "(SA_i, D_i, E_i)", "sa_e": "(SA, E)", "e": "(E)", "e_s": "(E, S_k)"}
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


def message(source, target, carries, delivers, toolkit=None):
    """Declare the decorated function the sender of the message from clique kind source to target.
    toolkit: a key of TOOLKITS, or a list of them."""
    def mark(f):
        _MESSAGES.append({"function": f.__name__, "from": source, "to": target, "carries": carries,
                          "delivers": delivers, "toolkit": toolkit})
        return f
    return mark


def root(kind, belief, does, toolkit=None):
    """Declare the decorated function the root's own step, at clique kind kind: the belief it holds, from
    the messages it receives, and how it moves towards it, line by line as a message's."""
    def mark(f):
        _ROOT.update(function=f.__name__, clique=kind, belief=belief, does=does, toolkit=toolkit)
        return f
    return mark


def describe():
    """The message passing as this module does it: the root, every message and the toolkits they use."""
    return {"root": _ROOT, "messages": _MESSAGES, "kinds": KINDS, "toolkits": TOOLKITS}


# ---- the tree as the app draws it ---------------------------------------------------------------------
#
# drawing() gives the Clique tree tab this module's tree for a BN, from its nodes ({"id", "row", "function",
# "parents"}, as compiled.nodes() gives them): every clique's members, the separator it shares with the clique it
# hangs off, its kind (the messages across that link, as describe() declares them) and the factors assigned to
# it, each {"p": [child], "given": [parents]} for P(child | parents) or {"f": [system], "given": [edges]} for a
# system's function. The app only draws it. cliques_series.drawing() is this with series edges.

def _p(of, given=()):
    return {"p": list(of), "given": list(given)}


def _role(node):
    return node["function"].split(".")[0]


def _is_field(node):
    """Whether the node is the shaking field's (SA, PGV, PGA at a site), not a ground-failure component's PGD."""
    return node["function"].startswith(("hazard.shakemap", "hazard.gmm"))


def _given(by, i):
    """P(i | its parents), as the BN has it."""
    return _p([i], by[i]["parents"])


def _rest(nodes, local, comp=()):
    """The cliques every tree has around its edges' (local) and components' (comp): one per measurement, hanging
    off the clique that holds its intensity; the hub, every edge and every field; the clique of every edge alone,
    between the hub and the systems; and one per system, the system and the edges it reads."""
    obs = []
    for n in nodes:
        if _role(n) != "observation":
            continue
        im = n["parents"][0]
        part = next((k for k, c in enumerate(comp) if im in c["members"]), -1)
        edge = comp[part]["edge"] if part >= 0 else next((k for k, c in enumerate(local) if im in c["members"]), -1)
        obs.append({"label": n["row"], "members": [im, n["id"]], "separator": [im], "edge": edge, "comp": part,
                    "kind": "sa_obs", "factors": [_p([n["id"]], [im])]})
    field = [n["id"] for n in nodes if _is_field(n)]           # a ground-failure chain sits in its component's clique
    edges = [n["id"] for n in nodes if _role(n) == "edge"]
    quantities = list(dict.fromkeys(f.split("::")[0] for f in field))
    hub = {"members": edges + field, "kind": "sa_e",
           "factors": [_p([f for f in field if f.split("::")[0] == q]) for q in quantities]}   # one Gaussian per field
    systems = [{"label": n["row"], "members": n["parents"] + [n["id"]], "separator": n["parents"], "kind": "e_s",
                "factors": [{"f": [n["id"]], "given": n["parents"]}]} for n in nodes if _role(n) == "system"]
    return {"obs": obs, "comp": list(comp), "local": local, "hub": hub,
            "edges": {"members": edges, "separator": edges, "kind": "e", "factors": ["no factor"]}, "systems": systems}


def drawing(nodes, tables=None):
    """The tree for the Clique tree tab: clique (SA_i, D_i, E_i) per edge, with P(D_i | SA_i) and P(E_i | D_i), and
    the cliques around them (_rest). tables: unused here (an edge's table orders a series edge's components)."""
    by, local = {n["id"]: n for n in nodes}, []
    for n in nodes:
        if _role(n) != "edge":
            continue
        (d,) = n["parents"]                                       # one bridge
        sa = by[d]["parents"]
        local.append({"label": n["row"], "members": sa + [d, n["id"]], "separator": sa + [n["id"]], "kind": "sa_d_e",
                      "factors": [_given(by, d), _p([n["id"]], [d])]})
    return _rest(nodes, local)


# ---- the tree's pieces ----------------------------------------------------------------------------------

def tree(probs, corr=None):
    """The pieces each clique needs, from tbnpy's probability objects (define_model()[1]).
    The edges follow the system kernels' edge order, which every E vector uses.
    The hazard is a Gaussian field of ln SA (kernels.hazard.GaussianField: a GMM or a ShakeMap),
    ln SA ~ N(ln_median, D R D) with D = diag(sigma), or several over disjoint sites (e.g. one GMM
    per region), independent of each other.
    corr: the correlation R (n_sites, n_sites) of ln SA, in the hazard's site order, to use instead of
    the model's; None for the model's own."""
    fields = [p for p in probs.values() if isinstance(p, GaussianField)]
    if not fields:
        raise ValueError("the clique tree needs a hazard field of SA (hazard.gmm or hazard.shakemap)")
    sites = [v.name for p in fields for v in p.childs]
    ln_median = torch.cat([p.ln_median for p in fields])
    sigma = torch.cat([p.sigma for p in fields])
    if corr is None:
        corr = torch.block_diag(*[p.corr for p in fields])
    systems = {k.split("::")[1]: p for k, p in probs.items() if k.startswith("SystemState::")}
    observed = {p.parents[0].name: p for p in probs.values() if isinstance(p, ObservedIntensity)}
    edges = []
    for e in next(iter(systems.values())).edge_ids:
        edge = probs[f"EdgeOpen::{e}"]
        assert len(edge.parents) == 1, "one bridge per edge"
        frag = probs[edge.parents[0].name]
        obs = observed.get(frag.parents[0].name)                         # None if SA_i has no measurement
        edges.append({"E": edge.childs[0].name, "D": frag.childs[0].name, "SA": frag.parents[0].name,
                      "Obs": obs and obs.childs[0].name, "obs": obs,
                      "site": sites.index(frag.parents[0].name), "frag": frag,
                      "to_e": torch.nn.functional.one_hot(edge.open_if.long(), 2).double()})   # (K, 2)
    return {"edges": edges, "ln_median": ln_median, "sigma": sigma, "corr": corr, "systems": systems,
            "sites": sites}


def field(t):
    """The factor of clique (SA, E) in edge order: ln SA ~ N(ln_median, D R D), with D = diag(sigma)
    and R = corr (the identity if None). Returns the mean (n,) and a square root A (n, n) of the
    covariance, A A' = D R D: its Cholesky factor, or from its eigenvectors when co-located sites make
    it singular."""
    site = torch.tensor([c["site"] for c in t["edges"]])
    mu, sd = t["ln_median"][site], t["sigma"][site]
    R = torch.eye(len(site), dtype=torch.float64) if t["corr"] is None else t["corr"][site][:, site]
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
    """The tree under do: an edge set to a state keeps it whatever its bridge's damage; a damage state
    set keeps it whatever SA, and its edge follows it. SA cannot be set: it is observed only through
    SAObs, and set here it would cut the field. System states and SAObs set are written into the
    rows by sample()."""
    if any(c["SA"] in do for c in t["edges"]):
        raise ValueError("SA cannot be set by do")
    edges = []
    for c in t["edges"]:
        c = dict(c)
        if c["D"] in do:
            c["frag"] = _Fixed(c["frag"], int(do[c["D"]]))
        if c["E"] in do:
            c["to_e"] = torch.nn.functional.one_hot(torch.full((len(c["to_e"]),), int(do[c["E"]])), 2).double()
        edges.append(c)
    return {**t, "edges": edges}


def field_draw(fld, N, generator=None):
    """N independent draws of SA (N, n) in g from field()'s Gaussian."""
    mean, chol = fld
    return torch.exp(mean + torch.randn(N, len(mean), dtype=torch.float64, generator=generator) @ chol.T)


def _lam(ev, name, k):
    """The evidence on a discrete variable over its k states: 1 at the observed state only, or 1 everywhere."""
    if name not in ev:
        return torch.ones(k, dtype=torch.float64)
    return torch.nn.functional.one_hot(torch.tensor(ev[name]), k).double()


# ---- clique (SA_i, SAObs_i), one per edge i -----------------------------------------------------------

@message("sa_obs", "sa_d_e", "m_up^SO(SA) = P(evidence | SA)",
         ["= N(SAObs = x; SA, (ε·SA)²) if SAObs is in the evidence;", "= 1 otherwise"], "normal")
def obs_up(t, i, ev):
    """Message to clique (SA_i, D_i, E_i), a function of SA_i: None (the constant 1) if SAObs_i is not
    observed, else the measurement's likelihood l(sa) = N(o; sa, (epsilon sa)^2) at SAObs_i = o.
    l.log is log l, finite where l underflows. n_ops = 0."""
    c = t["edges"][i]
    if c["obs"] is None or c["Obs"] not in ev:
        return None, 0
    kernel, o = c["obs"], float(ev[c["Obs"]])

    def log(sa):
        return kernel.distribution(sa).log_prob(torch.full_like(sa, o))

    def l(sa):
        return log(sa).exp()

    l.log = log
    return l, 0


@message("sa_d_e", "sa_obs", "m_down^Dam(SA) = P(SA | evidence)",
         ["sent as a sample of SA", "---", "P(SAObs | SA, evidence) = N(SA, (ε·SA)²) if SAObs is not in the evidence;",
          "= δ(SAObs − x) if it is, as x"], "normal")
def obs_down(t, i, sa, ev, generator=None):
    """SAObs_i given each row's SA_i (N,): a draw from the measurement kernel's distribution, by its
    inverse CDF. An observed SAObs_i keeps its value. Returns SAObs_i (N,) in g; n_ops = draws."""
    c = t["edges"][i]
    if c["Obs"] in ev:
        return torch.full((len(sa),), float(ev[c["Obs"]]), dtype=torch.float64), 0
    u = torch.rand(len(sa), dtype=torch.float64, generator=generator)
    return c["obs"].distribution(sa).icdf(u), len(sa)


# ---- clique (SA_i, D_i, E_i), one per edge i ----------------------------------------------------------

@message("sa_d_e", "sa_e", "m_up^Dam(SA, E) = P(E, evidence | SA)",
         ["= Σ_D P(D | SA) P(E | D) · m_up^SO(SA)   if neither D nor E is in the evidence",
          "= P(D = d | SA) P(E | D = d) · m_up^SO(SA)   if only D is, as d",
          "= Σ_D P(D | SA) P(E = e | D) · m_up^SO(SA) if E = e, 0 otherwise   if only E is, as e",
          "= P(D = d | SA) P(E = e | D = d) · m_up^SO(SA) if E = e, 0 otherwise   if both are"], "exact")
def edge_up(t, i, ev, l=None):
    """Message to clique (SA, E), a function of (SA_i, E_i), with l absorbed, the message from
    clique (SA_i, SAObs_i) (None for the constant 1):
        m(sa, e) = l(sa) sum_d P(D_i = d | sa) 1[open(d) = e] lambda_D(d) lambda_E(e).
    Returns m, mapping sa (N,) to an (N, 2) table over E_i, with m.held, m without l: as l(sa) is the
    same for both states of E_i, m.held is the message to clique (E) while SA is held, and it never
    vanishes where l(sa) underflows; and m.log, log m, finite there too. n_ops = 0: nothing is
    evaluated yet."""
    c = t["edges"][i]
    w = _lam(ev, c["D"], len(c["to_e"]))[:, None] * c["to_e"] * _lam(ev, c["E"], 2)          # (K, 2)

    def held(sa):
        return c["frag"].distribution(sa) @ w

    def m(sa):
        return held(sa) if l is None else held(sa) * l(sa)[:, None]

    def log(sa):
        return held(sa).log() if l is None else held(sa).log() + l.log(sa)[:, None]

    m.held, m.log = held, log
    return m, 0


@message("sa_e", "sa_d_e", "m_down^Sa(SA, E) = P(SA, E | evidence)",
         ["sent as a sample of (SA, E)", "---", "P(D | SA, E, evidence) ∝ P(D | SA) P(E | D) if D is not in the evidence,",
          "over the damage states that give E;", "= δ(D − d) if it is, as d"], "multinomial")
def edge_down(t, i, sa, e, ev, generator=None):
    """D_i given each row's SA_i and E_i (N,): a categorical over the damage states with weights
    P(D_i = d | sa) 1[open(d) = e] lambda_D(d). An observed D_i keeps its value (its only weight).
    Returns D_i (N,) state indices; n_ops = fragility evaluations."""
    c = t["edges"][i]
    w = c["frag"].distribution(sa) * _lam(ev, c["D"], len(c["to_e"])) * c["to_e"][:, e].T      # (N, K)
    return torch.multinomial(w, 1, generator=generator)[:, 0], len(sa)


# ---- clique (SA, E) -----------------------------------------------------------------------------------

@message("sa_e", "e", "m_up^Sa(E) = P(E, evidence | SA)",
         ["= Π_i m_up^Dam(SA_i, E_i)"], "exact")
def sa_e_up(t, ms, sa):
    """Message to clique (E), SA held at each chain's current values sa (C, n):
    q[c, i, e] = m_i(sa[c, i], e), up to the factor l_i(sa[c, i]), constant in E (m_i.held). Given SA
    the edges are independent, so q (C, n, 2) is the whole message, one prior per chain.
    n_ops = fragility evaluations."""
    return torch.stack([m.held(sa[:, i]) for i, m in enumerate(ms)], dim=1), sa.numel()


@message("e", "sa_e", "m_down^E(E) = P(E | evidence)",
         ["sent as a sample of E", "---",
          "P(ln SA | E, evidence) ∝ N(ln SA; μ, Σ) Π_i m_up^Dam(SA_i, E_i);",
          "SA is not observed directly, but only through SAObs"],
         "ess")
def sa_e_down(t, ms, fld, sa, e, generator=None):
    """Message from clique (E): the new E (C, n). SA moves one step from its current values sa (C, n)
    towards N(ln SA; field()) prod_i m_i(SA_i, e_i): one elliptical slice sampling update (Murray,
    Adams and MacKay 2010) of ln SA. Each chain shrinks its own angle bracket until its proposal is
    above its slice. Returns sa (C, n); n_ops = fragility evaluations."""
    mean, chol = fld
    C, n = sa.shape

    def loglik(s, rows):                                            # s (len(rows), n)
        r = torch.arange(len(rows))
        return sum(m.log(s[:, i])[r, e[rows, i]] for i, m in enumerate(ms))

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

@root("e", "P(E | SA, evidence)",
      ["∝ m_up^Sa(E) · Π_k m_up^Sys(E)", "---",
       "E takes one step of RSR-MCMC   if any S is in the evidence",
       "E is drawn directly from m_up^Sa(E)   otherwise"], ["rsr", "multinomial"])
def root_absorb(q, mdl, state, bridge=8, generator=None):
    """Clique (E) absorbs q, the message from clique (SA, E), (C, n, 2) one prior per chain, and the
    observed systems' boxes, and moves E one step towards its belief, for C chains.
    With boxes (mdl, the rsr.mcmc model of the boxes from _cut): one rsr.mcmc.step under the prior q
    of each chain from state, an rsr.mcmc.State (E as state indices (C, n), the labels and their
    memberships): a sweep over the observed systems then, unless bridge is None or 0, a tuple jump
    with bridge partner tuples. The step leaves P(E | SA, evidence) invariant, as the Gibbs sweep needs. Without boxes
    (mdl None): a direct draw from q, as each chain's E given SA is then independent by edge.
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
    upper, lower = t["systems"][k].rules
    return (lower if s == 0 else upper), 0


@message("e", "e_s", "m_down^E(E) = P(E | evidence)",
         ["sent as a sample of E", "---", "P(S | E, evidence) = δ(S − f(E)) if S is not in the evidence;",
          "= δ(S − x) if it is;", "f(E) by the rsr rules: S = 1 in a survival box, 0 in a failure box"],
         ["rsr_rules", "ndtools"])
def system_down(t, k, e, ev, opened=None):
    """S_k for each E row (N, n), by the system's rsr rules: a row in a box of the survival rules survives, one
    in a box of the failure rules fails. An observed S_k keeps its value.

    learn(): the rows no rule decides are learnt from first. rsr resolves them by the system function (ndtools)
    into new reference states, starting from the system's rules (its own, borrowed from other thresholds, or
    none) and keeping these rows as its samples, until every one is decided. The system keeps the rules for
    later messages, and they are stored in its own rules folder (rules_home: rsr/<origin>/..., where the builder
    finds them next time), created if new, unless QUAKENET_SAVE_RULES=0; that folder is its rules from then on.

    opened (column indices of e): S_k for each row with each of those edges opened as well, (N, len(opened)), the
    message under do(E_j = open), the rest of the row kept. A row whose edge j is already open keeps its state.
    One that is closed is decided from its violations of each rule (the components whose state the rule does
    not allow), which the rules' classification gives: opening j puts it in a rule's box if it already is in it
    and the rule allows j open, or if j was its one violation and the rule needs j open. Two matrix products per
    side decide every edge at once; the opened rows that no rule then decides are learnt from as above.

    Returns S_k, (N,) or (N, len(opened)); n_ops = system function calls to learn."""
    name, prob = f"SystemState::{k}", t["systems"][k]
    shape = (len(e),) if opened is None else (len(e), len(opened))
    if name in ev:
        return torch.full(shape, ev[name], dtype=torch.long), 0
    if not len(e):
        return torch.zeros(shape, dtype=torch.long), 0
    from rsr._checkpoint import save_json, save_tensor
    from rsr.rsr import classify_samples_with_indices, run_ref_extraction_by_mcs

    home, directory = getattr(prob, "rules_home", None), getattr(prob, "rules_directory", None)
    names = {"upper": "refs_up_1.pt", "lower": "refs_low_0.pt"}
    empty = torch.empty((0, len(prob.edge_ids), 2), dtype=torch.int32)
    n_ops = 0

    def onehot(x):
        return torch.nn.functional.one_hot(x.long(), 2).int()

    def learn(rows):
        """Extend the rules until they decide every one of rows (R, n); keep them, and store them if their own."""
        nonlocal n_ops
        rules = prob.rules if prob.rules is not None else (empty, empty)
        r = classify_samples_with_indices(onehot(rows), *rules, return_masks=True)
        prob.tally["unclassified"] += r["unknown"]
        if not r["unknown"]:
            return
        check_cancelled()
        save = home is not None and directory is not None and os.environ.get("QUAKENET_SAVE_RULES", "1") != "0"
        start = {"refs_upper": [], "refs_lower": [], "refs_mat_upper": rules[0], "refs_mat_lower": rules[1]}
        ref = prob.rules_reference or {}
        if save:                       # the rules the system has on file: the .json references with the .pt tensors
            for side in ("upper", "lower"):
                if ref.get(side):
                    f = Path(directory) / ref[side]
                    start[f"refs_{side}"] = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
                    start[f"refs_mat_{side}"] = torch.load(f, map_location="cpu", weights_only=True).to(torch.int32)
                else:
                    start[f"refs_mat_{side}"] = empty
        f = prob.system_function()
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
            res = run_ref_extraction_by_mcs(
                sfun=lambda comps: (f(comps)[1],) * 2 + (None,), probs=torch.full((len(prob.edge_ids), 2), 0.5),
                row_names=list(prob.edge_ids), n_state=2, sys_upper_st=1, **start,
                samples=rows.long()[r["mask_unknown"]], save_every=10 ** 9, ref_update_verbose=False,
                output_dir=scratch)
            mats = {side: torch.load(res[f"refs_{side}_pt_path"], map_location="cpu", weights_only=True).to(torch.int32)
                    for side in ("upper", "lower")}
            refs = {side: json.loads(Path(res[f"refs_{side}_path"]).read_text(encoding="utf-8"))
                    for side in ("upper", "lower")}
        prob.rules = (mats["upper"], mats["lower"])
        calls = sum(m.get("n_sfun_upper", 0) + m.get("n_sfun_lower", 0) for m in res["metrics_log"])
        n_ops += calls
        if save:
            folder = Path(directory) / home["folder"]
            folder.mkdir(parents=True, exist_ok=True)
            for side, file in names.items():
                save_json(refs[side], (folder / file).with_suffix(".json"))
                save_tensor(mats[side], folder / file)
            meta_file = folder / "model.json"
            # a new folder gets what the builder checks the rules against (their system and threshold)
            meta = (json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else
                    {"rules_sha256": home["rules_sha256"], "target_add_h": home["target_add_h"],
                     "origin": home["origin"], "edge_order": home["edge_order"],
                     "system_value": "binary state",
                     "runs": [{"proposal": "none: rows of a session's samples, not a search", "rounds": 0,
                               "started_from": "stored rules" if ref else "no rules", "metrics": None}]})
            learnt = meta.get("learnt_from_samples") or {"rows": 0, "sfun_calls": 0}
            meta["learnt_from_samples"] = {
                "how": "rows of a session's samples the rules left undecided, resolved by rsr (samples kept)",
                "started_from": learnt.get("started_from", "stored rules" if ref else "no rules"),
                "rows": learnt["rows"] + int(r["unknown"]), "sfun_calls": learnt["sfun_calls"] + calls,
                "rules": {side: len(mats[side]) for side in ("upper", "lower")}}
            save_json(meta, meta_file)
            prob.rules_reference = {side: (Path(home["folder"]) / file).as_posix() for side, file in names.items()}
            prob.rules_reference.update(upper_at=home["target_add_h"], lower_at=home["target_add_h"])

    prob.tally["seen"] += len(e)
    learn(e)
    up, lo = prob.rules
    if opened is None:
        r = classify_samples_with_indices(onehot(e), up, lo, return_masks=True)
        if r["unknown"]:
            raise RuntimeError(f"{name}: {r['unknown']} rows left undecided after learning rules from them")
        return r["mask_upper"].long(), n_ops

    cols = torch.as_tensor(opened, dtype=torch.long)
    # per rule and opened edge: does it allow the edge open; is the edge its one violation when closed
    allows = {side: m[:, cols, 1].float() for side, m in (("upper", up), ("lower", lo))}
    needs = {side: ((m[:, cols, 0] == 0) & (m[:, cols, 1] == 1)).float() for side, m in (("upper", up), ("lower", lo))}
    out = torch.empty(shape, dtype=torch.long)
    left = []                                           # (row, opened index) that no rule decides once opened
    for begin in range(0, len(e), 20_000):
        check_cancelled()
        x = e[begin:begin + 20_000].long()
        r = classify_samples_with_indices(onehot(x), up, lo, return_masks=True, return_violations=True)
        inside = {}
        for side in ("upper", "lower"):
            v = r[f"violations_{side}"]
            inside[side] = (torch.zeros(len(x), len(cols), dtype=torch.bool) if v is None else
                            ((v == 0).float() @ allows[side] + (v == 1).float() @ needs[side]) > 0)
        closed = x[:, cols] == 0
        out[begin:begin + len(x)] = torch.where(closed, inside["upper"].long(), r["mask_upper"].long()[:, None])
        row, j = torch.where(closed & ~inside["upper"] & ~inside["lower"])
        left.append((row + begin, j))
    row = torch.cat([a for a, _ in left])
    j = torch.cat([b for _, b in left])
    if len(row):
        y = e[row].long().clone()
        y[torch.arange(len(row)), cols[j]] = 1
        unique, inverse = torch.unique(y, dim=0, return_inverse=True)
        learn(unique)
        r = classify_samples_with_indices(onehot(unique), *prob.rules, return_masks=True)
        if r["unknown"]:
            raise RuntimeError(f"{name}: {r['unknown']} opened rows left undecided after learning rules from them")
        out[row, j] = r["mask_upper"].long()[inverse]
    return out, n_ops


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
    """n independent draws of SA, D and E straight down the tree: SA from the field, each bridge's damage from
    its fragility, each edge's state from its damage. Exact when nothing is observed. n_ops = rows."""
    sa = field_draw(field(t), n, generator)
    d, e = {}, torch.empty(n, len(t["edges"]), dtype=torch.long)
    for i, c in enumerate(t["edges"]):
        check_cancelled()
        d[i] = torch.multinomial(c["frag"].distribution(sa[:, i]), 1, generator=generator)[:, 0]
        e[:, i] = c["to_e"][d[i]].argmax(dim=1)
    return (sa, e, d), n


def sample(t, ev, n_chain=100, n_burn=200, n_keep=10_000, bridge=8, seed=0, sa0=None, do=None, e0=None):
    """Rows from P(everything | ev) under do (intervene()), N = n_keep * n_chain of them.

    Nothing observed: every up message is 1, so one pass down the tree is exact: N independent rows
    (_forward), no chains. Otherwise n_burn + n_keep sweeps of n_chain chains, each sweep sa_e_up,
    root_absorb and sa_e_down (see the module's docstring), each kept sweep then passed on down; row r
    from chain r % n_chain. The chains start from SA drawn from field(), or from sa0 (n_chain, n) in edge
    order if given; and E from e0 (n_chain, n) where it lies in a box of every observed system's
    certificates, else from rsr.mcmc's sequential draw (mcmc.start). bridge: the partner tuples of the
    tuple jump that ends each step at the root (None or 0 for none). Returns rows {variable: (N,)} and
    cost {step: {"sec", "n_ops"}}, summed over the cliques of each kind."""
    check_cancelled()
    if direct := [c["SA"] for c in t["edges"] if c["SA"] in ev]:
        raise ValueError(f"SA is observed only through SAObs, not directly: {direct}")
    do = dict(do or {})
    if do:
        t = intervene(t, do)
    g, cost = torch.Generator().manual_seed(seed), {}
    timed = _timer(cost)
    rows = {}
    if not ev:
        sa, e, d = timed("forward", _forward, t, n_chain * n_keep, g)
        for i, c in enumerate(t["edges"]):
            rows[c["SA"]], rows[c["E"]], rows[c["D"]] = sa[:, i], e[:, i], d[i]
            if c["obs"] is not None:
                rows[c["Obs"]] = timed("obs_down", obs_down, t, i, sa[:, i], ev, generator=g)
    else:
        from rsr import mcmc

        # the leaves' messages, built once
        ls = [timed("obs_up", obs_up, t, i, ev) for i in range(len(t["edges"]))]
        ms = [timed("edge_up", edge_up, t, i, ev, l) for i, l in enumerate(ls)]
        certs = [timed("system_up", system_up, t, k, ev) for k in t["systems"]]
        fld = field(t)
        sa = field_draw(fld, n_chain, g) if sa0 is None else sa0.clone()
        q = timed("sa_e_up", sa_e_up, t, ms, sa)
        support = torch.stack([(_lam(ev, c["D"], len(c["to_e"]))[:, None] * c["to_e"] * _lam(ev, c["E"], 2)).sum(0) > 0
                               for c in t["edges"]])                     # (n, 2), the E the evidence on D and E allows
        refs_list = _cut([r for r in certs if r is not None], support)
        if refs_list:
            # the boxes are built once; each step swaps in the chains' current prior
            mdl = mcmc.model(refs_list, (q / q.sum(dim=-1, keepdim=True)).expand(n_chain, -1, -1))
            state = mcmc.start(mdl, n_chain, g, x0=e0)   # warm start: E where it already is, if in the boxes
        else:
            mdl, state = None, mcmc.State(torch.zeros(n_chain, len(ms), dtype=torch.long), None, None)
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
        sa, e = torch.cat(kept_sa), torch.cat(kept_e)
        for i, c in enumerate(t["edges"]):
            rows[c["SA"]], rows[c["E"]] = sa[:, i], e[:, i]
            rows[c["D"]] = timed("edge_down", edge_down, t, i, sa[:, i], e[:, i], ev, generator=g)
            if c["obs"] is not None:
                rows[c["Obs"]] = timed("obs_down", obs_down, t, i, sa[:, i], ev, generator=g)
    # the systems, each distinct edge vector once (most rows repeat a few)
    unique, inverse = torch.unique(e, dim=0, return_inverse=True)
    for k in t["systems"]:
        if f"SystemState::{k}" not in do:                                     # a forced one is set below
            rows[f"SystemState::{k}"] = timed("system_down", system_down, t, k, unique, ev)[inverse]
    for name, x in do.items():                      # a system state or a reading set by do: as set
        if name.startswith("SystemState::"):
            rows[name] = torch.full((len(e),), int(x))
        elif name in rows and name.startswith("SAObs"):
            rows[name] = torch.full_like(rows[name], x)
    check_cancelled()
    return rows, cost
