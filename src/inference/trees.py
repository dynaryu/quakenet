"""Which clique tree a BN is sampled on.

cliques.py is the tree of one bridge per edge under SA (EMA, Hebei, toy-ca). cliques_series.py is the tree of
edges that carry several components in series, bridges and ground-failure components, under SA, PGV and PGA
(Kahramanmaraş). A BN goes to cliques_series.py only when cliques.py cannot take it: some edge is not one bridge,
or some field is not SA. So a network of one bridge per edge under SA is built, sampled and described exactly
as cliques.py does it, untouched by the series machinery.
"""

from __future__ import annotations

from . import cliques, cliques_series
from src.builder.kernels.hazard import GaussianField


def series(probs) -> bool:
    """Whether this BN (tbnpy's probability objects) needs cliques_series.py."""
    for p in probs.values():
        if isinstance(p, GaussianField) and any(not v.name.startswith("SA") for v in p.childs):
            return True                                               # a field of PGV, PGA, ...
    for k, p in probs.items():
        if k.startswith("EdgeOpen::") and (len(p.parents) != 1 or not p.parents[0].name.startswith("DamageState")):
            return True                                               # an edge of several components, or of a road
    return False


def module(t):
    """The module whose tree t is."""
    return cliques_series if t.get("series") else cliques


def tree(probs, corr=None):
    """The clique tree's pieces, by the module that takes this BN."""
    return (cliques_series if series(probs) else cliques).tree(probs, corr)


def sample(t, ev, **kwargs):
    """Rows drawn from P(everything | ev), by the module whose tree t is."""
    return module(t).sample(t, ev, **kwargs)


def method(t) -> str:
    """How the rows were drawn, for a result's "inference" entry."""
    return f"clique tree (src/inference/{module(t).__name__.rsplit('.', 1)[1]}.py)"


def describe(nodes=None):
    """The message passing for the Clique tree tab, for a BN of these nodes ({"id", "quantity", "function",
    "parents"}, as compiled.nodes() gives them): cliques.describe() unless the BN has series edges or a field
    other than SA; with no nodes, cliques.describe()."""
    if nodes is not None:
        lay = cliques_series.layout(nodes)
        if lay["kinds"] or lay["fields"] != ["SA"]:
            return cliques_series.describe(**lay)
    return cliques.describe()


def drawing(nodes, tables=None):
    """The clique tree as the Clique tree tab draws it, for a BN of these nodes: by the module that samples it, as
    describe() picks it. tables: each edge's table (server.edge_tables), which orders a series edge's components."""
    lay = cliques_series.layout(nodes)
    if lay["kinds"] or lay["fields"] != ["SA"]:
        return cliques_series.drawing(nodes, tables)
    return cliques.drawing(nodes)
