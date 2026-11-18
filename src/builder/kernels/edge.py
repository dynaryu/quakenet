"""Bridge damage and ground-failure road states -> edge state: a series system, as a composite-state CPT."""

from __future__ import annotations

import torch
from tbnpy.cpt import Cpt, _state_label

from . import Probability


class EdgeOpen(Probability, Cpt):
    """Whether an edge is open, given the damage state of each bridge on it and whether the road stays
    open at each ground-failure component on it: the components act in series.

        EdgeOpen = True if every bridge on the edge is in one of usable_states and every ground-failure
                   component on it leaves the road open, else False; True for an edge with neither

    The CPT is a tbnpy Cpt whose event matrix C uses composite states (Byun & Song 2021): with the n
    components in road order X1 ... Xn, row i (i = 1 ... n) is "X1 ... X(i-1) open, Xi closed, the rest
    any state" -> False, and row n + 1 is "every component open" -> True; p is 1 on every row. The rows
    are disjoint and cover every combination of parent states (branch and bound), so n + 1 rows stand
    for the full table. A bridge is open in usable_states and closed in the others; a ground-failure
    component's RoadOpen is open in True and closed in False.
    """

    __hash__ = object.__hash__          # Cpt's compare its tables; a kernel is one object
    __eq__ = object.__eq__

    # The log probabilities tbnpy.inference stores on the object after sampling, (N,) as every kernel's:
    # Cpt's own setter would make them a column.
    @property
    def ps(self):
        return self._ps

    @ps.setter
    def ps(self, value):
        self._ps = torch.empty(0, dtype=torch.float64) if isinstance(value, list) and not value else value

    def __init__(self, childs, parents, usable_states, n_bridges=None, order=None, device="cpu"):
        """
        childs        : [Variable EdgeOpen], values ["False", "True"] (state indices 0, 1)
        parents       : [Variable DS] of each bridge on the edge, values = the damage states, then
                        [Variable RoadOpen] of each ground-failure component on it, values ["False", "True"];
                        none for an edge with neither
        usable_states : the damage states (labels) in which a bridge stays open: the closure policy
        n_bridges     : how many of the parents are bridges, the rest being ground-failure components;
                        None when every parent is a bridge
        order         : the parents' indices in road order, from the edge's source: the order of C's rows;
                        None for the parents' own order
        """
        Probability.__init__(self, childs, parents, device)
        assert len(childs) == 1, "EdgeOpen: one EdgeOpen child"
        self.usable_states = list(usable_states)
        self.n_bridges = len(parents) if n_bridges is None else int(n_bridges)
        bridges = self.parents[:self.n_bridges]

        states = list(bridges[0].values) if bridges else []
        if any(list(p.values) != states for p in bridges):
            raise ValueError("every bridge on an edge must be on the same damage scale")
        unknown = sorted(set(self.usable_states) - set(states)) if bridges else []
        if unknown:
            raise ValueError(f"usable states {unknown} are not damage states: {states}")
        # Open (1) or not (0) for each DS index: what the clique-tree sampler reads.
        self.open_if = self.tensor([float(s in self.usable_states) for s in states])      # (K,)

        n = len(self.parents)
        self.order = list(range(n)) if order is None else [int(i) for i in order]
        if sorted(self.order) != list(range(n)):
            raise ValueError(f"order {self.order} is not a permutation of the {n} parents")
        Cpt.__init__(self, childs=self.childs, parents=self.parents, C=self._series_table(), p=[1.0] * self._n_rows(),
                     device=device)
        # Which basic states each row's composite state admits, per parent: (n_rows, K_j) each.
        self._admits = [var.get_Cst_to_Cbin(self.C[:, 1 + j]).bool() for j, var in enumerate(self.parents)]

    # ------------------------------------------------------------------
    # The table
    def _open_set(self, j: int) -> set[int]:
        return set(open_states(list(self.parents[j].values), j < self.n_bridges, self.usable_states))

    def _closed_set(self, j: int) -> set[int]:
        return set(range(len(self.parents[j].values))) - self._open_set(j)

    def _n_rows(self) -> int:
        return len(self._cells())

    def _cells(self) -> list[list[str]]:
        return series_rows(len(self.parents), self.order, [bool(self._closed_set(j)) for j in range(len(self.parents))])

    def _series_table(self) -> list[list[int]]:
        """C, with the cells of series_rows as tbnpy's composite indices (its B)."""
        state = {"any": lambda j: set(range(len(self.parents[j].values))), "open": self._open_set,
                 "closed": self._closed_set}
        return [[0 if child == "False" else 1] + [self.parents[j].get_state(state[cell](j)) for j, cell in enumerate(cells)]
                for child, *cells in self._cells()]

    # ------------------------------------------------------------------
    # Readings
    def form(self):
        if not self.parents:
            return "True: the edge has no bridge and no ground-failure component"
        n = len(self.parents)
        return (f"series system of {n} component{'s' if n > 1 else ''} in road order: "
                f"table of {self._n_rows()} rows (composite states){self.parameter_sets()}")

    def table(self) -> dict:
        """The event table for a reader: the edge, then its components in road order; each cell a state
        label, a set of them, or "any"; p of each row."""
        cols = [0] + [1 + j for j in self.order]
        variables = self.childs + self.parents
        full = {c: variables[c].get_state(set(range(len(variables[c].values)))) for c in cols}
        rows = [["any" if c and int(row[c]) == full[c] else _state_label(variables[c], int(row[c])) for c in cols]
                for row in self.C.tolist()]
        return {"columns": [variables[c].name for c in cols], "rows": rows,
                "p": [float(q) for q in self.p.reshape(-1).tolist()]}

    def form_details(self):
        """The whole event table, one row per line."""
        t = self.table()
        header = t["columns"] + ["p"]
        rows = [r + [f"{q:g}"] for r, q in zip(t["rows"], t["p"])]
        widths = [max([len(h)] + [len(r[j]) for r in rows]) for j, h in enumerate(header)]
        lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)).rstrip()]
        lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
        return lines + Probability.form_details(self)

    # ------------------------------------------------------------------
    def evaluate(self, Cs_pars):
        """
        Cs_pars : (N, n_bridges + n_ground) DS index of each bridge, then RoadOpen of each component
        Returns (N, 1) EdgeOpen index, read from the one row of C each sample falls in: 1 (True) open,
        0 (False) closed
        """
        if not self.parents:
            return torch.ones((len(Cs_pars), 1), dtype=torch.float64, device=self.device)
        x = Cs_pars.to(torch.float64)
        for j, var in enumerate(self.parents):
            col = x[:, j]
            if ((col != col.floor()) | (col < 0) | (col >= len(var.values))).any():
                raise ValueError(f"invalid state index for {var.name}")
        x = x.long()
        match = torch.ones((len(x), len(self.C)), dtype=torch.bool, device=self.device)
        for j, admits in enumerate(self._admits):
            match &= admits.to(self.device)[:, x[:, j]].T                      # (N, n_rows)
        hits = match.sum(dim=1)
        if (hits != 1).any():
            raise ValueError("EdgeOpen's table: a parent state falls in no row or in several")
        row = match.float().argmax(dim=1)
        return self.C[:, 0].to(self.device)[row].to(torch.float64)[:, None]


def open_states(values: list[str], bridge: bool, usable_states: list[str]) -> list[int]:
    """The states a component is open in: a bridge's usable states, or a road state's True."""
    return [i for i, v in enumerate(values) if (v in usable_states if bridge else v == "True")]


def series_rows(n: int, order: list[int], can_close: list[bool]) -> list[list[str]]:
    """The rows of a series system's table, each [edge state, then each parent's cell, in parent order]: one
    row per component that can close, taken in road order (the components before it "open", it "closed",
    those after it "any") with the edge "False", then every component "open" with the edge "True"."""
    rows = []
    for i, j in enumerate(order):
        if not can_close[j]:
            continue
        cells = ["any"] * n
        for k in order[:i]:
            cells[k] = "open"
        cells[j] = "closed"
        rows.append(["False"] + cells)
    return rows + [["True"] + ["open"] * n]


def series_table(child: str, parents: list[str], values: dict[str, list[str]], usable_states: list[str],
                 n_bridges: int | None = None, order: list[int] | None = None) -> dict:
    """An edge's table for a reader, from its spec alone (what EdgeOpen builds C from): the columns in road
    order, each row's cells ("open", "closed", "any"), and the states "open" and "closed" stand for."""
    n = len(parents)
    nb = n if n_bridges is None else n_bridges
    order = list(range(n)) if order is None else list(order)
    opens = [open_states(values[p], j < nb, usable_states) for j, p in enumerate(parents)]
    rows = series_rows(n, order, [len(o) < len(values[p]) for o, p in zip(opens, parents)])
    return {"columns": [child] + [parents[j] for j in order],
            "rows": [[r[0]] + [r[1 + j] for j in order] for r in rows], "p": [1.0] * len(rows),
            "states": {p: {"open": [values[p][i] for i in o],
                           "closed": [v for i, v in enumerate(values[p]) if i not in o]} for p, o in zip(parents, opens)}}
