"""Compile an assembly whose network's edges carry components in series (Kahramanmaraş) into a BN.

This is bn_builder.py extended, as src/inference/cliques_series.py is cliques.py extended. bn_builder.py
builds the networks of one bridge per edge under SA (toy-ca, EMA, Hebei) as it did, and builders.py sends an
assembly here only when bn_builder.py cannot take it: it binds a function bn_builder.py has no kernel for, or
an edge carries anything but one bridge. The assembly compiler, the spec writer and the generated model's
files are bn_builder.py's; this module adds what only these networks need:

- the ground-failure kernels' arguments: PGD from liquefaction, landslide and fault rupture, and whether the
  road stays open at each component (fragility.road);
- an edge's series table over its bridges and ground-failure components (edge.open), its rows in road order
  (chainage_km): row k says components 1..k-1 are open and k is closed, the last row that all are open;
- connectivity systems (system.reachable), whose rsr rules are stored per Support row, with no threshold;
- for rsr on a network this large: a proposal to search from (uniform_proposal, marginal_proposal), and
  resuming an extraction that stopped (extract_system_rules(resume=True)).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from . import bn_builder as base
from .bn_builder import (_GENERATED_MODEL, _SAME, RSR_MAX_ROUNDS, RSR_UNKNOWN_TARGET, Compiled, _network_inline,
                         _network_to_file, _rules_meta, _tbn_fingerprint, compile_path, compile_tbn, model_dir,
                         system_nodes)
from .load import RegistryError, assembly_path

# The functions bn_builder.py builds; an assembly binding any other is this module's.
BASE_FUNCTIONS = {"hazard.gmm", "hazard.shakemap", "observation.intensity", "fragility.bridge", "edge.open",
                  "system.travel-time-to-nearest"}


def takes(compiled: Compiled) -> bool:
    """Whether this assembly needs this module: it binds a function bn_builder.py has no kernel for, or one
    of its edges carries anything but one bridge (several components, or a ground-failure one)."""
    if any(b.function["id"] not in BASE_FUNCTIONS for b in compiled.bindings):
        return True
    by = {n["id"]: n for n in compiled.nodes()}
    return any(n["function"] == "edge.open" and
               (len(n["parents"]) != 1 or by[n["parents"][0]]["function"] != "fragility.bridge")
               for n in by.values())


# ---- the kernels' arguments ----------------------------------------------------------------------------

def _edge_arguments(rows, params, tables, parent_rows, states):
    """EdgeOpen: the damage states its closure policy keeps a bridge open in, from its parameter set; and,
    when ground-failure components lie on the edge, how many of its parents are bridges; and, where every
    component has a chainage_km, their road order, which orders the rows of its series table."""
    (p,) = params
    ground = sum(len(parent_rows[port][0]) for port in parent_rows if port != "damage")
    own = {"n_bridges": len(parent_rows["damage"][0])} if ground else {}
    ids = [pid for port in parent_rows for pid in parent_rows[port][0]]      # the parents, in port order
    on = {r["id"]: r for name in ("Bridge", "GroundFailure") for r in tables.get(name, [])}
    chainage = [on.get(pid, {}).get("chainage_km") for pid in ids]
    if len(ids) > 1 and all(c is not None for c in chainage):
        order = sorted(range(len(ids)), key=lambda i: (chainage[i], i))
        if order != list(range(len(ids))):
            own["order"] = order
    return {"usable_states": p["usable_states"]}, own


def _event_parameters(p):
    return {k: p.get(k) for k in ("event_id", "source", "version")}


def _liquefaction_pgd_arguments(rows, params, tables, parent_rows, states):
    """LiquefactionPGD: the scenario, Zhu's model and the Hazus tables inline; the site term, length and
    class the row's own."""
    (row,), (p,) = rows, params
    return ({**_event_parameters(p), "model": p["model"]},
            {"x0": row["x0"], "liq_class": row["liq_class"], "length_km": row.get("length_km", 0.0),
             "dw_km": row.get("dw_km")})


def _landslide_pgd_arguments(rows, params, tables, parent_rows, states):
    """LandslidePGD: Jessee's model and Jibson's coefficients inline; the site term, slope, critical
    acceleration and length the row's own."""
    (row,), (p,) = rows, params
    return {"model": p["model"]}, {"x0": row["x0"], "slope_deg": row["slope_deg"], "ac_g": row["ac_g"],
                                   "length_km": row.get("length_km", 0.0)}


def _fault_pgd_arguments(rows, params, tables, parent_rows, states):
    """FaultPGD: the scenario, the rupture and displacement relations inline; the crossing's place, its
    fault's slip type and catalogue id (which segment's measured offset applies) the row's own."""
    (row,), (p,) = rows, params
    return ({**_event_parameters(p), "model": p["model"]},
            {"lat": row["lat"], "lon": row["lon"], "slip_type": row.get("slip_type", ""),
             "fault_id": row.get("fault_id", "")})


def _road_arguments(rows, params, tables, parent_rows, states):
    """RoadFragility: the curve's closing state, median and beta, from its parameter set."""
    (p,) = params
    return ({"median_m": p["median_m"], "beta": p["beta"], "closing_state": p["closing_state"],
             **({"window_h": p["window_h"]} if "window_h" in p else {})}, {})


def _reachable_arguments(rows, params, tables, parent_rows, states):
    """Reachable: no parameter set; the origin, its source (None: any key destination), the random edges in
    parent order, the network (every edge, those that cannot close included)."""
    (row,) = rows
    return {}, {"origin": row["origin"], "destination": row.get("source"), "edge_ids": parent_rows["open"][0],
                "network": {name: table for name, table in tables.items()}}


# The constructor arguments of each kernel, as bn_builder.KERNEL_ARGUMENTS has them: its own, with an edge's
# series table in road order, the ground-failure kernels and connectivity.
KERNEL_ARGUMENTS = {
    **base.KERNEL_ARGUMENTS,
    "src.builder.kernels.edge:EdgeOpen": _edge_arguments,
    "src.builder.kernels.ground:LiquefactionPGD": _liquefaction_pgd_arguments,
    "src.builder.kernels.ground:LandslidePGD": _landslide_pgd_arguments,
    "src.builder.kernels.ground:FaultPGD": _fault_pgd_arguments,
    "src.builder.kernels.ground:RoadFragility": _road_arguments,
    "src.builder.kernels.system:Reachable": _reachable_arguments,
}


def tbn_data(compiled: Compiled, bindings: set[int] | None = None) -> dict[str, Any]:
    """The model as a tbnpy spec (bn_builder.tbn_data), with these kernels' arguments."""
    return base.tbn_data(compiled, bindings, kernels=KERNEL_ARGUMENTS)


# ---- rsr rules -----------------------------------------------------------------------------------------

# A proposal probability of 0 or 1 never draws that edge's other state, so rsr could not find the rules
# (failure modes) that need it: the marginal proposal is clipped to this range.
RSR_PROPOSAL_CLIP = (0.03, 0.97)


def _reachable(node: dict[str, Any]) -> bool:
    return node["metadata"]["function"] == "system.reachable"


def ruled_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The system-state nodes stored rsr rules can classify: those judged at a travel-time threshold, and
    connectivity (system.reachable), which has none."""
    return [node for node in system_nodes(data) if "target_add_h" in node.get("kwargs", {}) or _reachable(node)]


def rules_key(node: dict[str, Any]) -> str:
    """Whose rules: the origin of a travel-time system; the Support row of a reachability one."""
    return node["childs"][0].split("::", 1)[1] if _reachable(node) else node["kwargs"]["origin"]


def rules_threshold(node: dict[str, Any]) -> float | None:
    """The system event's threshold a system-state node is judged by: its target_add_h, in hours; None for
    connectivity, which has no threshold."""
    return None if _reachable(node) else float(node["kwargs"]["target_add_h"])


def rules_fingerprint(node: dict[str, Any]) -> str:
    """What a system's rules depend on, apart from the threshold: the graph, the destinations, the
    analysis, the origin (and, for connectivity, its source) and the edge order. Not the hazard: rules
    extracted once serve every scenario. The threshold is stored beside the rules, since some rules hold
    across thresholds. (The snapshot keeps the form it had before model.json became a tbnpy spec, so stored
    rules stay valid.)"""
    kwargs = node["kwargs"]
    snapshot = {"function": node["metadata"]["function"], "tables": kwargs["network"],
                "events": [{"analysis": node["metadata"].get("analysis")}],
                "origin": kwargs["origin"], "edges": kwargs["edge_ids"]}
    if _reachable(node):
        snapshot["destination"] = kwargs.get("destination")
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def rules_folder(directory: Path, origin: str, threshold: float | None) -> Path:
    """rsr/<origin>/target_add_h=<value>/: one set of rules per origin and threshold; rsr/<row>/reachable/
    for a connectivity system, which has no threshold."""
    if threshold is None:
        return Path(directory) / "rsr" / origin / "reachable"
    return Path(directory) / "rsr" / origin / f"target_add_h={threshold:g}"


def stored_rules(directory: Path, node: dict[str, Any]) -> list[tuple[float | None, Path]]:
    """(threshold, folder) for every set of rules stored for this system's definition, by threshold."""
    fingerprint, found = rules_fingerprint(node), []
    if _reachable(node):
        folder = rules_folder(directory, rules_key(node), None)
        meta = _rules_meta(folder / "refs_up_1.pt")
        if meta.get("rules_sha256") == fingerprint and (folder / "refs_up_1.pt").is_file():
            found.append((None, folder))
        return found
    for folder in (Path(directory) / "rsr" / node["kwargs"]["origin"]).glob("target_add_h=*"):
        meta = _rules_meta(folder / "refs_up_1.pt")
        if meta.get("rules_sha256") == fingerprint and (folder / "refs_up_1.pt").is_file():
            found.append((float(meta["target_add_h"]), folder))
    return sorted(found)


def choose_rules(stored: list[tuple[float, Path]], threshold: float) -> dict[str, Any] | None:
    """The rules that hold at ``threshold``, from those stored at any threshold.

    The state is monotone in the threshold: reaching a destination within t_s of the
    intact time means reaching it within any t >= t_s, and failing within t_f means
    failing within any t <= t_f. So survival rules hold when extracted at the nearest
    t_s <= t, and failure rules at the nearest t_f >= t. Only an exact match has both.
    """
    if threshold is None:                          # connectivity: one set, exact or none
        if not stored:
            return None
        folder = stored[0][1]
        return {"upper": folder / "refs_up_1.pt", "upper_at": None, "lower": folder / "refs_low_0.pt", "lower_at": None,
                "exact": True}
    survival = [(t, f) for t, f in stored if t <= threshold + _SAME]
    failure = [(t, f) for t, f in stored if t >= threshold - _SAME]
    upper, lower = (survival[-1] if survival else None), (failure[0] if failure else None)
    if upper is None and lower is None:
        return None
    return {"upper": upper[1] / "refs_up_1.pt" if upper else None, "upper_at": upper[0] if upper else None,
            "lower": lower[1] / "refs_low_0.pt" if lower else None, "lower_at": lower[0] if lower else None}


def rules_source(ref: dict[str, Any] | None, threshold: float) -> str:
    """How a system state's rules were found: in words, for notes and the page."""
    if ref is None:
        return "no stored rules"
    if ref.get("exact"):
        return "exact"
    up, low = ref.get("upper_at"), ref.get("lower_at")
    if up is not None and low is not None and abs(up - threshold) < _SAME and abs(low - threshold) < _SAME:
        return "exact"
    parts = ([f"survival rules from target_add_h={up:g}"] if up is not None else []) + \
            ([f"failure rules from target_add_h={low:g}"] if low is not None else [])
    return " and ".join(parts)


def attach_rules(data: dict[str, Any], directory: Path) -> int:
    """Give each system-state node the stored rules that hold at its threshold, as its ``rules``
    argument: file names relative to ``directory``. Returns how many origins have any."""
    found = 0
    for node in ruled_nodes(data):
        ref = choose_rules(stored_rules(directory, node), rules_threshold(node))
        if ref:
            ref = {k: (v.relative_to(directory).as_posix() if isinstance(v, Path) else v) for k, v in ref.items()}
        node["kwargs"]["rules"] = ref
        found += ref is not None
    return found


def _check_reference(directory: Path, node: dict[str, Any], ref: dict[str, Any]) -> None:
    threshold, fingerprint = rules_threshold(node), rules_fingerprint(node)
    if threshold is None:
        for side in ("upper", "lower"):
            if ref.get(side) and _rules_meta(Path(directory) / ref[side]).get("rules_sha256") != fingerprint:
                raise ValueError("RSR references belong to a different system definition; regenerate")
        return
    for side, holds in (("upper", lambda t: t <= threshold + _SAME), ("lower", lambda t: t >= threshold - _SAME)):
        if ref.get(side) is None:
            continue
        meta = _rules_meta(Path(directory) / ref[side])
        if meta.get("rules_sha256") != fingerprint or not holds(float(meta.get("target_add_h", "nan"))):
            raise ValueError("RSR references belong to a different system definition; regenerate")


def build_tbn(compiled: Compiled, bindings: set[int] | None = None, device: str = "cpu",
              rules: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """tbnpy's two dictionaries, variables and probabilities, for a compiled assembly.

    With ``rules``, system states reuse the rsr rules stored with the assembly's
    generated model that hold at the query's threshold (see ``choose_rules``);
    whatever they leave unclassified is evaluated by ndtools.
    """
    data = tbn_data(compiled, bindings)
    directory = None
    if rules:
        try:
            candidate = model_dir(assembly_path(compiled.assembly["id"]))
        except RegistryError:
            candidate = None
        if candidate is not None and attach_rules(data, candidate):
            directory = candidate
    return compile_tbn(data, device=device, directory=directory)


def extract_system_rules(compiled: Compiled, directory: Path, origins: set[str] | None = None, *,
                  seed: int = 20261022, log: Callable[[str], None] = print,
                  proposal: dict[str, Any] | None = None, unknown_target: float = RSR_UNKNOWN_TARGET,
                  max_rounds: int = RSR_MAX_ROUNDS, check_samples: int | None = None,
                  resume: bool = False) -> list[str]:
    """Run rsr for each system (or those whose key is in ``origins``) at its threshold, and store the
    rules in rules_folder(). Returns the keys extracted.

    resume: start from the rules already stored in each system's folder (finished or interrupted) rather
    than from none. max_rounds then caps the rounds of all runs together; each earlier run's metrics are
    kept as metrics_run<k>.json and model.json lists every run.

    proposal: the distribution rsr's search draws edge states from, {"p_open": {edge id: P(open)}, ...};
    None for the independent 50/50 one. Only the search uses it: the rules are exact either way.
    check_samples: the samples rsr draws to look for unclassified states and estimate their probability
    (e.g. 1e6 for a 1e-5 target); None for the quick default of 4096, searched at most twice per round."""
    import random
    import torch
    from rsr.rsr import run_ref_extraction_by_mcs

    data = tbn_data(compiled)
    _, probs = compile_tbn(data)
    done = []
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            random.seed(seed)
            for node in ruled_nodes(data):
                origin, edges, threshold = rules_key(node), node["kwargs"]["edge_ids"], rules_threshold(node)
                if origins is not None and origin not in origins:
                    continue
                folder = rules_folder(directory, origin, threshold)
                folder.mkdir(parents=True, exist_ok=True)
                start, runs = _resume_point(folder) if resume else ({}, [])
                rounds_left = max_rounds - sum(r["rounds"] for r in runs)
                if rounds_left <= 0:
                    log(f"{origin}: earlier runs used all {max_rounds} rounds; rules kept as they are")
                    continue
                # RSR appends metrics; each run starts a fresh record.
                (folder / "metrics.json").write_text("", encoding="utf-8")
                log(f"Extracting system rules for {origin}" +
                    (f" at target_add_h={threshold:g}..." if threshold is not None else " (connectivity)..."))
                evaluate = probs[node["name"]].system_function()

                def binary_system(comps, evaluate=evaluate):
                    # The system value rsr sees is the binary state itself; travel times
                    # (infinite when cut off) stay internal to ndtools' judgement.
                    _, state, _ = evaluate(comps)
                    return state, state, None

                # The independent 50/50 proposal explores topology; it does not replace
                # the BN's component distribution or its hazard-induced dependence. A proposal of each
                # edge's marginal P(open) under a scenario focuses the search on the states that matter.
                if proposal is None:
                    search = torch.full((len(edges), 2), 0.5, dtype=torch.float64)
                else:
                    p_open = torch.tensor([float(proposal["p_open"][e]) for e in edges], dtype=torch.float64)
                    search = torch.stack([1 - p_open, p_open], dim=1)
                run_ref_extraction_by_mcs(
                    sfun=binary_system, probs=search,
                    row_names=edges, n_state=2, sys_upper_st=1, samples_store=False, **start,
                    unk_prob_thres=unknown_target, unk_prob_opt="abs", max_rounds=rounds_left,
                    prob_update_every=10 if check_samples is None else 100, save_every=128,
                    n_sample=4096 if check_samples is None else int(check_samples),
                    sample_batch_size=4096, max_search_loops=2 if check_samples is None else 0,
                    ref_update_verbose=False, output_dir=str(folder))
                this_run = {"proposal": ("independent Bernoulli(0.5)" if proposal is None else
                                         "p_open in rsr/proposal.json"),
                            "rounds": _last_round(folder / "metrics.json"),
                            "started_from": "stored rules" if start else "no rules",
                            "seed": seed, "metrics": "metrics.json"}
                (folder / "model.json").write_text(json.dumps({
                    "rules_sha256": rules_fingerprint(node), "target_add_h": threshold,
                    "origin": origin, "edge_order": edges, "seed": seed,
                    "unknown_target": unknown_target, "max_rounds": max_rounds,
                    "check_samples": 4096 if check_samples is None else int(check_samples),
                    "proposal": ("independent Bernoulli(0.5), for rule search only" if proposal is None else
                                 {k: v for k, v in proposal.items() if k != "p_open"} | {"file": "rsr/proposal.json"}),
                    "system_value": "binary state",
                    "unclassified": "evaluate directly" if threshold is None else "evaluate with ndtools",
                    "runs": runs + [this_run],
                }, indent=2) + "\n", encoding="utf-8")
                done.append(origin)
    finally:
        random.setstate(python_state)
    return done


def _last_round(metrics: Path) -> int:
    lines = [x for x in metrics.read_text(encoding="utf-8").splitlines() if x.strip()] if metrics.is_file() else []
    return int(json.loads(lines[-1])["round"]) if lines else 0


def _resume_point(folder: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The rules stored in folder, as rsr's starting references, and the runs that found them. The
    last run's metrics move to metrics_run<k>.json so the next run starts its own record."""
    import torch

    files = {k: folder / f for k, f in (("refs_upper", "refs_up_1.json"), ("refs_lower", "refs_low_0.json"),
                                       ("refs_mat_upper", "refs_up_1.pt"), ("refs_mat_lower", "refs_low_0.pt"))}
    if not all(f.is_file() for f in files.values()):
        return {}, []
    start = {k: json.loads(f.read_text(encoding="utf-8")) if f.suffix == ".json" else torch.load(f)
             for k, f in files.items()}
    for side in ("upper", "lower"):
        if len(start[f"refs_{side}"]) != len(start[f"refs_mat_{side}"]):
            raise ValueError(f"{folder}: the {side} rules' .json and .pt files hold different numbers of rules")
    meta = json.loads((folder / "model.json").read_text(encoding="utf-8")) if (folder / "model.json").is_file() else {}
    # A model.json is written when a run finishes, so metrics.json is its run's record; with no model.json
    # (or one from an earlier run) metrics.json is the record of a run interrupted before it finished.
    runs = meta.get("runs") or ([{"proposal": meta["proposal"], "rounds": _last_round(folder / "metrics.json"),
                                  "started_from": "no rules", "metrics": "metrics.json"}] if meta else [])
    runs = [dict(r) for r in runs]
    if runs and runs[-1].get("metrics") == "metrics.json":
        runs[-1]["metrics"] = f"metrics_run{len(runs)}.json"
    elif _last_round(folder / "metrics.json"):
        runs.append({"proposal": "unrecorded (interrupted)", "rounds": _last_round(folder / "metrics.json"),
                     "started_from": "unrecorded", "metrics": f"metrics_run{len(runs) + 1}.json"})
    if (folder / "metrics.json").is_file() and runs:
        (folder / "metrics.json").replace(folder / runs[-1]["metrics"])
    return start, runs


def uniform_proposal(compiled: Compiled, p_open: float) -> dict[str, Any]:
    """Every random edge open with probability p_open, independently: the proposal rsr's rule search draws from."""
    if not 0.0 < p_open < 1.0:
        raise ValueError(f"p_open must be strictly between 0 and 1 (both states drawn), got {p_open}")
    data = tbn_data(compiled)
    edges = sorted({e for node in ruled_nodes(data) for e in node["kwargs"]["edge_ids"]})
    return {"kind": f"every random edge open with probability {p_open:g}, independently", "p_open_all": p_open,
            "p_open": {e: p_open for e in edges}}


def marginal_proposal(compiled: Compiled, n_samples: int = 10_000, seed: int = 20261022,
                      clip: tuple[float, float] = RSR_PROPOSAL_CLIP) -> dict[str, Any]:
    """Each random edge's marginal P(open) under the assembly's own scenario (its bound ShakeMap or GMM), from
    n_samples forward draws of the BN by the clique tree, clipped to ``clip``: the proposal rsr's rule search
    can draw from. The unclipped marginals are kept as p_open_marginal."""
    import torch
    from src.inference import trees

    data = tbn_data(compiled)
    _, probs = compile_tbn(data)
    t = trees.tree(probs)
    n_chain = 100
    rows, _ = trees.sample(t, {}, n_chain=n_chain, n_keep=max(1, n_samples // n_chain), seed=seed)
    p_open = {c["E"].split("::", 1)[1]: float(rows[c["E"]].double().mean()) for c in t["edges"]}
    scenario = {b["function"]: {k: b["parameters"].get(k) for k in ("event_id", "version", "imt") if k in b.get("parameters", {})}
                for b in compiled.assembly["bindings"] if b["function"] in ("hazard.shakemap", "hazard.gmm")}
    return clip_proposal({"kind": "marginal P(open) of each random edge under the assembly's scenario",
                          "scenario": scenario, "n_samples": n_chain * max(1, n_samples // n_chain), "seed": seed,
                          "p_open": p_open}, clip)


def clip_proposal(proposal: dict[str, Any], clip: tuple[float, float] = RSR_PROPOSAL_CLIP) -> dict[str, Any]:
    """The proposal with P(open) clipped to clip, so that rsr draws both states of every edge."""
    lo, hi = clip
    raw = proposal.get("p_open_marginal", proposal["p_open"])
    return proposal | {"clip": [lo, hi], "clip_reason": "a proposal probability of 0 or 1 never draws that edge's "
                       "other state, so rsr could not find the rules (failure modes) that need it",
                       "p_open_marginal": raw, "p_open": {e: min(max(p, lo), hi) for e, p in raw.items()}}


def generate_tbn(path: Path | str, *, extract_rules: bool = True, seed: int = 20261022,
                 proposal: str | float | None = None, unknown_target: float = RSR_UNKNOWN_TARGET,
                 max_rounds: int = RSR_MAX_ROUNDS, check_samples: int | None = None) -> Path:
    """Write a runnable tbnpy model beneath the assembly's example folder, with rules at its threshold.
    proposal: None for rsr's 50/50 search; a number for every edge open with that probability (uniform_proposal);
    "marginal" to search from each edge's marginal P(open) under the assembly's own scenario
    (marginal_proposal). Either is written to rsr/proposal.json for audit."""
    import importlib.metadata

    path = Path(path).resolve()
    compiled = compile_path(path)
    data = tbn_data(compiled)
    compile_tbn(data)                       # every object builds before anything is written
    fingerprint = _tbn_fingerprint(data)
    out = model_dir(path)
    out.mkdir(parents=True, exist_ok=True)
    data["source_sha256"] = fingerprint
    data["versions"] = {n: importlib.metadata.version(n) for n in ("tbnpy", "torch", "ndtools-duco", "rsr-duco")}
    if extract_rules:
        search = None
        if proposal == "marginal":
            search = marginal_proposal(compiled, seed=seed)
        elif isinstance(proposal, (int, float)) and not isinstance(proposal, bool):
            search = uniform_proposal(compiled, float(proposal))
        elif proposal is not None:
            raise ValueError(f"unknown proposal {proposal!r}; use None, a probability or 'marginal'")
        if search is not None:
            (out / "rsr").mkdir(parents=True, exist_ok=True)
            (out / "rsr" / "proposal.json").write_text(json.dumps(search, indent=2) + "\n", encoding="utf-8")
        extract_system_rules(compiled, out, seed=seed, proposal=search, unknown_target=unknown_target,
                             max_rounds=max_rounds, check_samples=check_samples)
    attach_rules(data, out)                 # the stored rules that hold, extracted now or before
    written = _network_to_file(data, out)
    (out / "model.json").write_text(json.dumps(written, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    script = _GENERATED_MODEL.replace("python -m src build toy-ca/demo", f"python -m src build {compiled.assembly['id']}")
    (out / "model.py").write_text(script, encoding="utf-8")
    return out


def load_generated(directory: Path | str, device: str = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    """A generated model's (varis, probs), after checking that model.json and its stored rules still
    belong together. (tbnpy.compile.compile_file loads the same file without the checks.)"""
    directory = Path(directory)
    data = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    inline = _network_inline(data, directory)
    if "source_sha256" in data and _tbn_fingerprint(inline) != data["source_sha256"]:
        raise ValueError("Data belong to a different generated model; regenerate")
    for node in system_nodes(inline):
        if node["kwargs"].get("rules"):
            _check_reference(directory, node, node["kwargs"]["rules"])
    return compile_tbn(data, device=device, directory=directory)
