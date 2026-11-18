"""Which builder an assembly is built by.

bn_builder.py builds the networks of one bridge per edge under SA (toy-ca, EMA, Hebei). bn_builder_series.py
builds the networks whose edges carry components in series, bridges and ground-failure components
(Kahramanmaraş). An assembly goes to bn_builder_series.py only when bn_builder.py cannot take it: it binds a
function bn_builder.py has no kernel for, or an edge carries anything but one bridge. So a network of one bridge
per edge is built exactly as bn_builder.py builds it, untouched by the series machinery: its BN is the one
src/inference/cliques.py samples, as src/inference/trees.py sends it there.

Both share bn_builder.py's assembly compiler (compile_assembly, compile_path) and its spec writer; what differs
is each kernel's arguments and the systems' rsr rules, which the functions here take from the right module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import bn_builder, bn_builder_series
from .bn_builder import Compiled, compile_path


def series(compiled: Compiled) -> bool:
    """Whether this assembly needs bn_builder_series.py."""
    return bn_builder_series.takes(compiled)


def module(compiled: Compiled):
    """The module that builds this assembly."""
    return bn_builder_series if series(compiled) else bn_builder


def _of_spec(data: dict[str, Any]):
    """The module a spec (tbn_data) was written by: the series one if it has a node bn_builder.py does not build."""
    functions = {node["metadata"]["function"] for node in data["nodes"]}
    if functions - bn_builder_series.BASE_FUNCTIONS:
        return bn_builder_series
    edges = [node for node in data["nodes"] if node["metadata"]["function"] == "edge.open"]
    return bn_builder_series if any(len(node["parents"]) != 1 for node in edges) else bn_builder


def tbn_data(compiled: Compiled, bindings: set[int] | None = None) -> dict[str, Any]:
    return module(compiled).tbn_data(compiled, bindings)


def build_tbn(compiled: Compiled, bindings: set[int] | None = None, device: str = "cpu",
              rules: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    return module(compiled).build_tbn(compiled, bindings, device=device, rules=rules)


def attach_rules(data: dict[str, Any], directory: Path) -> int:
    return _of_spec(data).attach_rules(data, directory)


def rules_status(data: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    """Per system with rsr rules: its key (the origin, or the Support row of a reachability system), its threshold
    (None for connectivity), and whether it has rules of its own stored at that threshold (``exact``). Attaches the
    stored rules to ``data`` as attach_rules does."""
    builder = _of_spec(data)
    attach_rules(data, directory)
    nodes = builder.ruled_nodes(data) if builder is bn_builder_series else builder.system_nodes(data)
    key = bn_builder_series.rules_key if builder is bn_builder_series else (lambda node: node["kwargs"]["origin"])
    out = []
    for node in nodes:
        threshold = builder.rules_threshold(node)
        out.append({"origin": key(node), "target_add_h": threshold,
                    "exact": builder.rules_source(node["kwargs"].get("rules"), threshold) == "exact"})
    return out


def extract_system_rules(compiled: Compiled, directory: Path, origins: set[str] | None = None, **kwargs) -> list[str]:
    """rsr's rules for each system; the options beyond bn_builder.extract_system_rules' (a proposal, resuming)
    are bn_builder_series.py's. With no options, an assembly's own rsr recipe ("rsr": its proposal, target,
    rounds and check samples) is followed, as python -m src rules and the page's Initialise do."""
    if not kwargs and series(compiled) and compiled.assembly.get("rsr"):
        return bn_builder_series.recipe_extract(compiled, directory, origins)
    return module(compiled).extract_system_rules(compiled, directory, origins, **kwargs)


def generate_tbn(path: Path | str, **kwargs) -> Path:
    """The generated model beneath the assembly's example folder, by the module that builds it."""
    return module(compile_path(path)).generate_tbn(path, **kwargs)


def load_generated(directory: Path | str, device: str = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    import json

    data = json.loads((Path(directory) / "model.json").read_text(encoding="utf-8"))
    return _of_spec(data).load_generated(directory, device=device)
