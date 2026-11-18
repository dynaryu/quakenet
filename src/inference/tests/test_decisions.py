"""Causal semantics, joint dependence, evidence denominators and decision value."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.inference import engine
from src.inference.comparisons import execute
from src.inference.decision import from_distribution

TOY = "toy-ca/demo"


def test_forced_states_propagate_and_have_distinct_sample_banks():
    base = {"assembly": TOY, "samples": 200, "targets": ["EdgeOpen::e2"]}
    a = engine.answer({**base, "do": {"EdgeOpen::e2": 0}}, record={"source": "test"})         # by index
    b = engine.answer({**base, "do": {"EdgeOpen::e2": "True"}}, record={"source": "test"})    # by label
    assert a["targets"][0]["summary"]["P"]["False"] == 1
    assert b["targets"][0]["summary"]["P"]["True"] == 1
    assert a["record"]["samples_file"]["bank"] != b["record"]["samples_file"]["bank"]
    again = engine.answer({**base, "do": {"EdgeOpen::e2": 0}}, record={"source": "test"})
    assert again["record"]["samples_file"]["reused"]


def test_joint_probability_uses_the_same_samples_and_evidence_denominator():
    leaf = {"node": "EdgeOpen::e2", "op": "==", "value": 0}
    r = engine.answer({"assembly": TOY, "samples": 1000, "events": [leaf], "joint_events": [
        {"name": "same twice", "expression": {"all": [leaf, leaf]}},
        {"name": "tautology", "expression": {"any": [leaf, {"not": leaf}]}}
    ]})
    assert r["events"][0]["p"] == r["events"][1]["p"]
    assert r["events"][2]["p"] == 1
    observed = engine.answer({"assembly": TOY, "samples": 5000, "evidence": {"EdgeOpen::e2": 0}, "events": [leaf]})
    assert observed["events"][0]["p"] == 1
    assert observed["inference"]["accepted"] == 5000       # the clique tree conditions on it: every row counts


def test_observation_and_intervention_differ_with_a_common_cause():
    """SA drives both b4's damage and its station's reading: observing complete damage raises the reading,
    forcing it does not."""
    event = [{"node": "SAObs(1.0)::b4", "op": ">", "value": 0.5}]
    base = {"assembly": TOY, "samples": 20000, "events": event}
    prior = engine.answer(base)["events"][0]["p"]
    observed = engine.answer({**base, "evidence": {"DamageState.HazusBridge::b4": 4}})["events"][0]["p"]
    forced = engine.answer({**base, "do": {"DamageState.HazusBridge::b4": 4}})["events"][0]["p"]
    assert observed > prior + 0.1
    assert forced == pytest.approx(prior, abs=0.03)


def test_impossible_evidence_is_not_reported_as_zero_probability():
    with pytest.raises(engine.QueryError, match="contradicts"):
        engine.answer({"assembly": TOY, "samples": 100, "targets": ["EdgeOpen::e2"],
                       "do": {"EdgeOpen::e2": 0}, "evidence": {"EdgeOpen::e2": 1}})
    with pytest.raises(engine.QueryError, match="continuous observations"):
        engine.answer({"assembly": TOY, "targets": ["EdgeOpen::e2"], "evidence": {"SA(1.0)::b2": .5}})
    with pytest.raises(engine.QueryError, match="not a state"):
        engine.answer({"assembly": TOY, "targets": ["EdgeOpen::e2"], "do": {"EdgeOpen::e2": "unknown"}})


def test_forced_system_state_does_not_invoke_rule_reports():
    result = engine.answer({"assembly": TOY, "samples": 100, "do": {"SystemState::n4": 0},
                            "events": [{"node": "SystemState::n4", "op": "==", "value": 0}]})
    assert result["events"][0]["p"] == 1
    assert "n4" not in [o["origin"] for o in result["rules"]["origins"]]      # the forced origin is not evaluated


def test_voi_optimises_after_each_observation_and_accounts_for_cost():
    task = {"state_nodes": ["EdgeOpen::e2"], "loss_units": "test units", "actions": [
        {"name": "repair", "losses": {"0": 20, "1": 20}}, {"name": "defer", "losses": {"0": 100, "1": 0}}],
        "inspections": [{"node": "EdgeOpen::e2", "cost": 2, "accuracy": [[1, 0], [0, 1]]}]}
    result = from_distribution([(0,), (1,)], [.2, .8], task)
    assert result["current_expected_loss"] == 20
    assert result["rows"][0]["evsi"] == pytest.approx(16)
    assert result["rows"][0]["net_value"] == pytest.approx(14)
    assert [o["action"] for o in result["rows"][0]["outcomes"]] == ["repair", "defer"]
    task["inspections"][0]["accuracy"] = [[.5, .5], [.5, .5]]
    assert from_distribution([(0,), (1,)], [.2, .8], task)["recommendation"] == "do not inspect"


def test_path_candidates_end_at_requested_destination_and_have_joint_events():
    result = execute({"kind": "paths", "query": {"assembly": TOY, "samples": 300},
                      "origin": "n4", "destination": "n1", "limit": 3})
    assert len(result["rows"]) == 3
    assert all(r["nodes"][0] == "n4" and r["nodes"][-1] == "n1" for r in result["rows"])
    assert all(r["travel_time_h"] > 0 and 0 <= r["p"] <= 1 for r in result["rows"])


# ---- inspect: E[T | do(E_i = open), ev] by the clique tree ----------------------------------------------

@pytest.fixture
def toy_session():
    import torch
    from src.builder.bn_builder import build_tbn
    from src.inference import cliques
    from src.inference.session import Session

    variables, probs = build_tbn(engine.compile_query({"assembly": "toy-ca/demo"}))
    torch.manual_seed(0)
    return Session(cliques.tree(probs), probs, variables, sampler={"n_chain": 20, "n_burn": 5, "n_keep": 50})


def _by_intervening(s, spec):
    """VOI_i as before: every column of edge i's closed rows, E_i set open and every system evaluated again."""
    import torch
    from src.inference.session import _comps, _edge_label

    now, out = s._target(spec["target"], s.rows), {}
    for c in [c for c in s.t["edges"] if _comps(c) and c["E"] not in s.evidence]:
        closed, F = s.rows[c["E"]] == 0, torch.zeros(s.n, 1, dtype=torch.float64)
        if closed.any():
            sub = {k: v[closed] for k, v in s.rows.items()}
            F[closed, 0] = now[closed] - s._target(spec["target"], s._intervene({c["E"]: 1}, sub))
        out[_edge_label(c)] = s._means(F)[0][0].item()
    return out


@pytest.mark.parametrize("target", [{"kind": "cut_off"}, {"kind": "fail", "origin": "n3"}, {"kind": "closed", "edge": "e2"},
                                    {"kind": "cut_off_at_least", "at_least": 2}])
def test_inspect_passes_only_the_system_messages_again_and_agrees_with_evaluating_everything(toy_session, target):
    spec = {"target": target, "decision": "inspect"}
    got, ref = toy_session._values(spec), _by_intervening(toy_session, spec)
    assert got.keys() == ref.keys() and all(abs(got[k][0] - ref[k]) < 1e-9 for k in ref)


def test_the_system_message_learns_rules_from_the_rows_they_leave_and_keeps_them(toy_session):
    """system_down decides every row by the rsr rules: with none stored, it learns them from the rows (rsr with
    the rows as its samples) and keeps them, so the next message learns nothing; the rules agree with the system
    function on every row."""
    import torch
    from src.inference import cliques

    t = toy_session.t
    k, prob = next(iter(t["systems"].items()))
    e = torch.stack([toy_session.rows[v.name] for v in prob.parents], dim=1).long()
    prob.rules = None
    first, n_learnt = cliques.system_down(t, k, e, {})
    assert prob.rules is not None and n_learnt > 0
    assert torch.equal(first, prob.evaluate(e.double())[:, 0].long())
    again, n_again = cliques.system_down(t, k, e, {})
    assert torch.equal(again, first) and n_again == 0


def test_inspect_with_a_system_observed_opens_the_edge_in_the_rows_the_evidence_left(toy_session):
    """The observed system state is the one before the repair: each row keeps it until E_i opens, then its system
    is passed down again; the tree is not sampled again under do (where it could not hold)."""
    toy_session.observe("SystemState::n3", 0)
    spec = {"target": {"kind": "fail", "origin": "n3"}, "decision": "inspect"}
    got, ref = toy_session._values(spec), _by_intervening(toy_session, spec)
    assert got.keys() == ref.keys() and all(abs(got[k][0] - ref[k]) < 1e-9 for k in ref)
    assert any(v > 0 for v, _ in got.values())


def test_rules_learnt_by_the_system_message_are_stored_with_the_system(toy_session, tmp_path, monkeypatch):
    """A system's own rules, extended from the rows they leave undecided, are written back to its rsr folder:
    the .json references and the .pt tensors alike, with what was learnt recorded in its model.json."""
    import json
    import shutil

    import torch
    from src.inference import cliques

    t = toy_session.t
    k, prob = next(iter(t["systems"].items()))
    ref = prob.rules_reference
    for side in ("upper", "lower"):                      # a copy of the stored rules, emptied
        src = prob.rules_directory / ref[side]
        dst = tmp_path / ref[side]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src.parent / "model.json", dst.parent / "model.json")
        json.dump(json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))[:0],
                  open(dst.with_suffix(".json"), "w", encoding="utf-8"))
        torch.save(torch.load(src, weights_only=True)[:0], dst)
    monkeypatch.setattr(prob, "rules_directory", tmp_path)
    monkeypatch.setattr(prob, "rules", tuple(torch.load(tmp_path / ref[s], weights_only=True) for s in ("upper", "lower")))
    monkeypatch.setenv("QUAKENET_SAVE_RULES", "1")

    e = torch.stack([toy_session.rows[v.name] for v in prob.parents], dim=1).long()
    states, n_learnt = cliques.system_down(t, k, e, {})
    assert n_learnt > 0 and torch.equal(states, prob.evaluate(e.double())[:, 0].long())
    for side, mat in zip(("upper", "lower"), prob.rules):
        stored = torch.load(tmp_path / ref[side], weights_only=True)
        refs = json.loads((tmp_path / ref[side]).with_suffix(".json").read_text(encoding="utf-8"))
        assert torch.equal(stored, mat) and len(refs) == len(stored)
    meta = json.loads(((tmp_path / ref["upper"]).parent / "model.json").read_text(encoding="utf-8"))
    assert meta["learnt_from_samples"]["sfun_calls"] == n_learnt


def test_rules_learnt_without_stored_ones_go_to_the_system_s_own_folder_where_the_builder_finds_them(
        toy_session, tmp_path, monkeypatch):
    """A system with no stored rules (a fresh clone) learns them from the rows and stores them in its own
    rules folder (rules_home), created with what the builder checks; the next build finds them as its own."""
    import json

    import torch
    from src.builder import bn_builder
    from src.inference import cliques

    t = toy_session.t
    k, prob = next(iter(t["systems"].items()))
    monkeypatch.setattr(prob, "rules", None)
    monkeypatch.setattr(prob, "rules_reference", None)
    monkeypatch.setattr(prob, "rules_directory", tmp_path)
    monkeypatch.setenv("QUAKENET_SAVE_RULES", "1")

    e = torch.stack([toy_session.rows[v.name] for v in prob.parents], dim=1).long()
    states, n_learnt = cliques.system_down(t, k, e, {})
    assert n_learnt > 0 and torch.equal(states, prob.evaluate(e.double())[:, 0].long())
    folder = tmp_path / prob.rules_home["folder"]
    for side, file in (("upper", "refs_up_1.pt"), ("lower", "refs_low_0.pt")):
        stored = torch.load(folder / file, weights_only=True)
        assert torch.equal(stored, prob.rules[0 if side == "upper" else 1])
        assert len(json.loads((folder / file).with_suffix(".json").read_text(encoding="utf-8"))) == len(stored)
    assert prob.rules_reference["upper"] == f"{prob.rules_home['folder']}/refs_up_1.pt"

    data = bn_builder.tbn_data(engine.compile_query({"assembly": TOY}))
    node = next(n for n in data["nodes"] if n["childs"] == [f"SystemState::{k}"])
    assert bn_builder.stored_rules(tmp_path, node) == [(prob.target_add_h, folder)]
    bn_builder.attach_rules(data, tmp_path)
    assert node["kwargs"]["rules"]["upper"] == prob.rules_reference["upper"]
