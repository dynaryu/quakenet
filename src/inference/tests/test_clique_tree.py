"""The clique tree on two kinds of network. One bridge per edge under SA (EMA, Hebei, toy-ca) is cliques.py's, as
it was before the series edges: trees.py sends such a BN there and never to cliques_series.py. The Kahramanmaraş
network, whose edges carry several bridges and ground-failure components and whose hub holds SA, PGV and PGA,
is cliques_series.py's."""

import pytest
import torch

from src.app import server
from src.builder.builders import build_tbn
from src.inference import cliques, cliques_series, engine, trees

SA_ONLY = ["toy-ca/demo", "ema-highway/gmm"]
KM = "kahramanmaras-2023/shakemap"


@pytest.fixture(scope="module")
def toy_probs():
    return build_tbn(engine.compile_query({"assembly": "toy-ca/demo"}))[1]


@pytest.mark.parametrize("assembly", SA_ONLY)
def test_an_sa_only_example_is_described_by_cliques_py(assembly):
    nodes = server.example(assembly)["nodes"]
    assert cliques_series.layout(nodes) == {"fields": ["SA"], "kinds": []}
    assert trees.describe(nodes) == cliques.describe() == server.example(assembly)["inference"]


def test_an_sa_only_bn_is_sampled_by_cliques_py_alone(toy_probs, monkeypatch):
    def unused(*args, **kwargs):
        raise AssertionError("a network of one bridge per edge went to cliques_series.py")

    assert not trees.series(toy_probs)
    t = trees.tree(toy_probs)
    assert trees.module(t) is cliques and "series" not in t
    assert trees.method(t) == "clique tree (src/inference/cliques.py)"
    for name in ("tree", "sample", "describe", "layout"):
        monkeypatch.setattr(cliques_series, name, unused)
    c = t["edges"][0]
    for ev in [{}, {c["E"]: 0}, {c["D"]: 3}, {"SystemState::n3": 0}, {c["Obs"]: 0.4}]:
        rows, _ = trees.sample(t, ev, n_chain=10, n_burn=3, n_keep=3, seed=1)
        ref, _ = cliques.sample(cliques.tree(toy_probs), ev, n_chain=10, n_burn=3, n_keep=3, seed=1)
        assert list(rows) == list(ref) and all(torch.equal(rows[k], ref[k]) for k in ref)


# ---- the Kahramanmaraş network --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def km():
    try:
        _, probs = build_tbn(engine.compile_query({"assembly": KM}))
    except Exception as exc:                                   # its ShakeMap is fetched unless cached
        pytest.skip(f"the {KM} BN cannot be built here: {exc}")
    assert trees.series(probs)
    t = trees.tree(probs)
    assert trees.module(t) is cliques_series
    return t


def _closed(rows, comp):
    return (comp["open_if"][rows[comp["D"]]] == 0) if comp["kind"] == "bridge" else (rows[comp["R"]] == 0)


def _consistent(t, rows):
    """Each series edge is open exactly when all its components are; a closed road has ground displacement."""
    for c in t["edges"]:
        if c["lone"]:
            continue
        closed = torch.zeros_like(rows[c["E"]], dtype=torch.bool)
        for comp in c["comps"]:
            closed |= _closed(rows, comp)
            if comp["kind"] == "ground":
                assert not ((rows[comp["R"]] == 0) & (rows[comp["PGD"]] <= 0)).any(), comp["R"]
        assert torch.equal(closed, rows[c["E"]] == 0), c["E"]


def test_the_kahramanmaras_example_is_described_with_its_fields_and_series_cliques():
    lay = cliques_series.layout(server.example(KM)["nodes"])
    assert lay == {"fields": ["SA", "PGV", "PGA"], "kinds": ["fault_e", "fault_r", "gf_e", "gf_r", "sa_d", "x_dr_e"]}
    d = cliques_series.describe(**lay)
    assert server.example(KM)["inference"] == d
    assert d["kinds"]["sa_e"] == "(SA, PGV, PGA, E)"
    pairs = {(m["from"], m["to"]) for m in d["messages"]}
    assert {("gf_r", "x_dr_e"), ("x_dr_e", "gf_r"), ("fault_r", "x_dr_e"), ("x_dr_e", "fault_r"), ("x_dr_e", "sa_e"),
            ("sa_e", "x_dr_e"), ("gf_e", "sa_e"), ("sa_e", "gf_e"), ("sa_d", "x_dr_e"), ("x_dr_e", "sa_d"),
            ("sa_obs", "sa_d"), ("sa_d", "sa_obs")} <= pairs
    assert not {("sa_obs", "x_dr_e"), ("x_dr_e", "sa_obs")} & pairs     # a measurement hangs off its bridge's clique
    assert len(pairs) == len(d["messages"])                     # one message each way across a link
    assert "{X}" not in str(d) and "{H}" not in str(d) and "F_" not in str(d)


def test_a_component_s_messages_name_only_the_field_it_reads():
    d = cliques_series.describe(["SA", "PGV", "PGA"], ["x_dr_e", "gf_r", "fault_r"])
    said = {(m["from"], m["to"]): m for m in d["messages"]}
    fault = str(said[("fault_r", "x_dr_e")]) + str(said[("x_dr_e", "fault_r")])
    assert not any(s in fault for s in ("X_j", "SA_j", "PGV_j", "PGA_j"))
    assert "P(PGD_j | R_j, evidence) ∝ P(PGD_j) P(R_j | PGD_j)" in fault
    assert said[("fault_r", "x_dr_e")]["carries"] == "m_up^Comp(R_j) = P(R_j, evidence_j)"
    assert said[("gf_r", "x_dr_e")]["carries"] == "m_up^Comp(PGV_j, PGA_j, R_j) = P(R_j, evidence_j | PGV_j, PGA_j)"
    assert said[("x_dr_e", "gf_r")]["carries"].startswith("m_down^Edge(PGV_j, PGA_j, R_j)")


@pytest.mark.parametrize("assembly", SA_ONLY)
def test_an_sa_only_tree_as_drawn_is_one_clique_sa_d_e_per_edge(assembly):
    ex = server.example(assembly)
    t = ex["cliques"]
    assert t == cliques.drawing(ex["nodes"]) and not t["comp"]
    for c in t["local"]:
        (sa, d, e), f = c["members"], c["factors"]
        assert c["kind"] == "sa_d_e" and c["separator"] == [sa, e]
        assert f == [{"p": [d], "given": [sa]}, {"p": [e], "given": [d]}]


def test_a_kahramanmaras_edge_clique_holds_its_table_as_the_bn_has_it():
    ex = server.example(KM)
    t, by = ex["cliques"], {n["id"]: n for n in ex["nodes"]}
    drawn = {(m["from"], m["to"]) for m in ex["inference"]["messages"]}
    for c in t["local"]:
        e = c["members"][-1]
        assert {(c["kind"], "sa_e"), ("sa_e", c["kind"])} <= drawn
        if c["kind"] == "x_dr_e":                       # P(E | D, R) over the BN's parents, in road order
            assert c["factors"][-1] == {"p": [e], "given": ex["tables"][e]["columns"][1:]}
            assert sorted(c["factors"][-1]["given"]) == sorted(by[e]["parents"])
    for c in t["comp"]:
        edge = t["local"][c["edge"]]
        assert edge["kind"] == "x_dr_e" and c["members"][-1] in edge["members"]
        assert set(c["separator"]) == set(c["members"]) & set(edge["members"])        # what the two share
        assert {(c["kind"], "x_dr_e"), ("x_dr_e", c["kind"])} <= drawn
    for c in t["local"]:                                # the edge clique holds its table alone
        if c["kind"] == "x_dr_e":
            assert len(c["factors"]) == 1
    bridges = [c for c in t["comp"] if c["kind"] == "sa_d"]
    assert bridges
    for c in bridges:                                   # (SA_j, D_j) with P(D_j | SA_j), sharing both
        (sa, d) = c["members"]
        assert by[d]["function"] == "fragility.bridge" and by[d]["parents"] == [sa]
        assert c["separator"] == [sa, d] and c["factors"] == [{"p": [d], "given": [sa]}]
    for o in t["obs"]:                                  # an SA measurement on a series edge hangs off its bridge
        if o["comp"] >= 0 or t["local"][o["edge"]]["kind"] == "x_dr_e":
            assert t["comp"][o["comp"]]["kind"] == "sa_d" and o["members"][0] in t["comp"][o["comp"]]["members"]
    # every BN variable is in some clique, no helper variable is, and every node's own factor is in exactly one
    cards = [c for k in ("obs", "comp", "local", "systems") for c in t[k]] + [t["hub"]]
    assert {m for c in cards for m in c["members"]} == set(by)
    owned = [v for c in cards for f in c["factors"] for v in (f.get("p") or f.get("f"))]
    assert sorted(owned) == sorted(by)


def test_a_kahramanmaras_edge_lists_its_components_in_road_order(km):
    e20 = next(c for c in km["edges"] if c["E"] == "EdgeOpen::E20")
    assert not e20["lone"] and len(e20["comps"]) == 35
    assert sum(not c["lone"] for c in km["edges"]) == 63


@pytest.mark.parametrize("ev", [{}, {"EdgeOpen::E20": 0}, {"DamageState.HazusBridge::B063": 4},
                                {"PGD.Landslide::G067": 1}, {"PGD.Landslide::G067": 1, "RoadOpen.Landslide::G067": 0},
                                {"SystemState::T01-S1": 0}])
def test_kahramanmaras_rows_hold_the_evidence_and_their_series(km, ev):
    rows, _ = trees.sample(km, ev, n_chain=20, n_burn=10, n_keep=10, seed=3)
    _consistent(km, rows)
    for var, value in ev.items():
        x = rows[var]
        assert ((x > 0) == bool(value)).all() if var.startswith("PGD") else (x == value).all()


@pytest.mark.parametrize("ln_med", [-1.0, -3.0, -6.0, 1.5])
@pytest.mark.parametrize("closed", [True, False])
def test_pgd_given_its_road_s_state_is_drawn_as_rejection_draws_it(ln_med, closed):
    """PGD ~ LogNormal(ln_med, sigma) given the road closed (or open) under a lognormal roadway curve, against
    PGD drawn from its own law and kept with P(that state | PGD), also where that state is rare."""
    from src.builder.kernels.ground import RoadFragility

    road, sigma, n = RoadFragility(["R"], ["PGD"], median_m=0.6096, beta=0.7), 1.174, 200_000
    g = torch.Generator().manual_seed(0)
    got = cliques_series._lognormal_given(torch.full((n,), ln_med, dtype=torch.float64), sigma, road,
                                          torch.full((n,), closed), g).log()
    big = ln_med + sigma * torch.randn(20_000_000, dtype=torch.float64, generator=g)
    pc = road.closed(big.exp())
    ref = big[torch.rand(len(big), dtype=torch.float64, generator=g) < (pc if closed else 1 - pc)]
    assert len(ref) > 500
    qs = torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)
    assert torch.allclose(got.quantile(qs), ref.quantile(qs), atol=0.15)


def test_the_system_message_with_edges_opened_agrees_with_the_system_function(km):
    """system_down(opened=...) decides each row with each edge opened from the rules' violations (learning the
    rows they leave): it agrees with evaluating the system on each opened edge vector."""
    rows, _ = trees.sample(km, {"EdgeOpen::E08": 1}, n_chain=20, n_burn=10, n_keep=10, seed=3)
    for k in ("T05-any", "T22-S2"):
        prob = km["systems"][k]
        x = torch.stack([rows[v.name] for v in prob.parents], dim=1).long()
        cols = list(range(x.shape[1]))
        got, _ = cliques_series.system_down(km, k, x, {}, opened=cols)
        assert got.shape == (len(x), len(cols))
        for j in cols:
            y = x.clone()
            y[:, j] = 1
            assert torch.equal(got[:, j], prob.evaluate(y.double())[:, 0].long()), (k, j)
