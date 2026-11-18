"""The ground-failure kernels on the Kahramanmaraş network: fault crossings offset by their segment's measured average
(a share of it at the trace) and closed by a curve of their own; lateral spread only near a free face."""

import math

import pytest
import torch

from src.builder.builders import build_tbn
from src.builder.kernels import ground
from src.inference import engine

KM = "kahramanmaras-2023/shakemap"
EVENT_FUNCTIONS = ["hazard.shakemap", "hazard.shakemap.ground", "groundfailure.liquefaction", "hazard.fault-rupture"]


def _build(event=None):
    query = {"assembly": KM}
    if event:
        query["parameters"] = {f: {"event_id": event} for f in EVENT_FUNCTIONS}
    try:
        return build_tbn(engine.compile_query(query))[1]
    except Exception as exc:                                   # its ShakeMaps are fetched unless cached
        pytest.skip(f"the {KM} BN cannot be built here: {exc}")


@pytest.fixture(scope="module")
def m78():
    return _build()


def _p_closed(probs, gid):
    pgd, road = probs[f"PGD.FaultOffset::{gid}"], probs[f"RoadOpen.FaultRupture::{gid}"]
    return pgd.q * float(road.closed_given(torch.tensor([pgd.ln_med]), pgd.sigma)[0])


@pytest.mark.parametrize("gid, segment, average", [("G182", "Amanos", 3.8), ("G044", "Amanos", 3.8),
                                                   ("G233", "Pazarcık", 6.0), ("G125", "Erkenek", 5.3)])
def test_a_crossing_on_the_m78_rupture_is_offset_by_its_segment(m78, gid, segment, average):
    pgd = m78[f"PGD.FaultOffset::{gid}"]
    assert pgd.segment == segment and pgd.average_m == average
    assert math.exp(pgd.ln_med) == pytest.approx(0.8 * average) and pgd.sigma == 0.3
    assert pgd.q == pytest.approx(0.97, abs=0.01)


def test_a_fault_with_no_measured_segment_keeps_wells_and_coppersmith(m78):
    pgd = m78["PGD.FaultOffset::G001"]                          # ME_TRCS999: not one of the M7.8's segments
    assert pgd.segment is None
    assert pgd.average_m == pytest.approx(10 ** (-6.32 + 0.90 * pgd.mw))
    assert math.exp(pgd.ln_med) == pytest.approx(0.8 * pgd.average_m)


def test_another_event_has_no_measured_segments():
    probs = _build("us6000jlqa")
    pgd = probs["PGD.FaultOffset::G213"]
    assert pgd.q > 0 and pgd.segment is None
    assert math.exp(pgd.ln_med) == pytest.approx(0.8 * 10 ** (-6.32 + 0.90 * pgd.mw))


def test_fault_crossings_close_by_their_own_curve(m78):
    road = m78["RoadOpen.FaultRupture::G182"]
    assert (road.median, road.beta, road.closing_state, road.window_h) == (4.5, 0.7, "impassable", 72)
    assert m78["RoadOpen.Liquefaction::G022"].median == pytest.approx(1.524)          # Hazus, unchanged
    assert _p_closed(m78, "G182") == pytest.approx(0.29, abs=0.01)                    # Amanos: about 0.3
    on = [k.split("::")[1] for k, p in m78.items() if type(p).__name__ == "FaultPGD" and p.q > 0]
    assert len(on) == 10
    assert 3.0 < sum(_p_closed(m78, g) for g in on) < 4.5                             # was about 8.8


def test_lateral_spread_only_near_water(m78):
    near, far = m78["PGD.Liquefaction::G022"], m78["PGD.Liquefaction::G175"]        # 94 m and 1.1 km from water
    assert near.free_face and near.kdelta > 1 and not far.free_face and far.kdelta == 0
    pga = torch.tensor([0.45])
    assert math.exp(float(near.given_occurrence(pga)[0])) == pytest.approx(0.92, abs=0.02)   # lateral spread
    assert math.exp(float(far.given_occurrence(pga)[0])) == pytest.approx(far.settle_m)      # settlement only
    assert sum(p.free_face for p in m78.values() if type(p).__name__ == "LiquefactionPGD") == 9


def test_displacement_takes_k_delta_per_stretch():
    # The clique tree's batched edges pass K_delta per stretch: 0 where there is no free face, so it only settles.
    r = torch.tensor([0.45 / 0.15])
    d = ground.liquefaction_ln_displacement(r * 0.15, 0.15, 0.0508, torch.tensor([1.2015, 0.0]))
    assert torch.exp(d).tolist() == pytest.approx([30 * 1.2015 * ground.INCH, 0.0508])
