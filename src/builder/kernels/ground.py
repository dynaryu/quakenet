"""Ground-failure kernels, one chain per mechanism: shaking -> permanent ground displacement (PGD, 0 where the
ground does not fail) -> road open. Each PGD kernel holds both models: whether the ground fails, and how far it
moves given it does.

    Liquefaction   occurs: Zhu et al. (2017), from PGV (and a PGA floor)
                   PGD | occurs: max(Hazus lateral spread from PGA, Hazus settlement)   (deterministic); settlement
                   only where the stretch has no free face (farther from water than the model's limit)
    Landslide      occurs: Nowicki Jessee et al. (2018), from PGV (and a PGA floor)
                   PGD | occurs: lognormal about Jibson (2007) Eq. 6, from PGA and the slope's critical acceleration
    Fault crossing occurs: on the scenario's rupture, P(surface rupture | M) (Wells & Coppersmith 1993)   (a root)
                   PGD | occurs: lognormal about the offset at the trace: a share of the segment's average, measured
                   for the scenario where given, else the Wells & Coppersmith (1994) average displacement for M

RoadFragility closes the road with P = Phi(ln(PGD / median) / beta): a Hazus roadway curve, or a curve of the
assembly's own (e.g. for strike-slip offset at a fault crossing).

The formulas are module functions on broadcastable tensors, which the kernels and the clique tree's
batched edges (src/inference/cliques_series.py) both call. Each PGD kernel has probability(...), P(the ground
fails), and given_occurrence(...) -> (ln median PGD, sigma): sigma 0 for a fixed displacement, ln median -inf
for none. RoadFragility.closed_given(ln_median, sigma) then sums PGD out in closed form
(a lognormal PGD through a lognormal curve is a normal CDF).
"""

from __future__ import annotations

import math

import torch

from . import Probability

INCH = 0.0254
LN10 = math.log(10.0)


# ---- the formulas --------------------------------------------------------------------------------------

def _stretch(f, length_m, patch_m):
    """P(at least one of L / patch independent patches fails), each with f."""
    n = torch.clamp_min(torch.as_tensor(length_m, dtype=torch.float64), patch_m) / patch_m
    return 1.0 - torch.pow((1.0 - f).clamp(0.0, 1.0), n)


def liquefaction_probability(pgv, pga, x0, msf, length_m, patch_m, zhu):
    """P(the stretch liquefies): Zhu et al. (2017) coverage f, 0 below the PGV and PGA floors, over the stretch."""
    a, b, c = zhu["coverage"]
    x = x0 + zhu["b_pgv"] * torch.log(pgv.clamp(1e-6, zhu["pgv_clip_cms"]) * msf)
    f = a / (1 + b * torch.exp(-c * torch.sigmoid(x))) ** 2
    f = torch.where((pgv < zhu["pgv_min_cms"]) | (pga < zhu["pga_min_g"]), torch.zeros_like(f), f)
    return _stretch(f, length_m, patch_m)


def liquefaction_ln_displacement(pga, pga_t, settle_m, kdelta):
    """ln PGD (m) given liquefaction: max(Hazus lateral spread, settlement); -inf for none."""
    r = pga / pga_t
    lat = torch.where(r <= 1, torch.zeros_like(r),
                      torch.where(r <= 2, 12 * r - 12, torch.where(r <= 3, 18 * r - 24, 70 * r - 180)))
    d = torch.maximum(kdelta * lat * INCH, torch.as_tensor(settle_m, dtype=torch.float64).expand_as(r))
    return torch.where(d > 0, torch.log(d.clamp_min(1e-300)), torch.full_like(d, -math.inf))


def landslide_probability(pgv, pga, x0, slope, length_m, patch_m, jes):
    """P(the slope fails): Nowicki Jessee et al. (2018) coverage f, 0 below the PGA floor, over the stretch."""
    a, b, c, d = jes["coverage"]
    lpgv = torch.log(pgv.clamp(1e-6, jes["pgv_clip_cms"]))
    pp = torch.sigmoid(x0 + jes["b_pgv"] * lpgv + jes["b_int"] * lpgv * slope)
    f = torch.exp(a + b * pp + c * pp ** 2 + d * pp ** 3)
    f = torch.where(pga < jes["pga_min_g"], torch.zeros_like(f), f)
    return _stretch(f, length_m, patch_m)


def landslide_ln_displacement(pga, ac, jib):
    """ln median PGD (m) given failure, Jibson (2007) Eq. 6; -inf where ac / PGA >= 1 (no sliding)."""
    r = (ac / pga.clamp_min(1e-9)).clamp_min(jib["ratio_min"])
    rr = r.clamp_max(1 - 1e-12)
    log10_cm = jib["c1"] + jib["c2"] * torch.log10(1 - rr) + jib["c3"] * torch.log10(rr)
    return torch.where(r < 1, (log10_cm - 2.0) * LN10, torch.full_like(r, -math.inf))


def _event(event_id, source, version):
    """The scenario ShakeMap's magnitude, and the grid (for its id and version)."""
    from .. import shakemap

    grid = shakemap.load(event_id, source, version, "PGA")
    mw = grid.meta.get("magnitude")
    if mw is None:
        raise ValueError(f"ShakeMap {event_id} gives no magnitude, which the ground-failure models need")
    return float(mw), grid


# ---- PGD: whether the ground fails and how far it moves, in one variable ----------------------------------

class _GroundPGD(Probability):
    """PGD (m) at a ground-failure component, zero-inflated: the ground fails with P (the occurrence model) and,
    given it fails, moves exp(ln median + sigma z) (the displacement model; a fixed value if sigma = 0, none if
    ln median = -inf); else PGD = 0. So P(PGD > 0) = P [ln median > -inf], and the ground failed (moved) exactly
    when PGD > 0.

    Subclasses give probability(...), P(the ground fails), and given_occurrence(...) -> (ln median, sigma), which
    the clique tree's edges also call, and _model(Cs_pars, n) -> (P, ln median, sigma) at the parents."""

    def _parts(self, Cs_pars, n):
        p, ln_med, sigma = self._model(Cs_pars, n)
        p = torch.as_tensor(p, dtype=torch.float64).expand(n)
        ln_med = torch.as_tensor(ln_med, dtype=torch.float64).expand(n)
        return p, ln_med, float(sigma)

    def _model(self, Cs_pars, n):
        raise NotImplementedError

    def sample(self, Cs_pars=None, n_sample=None, generator=None):
        n = n_sample if Cs_pars is None else len(Cs_pars)
        pars = None if Cs_pars is None else Cs_pars.to(device=self.device, dtype=torch.float64)
        p, ln_med, sigma = self._parts(pars, n)
        moves = (torch.rand(n, dtype=torch.float64, generator=generator) < p) & torch.isfinite(ln_med)
        z = torch.randn(n, dtype=torch.float64, generator=generator)
        pgd = torch.where(moves, torch.exp(ln_med.clamp_min(-700) + sigma * z), torch.zeros(n, dtype=torch.float64))
        return pgd[:, None], self._log_prob(pgd, p, ln_med, sigma)

    def log_prob(self, Cs):
        Cs = Cs.to(device=self.device, dtype=torch.float64)
        p, ln_med, sigma = self._parts(Cs[:, 1:] if Cs.shape[1] > 1 else None, len(Cs))
        return self._log_prob(Cs[:, 0], p, ln_med, sigma)

    @staticmethod
    def _log_prob(x, p, ln_med, sigma):
        """log P(PGD = 0) = log(1 - P [it can move]) at 0; above 0, log P + the displacement's log density (a
        point mass at exp(ln median) if sigma = 0)."""
        can = torch.isfinite(ln_med)
        at_zero = torch.log((1 - p * can).clamp_min(0))
        safe = torch.where(can, ln_med, torch.zeros_like(ln_med))
        if sigma > 0:
            moved = torch.distributions.LogNormal(safe, sigma).log_prob(x.clamp_min(1e-300))
        else:
            moved = torch.where(torch.isclose(torch.log(x.clamp_min(1e-300)), safe, rtol=0, atol=1e-9), 0.0, -math.inf)
        moved = torch.where(can & (x > 0), torch.log(p) + moved, torch.full_like(x, -math.inf))
        return torch.where(x == 0, at_zero, torch.where(x > 0, moved, torch.full_like(x, -math.inf)))


class LiquefactionPGD(_GroundPGD):
    """PGD at a liquefaction stretch, given PGV (cm/s) and PGA (g):

        liquefies with P = 1 - (1 - f)^(L / patch),  f = Zhu et al. (2017) coverage at
                     x0 + b_pgv ln(min(PGV, clip) MSF(M)), MSF = 1 / (1 + e^(-2 (M - 6))); 0 below the PGV and PGA floors
        given it liquefies, PGD = max(K_delta(M) a(PGA / PGA_t), settlement)   (Hazus; inches -> m; a fixed value)
        else PGD = 0
    """

    def __init__(self, childs, parents, event_id, model, x0, liq_class, length_km=0.0, dw_km=None, source=None,
                 version=None, device="cpu"):
        super().__init__(childs, parents, device)
        assert len(childs) == 1 and len(parents) == 2, "LiquefactionPGD: one child, PGV and PGA parents"
        self.mw, grid = _event(event_id, source, version)
        self.event = f"ShakeMap {grid.event_id} v{grid.version}, M{self.mw:g}"
        self.zhu, self.patch_m = model["zhu2017"], float(model["patch_m"])
        self.msf = 1.0 / (1.0 + math.exp(-2.0 * (self.mw - 6.0)))
        self.x0, self.length_m = float(x0), float(length_km or 0.0) * 1000.0
        hz, self.liq_class = model["hazus"], str(liq_class)
        self.pga_t = float(hz["pga_threshold_g"][self.liq_class])
        self.settle_m = float(hz["settlement_in"][self.liq_class]) * INCH
        k, m = hz["kdelta"], self.mw
        # Hazus lateral spread needs a free face: within lateral_spread_within_km of water, if the model sets it
        # (distance to water standing in for the free face); elsewhere the stretch only settles (K_delta taken as 0).
        within = model.get("lateral_spread_within_km")
        self.free_face = within is None or (dw_km is not None and float(dw_km) <= float(within))
        self.dw_km = dw_km
        self.kdelta = (k[0] * m ** 3 + k[1] * m ** 2 + k[2] * m + k[3]) if self.free_face else 0.0

    def probability(self, pgv, pga):
        return liquefaction_probability(pgv, pga, self.x0, self.msf, self.length_m, self.patch_m, self.zhu)

    def given_occurrence(self, pga=None):
        return liquefaction_ln_displacement(pga, self.pga_t, self.settle_m, self.kdelta), 0.0

    def _model(self, Cs_pars, n):
        pgv, pga = Cs_pars[:, 0], Cs_pars[:, 1]
        return (self.probability(pgv, pga), *self.given_occurrence(pga))

    def form(self):
        moves = (f"max(K_delta {self.kdelta:.3f} x Hazus lateral spread over PGA_t {self.pga_t:g} g, settlement "
                 f"{self.settle_m:.3f} m)" if self.free_face else
                 f"Hazus settlement {self.settle_m:.3f} m only (no free face: {self.dw_km:g} km from water)")
        return (f"PGD = 0 unless it liquefies (P Zhu et al. 2017 at x0 {self.x0:.3f}, L {self.length_m:g} m, "
                f"{self.patch_m:g} m patches); then {moves}, class {self.liq_class}; {self.event}")


class LandslidePGD(_GroundPGD):
    """PGD at a landslide stretch, given PGV (cm/s) and PGA (g):

        the slope fails with P = 1 - (1 - f)^(L / patch),  f = Nowicki Jessee et al. (2018) coverage at
                     x0 + b_pgv ln PGV' + b_int ln PGV' * slope, PGV' = min(PGV, clip); 0 below the PGA floor
        given it fails, it slides where r = ac / PGA < 1: log10 PGD(cm) ~ N(c1 + log10((1 - r)^c2 r^c3), sigma_log10),
                     r = max(ac / PGA, ratio_min)   (Jibson 2007 Eq. 6)
        else PGD = 0
    """

    def __init__(self, childs, parents, model, x0, slope_deg, ac_g, length_km=0.0, device="cpu"):
        super().__init__(childs, parents, device)
        assert len(childs) == 1 and len(parents) == 2, "LandslidePGD: one child, PGV and PGA parents"
        self.jes, self.patch_m = model["jessee2018"], float(model["patch_m"])
        self.x0, self.slope, self.length_m = float(x0), float(slope_deg), float(length_km or 0.0) * 1000.0
        self.jib, self.ac = model["jibson2007"], float(ac_g)

    def probability(self, pgv, pga):
        return landslide_probability(pgv, pga, self.x0, self.slope, self.length_m, self.patch_m, self.jes)

    def given_occurrence(self, pga=None):
        return landslide_ln_displacement(pga, self.ac, self.jib), self.jib["sigma_log10"] * LN10

    def _model(self, Cs_pars, n):
        pgv, pga = Cs_pars[:, 0], Cs_pars[:, 1]
        return (self.probability(pgv, pga), *self.given_occurrence(pga))

    def form(self):
        return (f"PGD = 0 unless the slope fails (P Nowicki Jessee et al. 2018 at x0 {self.x0:.3f}, slope {self.slope:g} deg, "
                f"L {self.length_m:g} m) and slides (PGA > ac {self.ac:g} g); then Jibson (2007) Eq. 6, "
                f"sigma {self.jib['sigma_log10']:g} log10 units")


class FaultPGD(_GroundPGD):
    """PGD at a road crossing of a mapped active fault (a root):

        the fault ruptures to the surface there with P = 1[within on_rupture_km of the rupture model's top edge]
                     / (1 + exp(-(a + b M)))   (Wells & Coppersmith 1993)
        given it does, PGD ~ lognormal about the offset at the trace, on_trace_fraction x the segment's average:
                     the scenario's measured average for the crossed fault's segment, if segment_offsets gives one
                     (sigma its sigma_ln); else the Wells & Coppersmith (1994) average displacement,
                     log10 AD = a + b M (strike-slip if the fault is, else all slip types; sigma theirs)
        else PGD = 0
    """

    def __init__(self, childs, parents, event_id, model, lat, lon, slip_type="", fault_id="", source=None, version=None,
                 device="cpu"):
        super().__init__(childs, parents, device)
        assert len(childs) == 1 and not parents, "FaultPGD: one child, a root"
        from .. import shakemap

        self.mw, grid = _event(event_id, source, version)
        self.event = f"ShakeMap {grid.event_id} v{grid.version}, M{self.mw:g}"
        path = shakemap.rupture_path(event_id, source, version)
        self.distance_km = math.inf if path is None else shakemap.distance_to_top_edge_km(path, float(lat), float(lon))
        if path is None:
            self.notes.add(f"{self.event} has no rupture model, so no fault crossing ruptures in this scenario")
        ps = model["p_surface_rupture"]
        p_sr = 1.0 / (1.0 + math.exp(-(ps["a"] + ps["b"] * self.mw)))
        self.q = p_sr if self.distance_km <= float(model["on_rupture_km"]) else 0.0
        st = str(slip_type or "").lower()
        self.slip_key = "strike-slip" if any(w in st for w in ("sinistral", "dextral", "strike")) else "all"
        self.on_trace = float(model.get("on_trace_fraction", 1.0))
        seg = (model.get("segment_offsets") or {}).get(str(event_id), {}).get("faults", {}).get(str(fault_id or ""))
        if seg is not None:                          # the scenario's own measured average on this segment
            self.segment = seg["segment"]
            self.average_m = float(seg["average_m"])
            self.sigma = float(model["segment_offsets"][str(event_id)]["sigma_ln"])
        else:                                        # Wells & Coppersmith (1994) for the magnitude
            self.segment = None
            ad = model["average_displacement"][self.slip_key]
            self.average_m = 10.0 ** (ad["a"] + ad["b"] * self.mw)
            self.sigma = ad["sigma_log10"] * LN10
        self.ln_med = math.log(self.average_m * self.on_trace)

    def probability(self, n=1):
        return self.tensor([self.q]).expand(n)

    def given_occurrence(self, pga=None):
        return self.tensor([self.ln_med]), self.sigma

    def _model(self, Cs_pars, n):
        return self.q, self.ln_med, self.sigma

    def form(self):
        where = f"{self.distance_km:.1f} km from the rupture" if math.isfinite(self.distance_km) else "no rupture model"
        average = (f"the {self.segment} segment's measured average {self.average_m:g} m" if self.segment else
                   f"the Wells & Coppersmith (1994) average displacement ({self.slip_key}) {self.average_m:.2f} m")
        return (f"PGD = 0 unless the fault ruptures to the surface (P {self.q:.3g}, {where}); then lognormal about "
                f"{self.on_trace:g} x {average} = {math.exp(self.ln_med):.2f} m at the trace (sigma {self.sigma:.2f} ln); "
                f"{self.event}")


# ---- the road ------------------------------------------------------------------------------------------

class RoadFragility(Probability):
    """Whether the road is open at a ground-failure component given its PGD (m):

        P(closed | PGD) = Phi(ln(PGD / median) / beta),  0 at PGD = 0   (Hazus roadway curve of the closing state)
    """

    def __init__(self, childs, parents, median_m, beta, closing_state="extensive", window_h=None, device="cpu"):
        super().__init__(childs, parents, device)
        assert len(childs) == 1 and len(parents) == 1, "RoadFragility: one RoadOpen child, one PGD parent"
        self.median, self.beta, self.closing_state = float(median_m), float(beta), closing_state
        self.window_h = window_h                     # the hours after the shock "closed" refers to, if the curve says

    def closed(self, pgd: torch.Tensor) -> torch.Tensor:
        """P(closed | PGD) (N,)."""
        out = torch.special.ndtr((torch.log(pgd.clamp_min(1e-300)) - math.log(self.median)) / self.beta)
        return torch.where(pgd > 0, out, torch.zeros_like(out))

    def closed_given(self, ln_med: torch.Tensor, sigma_ln) -> torch.Tensor:
        """P(closed) when PGD ~ LogNormal(ln_med, sigma_ln) (a point at exp(ln_med) if sigma_ln = 0;
        0 where ln_med = -inf): Phi((ln_med - ln median) / sqrt(beta^2 + sigma_ln^2))."""
        s = math.sqrt(self.beta ** 2 + float(sigma_ln) ** 2)
        out = torch.special.ndtr((ln_med - math.log(self.median)) / s)
        return torch.where(torch.isinf(ln_med) & (ln_med < 0), torch.zeros_like(out), out)

    def sample(self, Cs_pars, generator=None):
        pgd = Cs_pars.to(device=self.device, dtype=torch.float64)[:, 0]
        pc = self.closed(pgd)
        open_ = (torch.rand(len(pgd), dtype=torch.float64, generator=generator) >= pc).to(torch.float64)
        return open_[:, None], torch.log(torch.where(open_ == 1, 1 - pc, pc))

    def log_prob(self, Cs):
        Cs = Cs.to(device=self.device, dtype=torch.float64)
        state, pgd = Cs[:, 0], Cs[:, 1]
        pc = self.closed(pgd)
        valid = (state == 0) | (state == 1)
        return torch.where(valid, torch.log(torch.where(state == 1, 1 - pc, pc)), torch.full_like(pc, -math.inf))

    def form(self):
        state = (f"Hazus roadway {self.closing_state}" if self.closing_state in ("slight", "moderate", "extensive")
                 else self.closing_state)
        window = f", within {self.window_h:g} h" if self.window_h else ""
        return (f"P(closed | PGD) = Phi(ln(PGD / {self.median:g} m) / {self.beta:g}), "
                f"{state}{window}{self.parameter_sets()}")
