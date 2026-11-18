"""Export the schematic network to a quakenet example folder.

Writes network.json (Node, Edge, Bridge tables), bridges.csv (per-bridge provenance and the
reason for its HAZUS class) and edges.csv (per-edge road classes and speed basis).

usage: python export_quakenet.py <network dir> <osm dir> <Bas et al. CSV> <output folder>
"""
import sys, json
from collections import Counter
from math import cos, radians
from pathlib import Path
import pandas as pd

net, osm, paper_csv, out = (Path(a) for a in sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
nodes = json.load(open(net / "nodes.json", encoding="utf-8"))
edges = json.load(open(net / "edges.json", encoding="utf-8"))
bridges = json.load(open(net / "bridges.json", encoding="utf-8"))
dest = pd.read_csv(osm / "destinations.csv").set_index("id")

# ---------------- HAZUS classification of the Bas et al. (2024) records ----------------
SEISMIC_FROM = 2000      # project choice: built in or after 2000 -> seismic design
p = pd.read_csv(paper_csv, header=2, encoding="cp1254")
p.columns = [" ".join(str(c).split()) for c in p.columns]
p = p.dropna(subset=["Latitude (decimal)"])
p = p[p["Pedestrian or Vehicular Bridge?"].astype(str).str.strip() == "Vehicular"].reset_index(drop=True)


def num(v):
    return pd.to_numeric(v, errors="coerce")


def girder_group(g):
    g = str(g).strip().lower()
    if g in ("nan", "not observed", ""):
        return None
    if "arch" in g:
        return "arch"
    if "truss" in g:
        return "steel-truss"
    if "steel box" in g:
        return "steel-continuous"
    if g.startswith("steel"):
        return "steel-simple"
    if "post tensioned" in g:
        return "prestressed-continuous"
    if g.startswith("cast-in-place; precast"):
        return "prestressed-simple"
    if "cast-in-place" in g or "concrete box girder" in g or g == "concrete":
        return "concrete-continuous"
    if "precast" in g or "prestressed" in g:
        return "prestressed-simple"
    return None


def classify(r):
    """Return (class or None, basis). None means the girder type was not observed."""
    import re
    yv = str(r["Estimated Vintage of Construction (decade)"])
    m = re.search(r"(1[89]\d\d|20\d\d)", yv)
    year = int(m.group(1)) if m else None
    spans, span, length = num(r["No. of Spans"]), num(r["Avg. Span Length (meters)"]), \
        num(r["Est. Total Bridge Length, Abutment-to-Abutment (meters)"])
    seismic = year is not None and year >= SEISMIC_FROM
    era = (f"built {year}" if year else f"year '{yv}' unreadable") + f" -> {'seismic' if seismic else 'conventional'}"
    if span > 150:
        return ("HWB2" if seismic else "HWB1"), f"major bridge (avg span {span:g} m > 150 m), {era}"
    if spans == 1:
        return ("HWB4" if seismic else "HWB3"), f"single span, {era}"
    grp = girder_group(r["Girder Construction"])
    g = str(r["Girder Construction"]).strip()
    if grp is None:
        return None, f"girder '{g}' not observed"
    if grp == "prestressed-simple":
        return ("HWB19" if seismic else "HWB17"), f"girder '{g}' -> simply supported prestressed concrete, {era}"
    if grp == "concrete-continuous":
        return ("HWB11" if seismic else "HWB10"), f"girder '{g}' -> continuous concrete, {era}"
    if grp == "prestressed-continuous":
        return ("HWB23" if seismic else "HWB22"), f"girder '{g}' -> continuous prestressed concrete, {era}"
    if grp == "steel-simple":
        c = "HWB14" if seismic else ("HWB24" if length < 20 else "HWB12")
        return c, f"girder '{g}' -> simply supported steel, length {length:g} m, {era}"
    if grp == "steel-continuous":
        c = "HWB16" if seismic else ("HWB26" if length < 20 else "HWB15")
        return c, f"girder '{g}' -> continuous steel, length {length:g} m, {era}"
    return "HWB28", f"girder '{g}' ({grp}) is outside HAZUS classes HWB1-27"


p["cls"], p["basis"] = zip(*p.apply(classify, axis=1))
counts = Counter(c for c in p["cls"] if isinstance(c, str))
DEFAULT = counts.most_common(1)[0][0]
print("classes of the paper's road bridges:", dict(counts.most_common()), "| default:", DEFAULT)
key = {(str(r["Bridge Site ID"]), str(r["Bridge Name"])): r for _, r in p.iterrows()}

# ---------------- tables ----------------
names_S = {"S1": "Adana (O-52 Adana Doğu junction)", "S2": "Ceyhan (O-52 Ceyhan-Yumurtalık junction)",
           "S3": "Gaziantep (O-52 Havalimanı junction)"}
Node, Edge, Bridge, brows, erows = [], [], [], [], []
for k, v in nodes.items():
    row = {"id": k, "lat": v["y"], "lon": v["x"]}
    if k.startswith("S"):
        row["name"] = names_S[k]
    elif k.startswith("T"):
        row["name"] = dest.loc[k, "name"]
        if pd.notna(dest.loc[k, "population"]):
            row["population"] = float(dest.loc[k, "population"])
    elif k.startswith("J"):
        row["name"] = "junction: " + v["name"].replace("(auto)", "").strip()
    else:
        row["name"] = "bridge split: " + v["name"].replace("split on ", "")
    row["key_destination"] = k.startswith("S")
    Node.append(row)

# ---------------- topological edges ----------------
# The build splits each road into short pieces, one per component (a bridge or a ground-failure component),
# with connectors between. An edge here runs between two nodes where the route can split or ends: a node
# where three or more pieces meet, a town (T) or a source (S). The pieces between are merged into it, with
# every component on them; the shape points (X) and the junctions where only two pieces meet drop out.
ground = json.load(open(net / "ground.json", encoding="utf-8")) if (net / "ground.json").is_file() else {}
SPEED = {"motorway": 100, "trunk": 80, "primary": 70, "secondary": 50}


def piece_speed(e):
    return SPEED.get(max(e["road_class_km"].items(), key=lambda kv: kv[1])[0], 50)


degree, pieces_at = Counter(), {}
for k0, e in edges.items():
    for n in (e["from"], e["to"]):
        degree[n] += 1
        pieces_at.setdefault(n, []).append(k0)
ends = [k for k in nodes if degree[k] != 2 or k[0] in "ST"]
dropped_junctions = sorted((k for k in nodes if k[0] == "J" and k not in set(ends)), key=lambda s: int(s[1:]))
chains, seen = [], set()
for start in ends:                                  # walk every chain of pieces from each end node
    for k0 in pieces_at.get(start, []):
        if k0 in seen:
            continue
        path, steps, at = [start], [], start
        while True:
            e = edges[k0]
            seen.add(k0)
            forward = e["from"] == at
            at = e["to"] if forward else e["from"]
            steps.append((k0, forward))
            path.append(at)
            if at in set(ends):
                break
            (k0,) = [k for k in pieces_at[at] if k != k0]
        chains.append((path, steps))
assert seen == set(edges), "every piece lies on one chain"
with_gf = {g["edge"] for g in ground.values()}
can_close = [any(edges[k0]["bridge"] or k0 in with_gf for k0, _ in steps) for _, steps in chains]
n_random = sum(can_close)
names, ir, ic = [], 0, 0
for c in can_close:
    if c:
        ir += 1
        names.append(f"E{ir:02d}")
    else:
        ic += 1
        names.append(f"C{ic:02d}")
print(f"edges: {len(chains)} topological edges from {len(edges)} pieces: {n_random} random (E), "
      f"{len(chains) - n_random} fixed open (C); junctions merged away: {dropped_junctions}")

# piece -> (edge, km along the edge at the piece's start, whether the piece runs the edge's way)
on_edge = {}
for name, (path, steps), c in zip(names, chains, can_close):
    km, hours, road_km, refs = 0.0, 0.0, Counter(), []
    for k0, forward in steps:
        e = edges[k0]
        on_edge[k0] = (name, km, forward)
        km += e["length_km"]
        hours += e["length_km"] / piece_speed(e)
        road_km.update(e["road_class_km"])
        refs += [r for r in e["road_refs"] if r not in refs]
    row = {"id": name, "source": path[0], "target": path[-1], "directed": False, "length_km": round(km, 3),
           "capacity": 1, "avg_speed_kph": round(km / hours, 2),
           "geometry": [[nodes[n]["y"], nodes[n]["x"]] for n in path]}
    if not c:
        row["fixed_open"] = True
    Edge.append(row)
    erows.append(dict(id=name, fixed_open=not c, source=path[0], target=path[-1], length_km=round(km, 3),
                      avg_speed_kph=round(km / hours, 2), travel_time_h=round(hours, 4), n_pieces=len(steps),
                      pieces=" ".join(k0 for k0, _ in steps), links=" ".join(dict.fromkeys(edges[k0]["link"] for k0, _ in steps)),
                      road_class_km=json.dumps({k: round(v, 3) for k, v in road_km.items()}), road_refs=" ".join(refs),
                      bridges="", ground_failure=""))
NODE_KEPT = set(ends)
Node = [r for r in Node if r["id"] in NODE_KEPT]


def chainage(k0, lat, lon):
    """km from its edge's source to the point (lat, lon) on piece k0: the point projected on the piece's chord."""
    name, km0, forward = on_edge[k0]
    e = edges[k0]
    a, b = (nodes[e["from"]], nodes[e["to"]]) if forward else (nodes[e["to"]], nodes[e["from"]])
    kx = cos(radians((a["y"] + b["y"]) / 2))
    dx, dy = (b["x"] - a["x"]) * kx, b["y"] - a["y"]
    t = 0.5 if dx == dy == 0 else ((lon - a["x"]) * kx * dx + (lat - a["y"]) * dy) / (dx * dx + dy * dy)
    return round(km0 + min(max(t, 0.0), 1.0) * e["length_km"], 3)

for k, b in bridges.items():
    recs = [key[(q["site_id"], q["name"])] for q in b["paper"] if (q["site_id"], q["name"]) in key]
    cls = [r["cls"] for r in recs if isinstance(r["cls"], str)]
    if cls:
        hc = Counter(cls).most_common(1)[0][0]
        basis = recs[[r["cls"] for r in recs].index(hc)]["basis"]
        if len(set(cls)) > 1:
            basis += f" (most common of the records' classes {sorted(set(cls))})"
        how = "Bas et al. record"
    else:
        hc = DEFAULT
        basis = ("girder type not observed in the Bas et al. record" if recs else "no structural data (OSM only)") + \
                f" -> most common class among the paper's road bridges ({DEFAULT})"
        how = "default"
    # place the bridge at its edge's midpoint (simplest schematic); keep the true location in bridges.csv
    e = edges[b["edge"]]
    na, nb = nodes[e["from"]], nodes[e["to"]]
    mlat, mlon = round((na["y"] + nb["y"]) / 2, 5), round((na["x"] + nb["x"]) / 2, 5)
    moved_km = round(((mlon - b["x"]) * 111.195 * cos(radians(b["y"]))) ** 2 + ((mlat - b["y"]) * 111.195) ** 2, 6) ** 0.5
    row = {"id": k, "lat": mlat, "lon": mlon, "edge": on_edge[b["edge"]][0],
           "chainage_km": chainage(b["edge"], mlat, mlon), "hazus_class": hc}
    vs = [num(r["Vs30 (m/s)"]) for r in recs if pd.notna(num(r["Vs30 (m/s)"]))]
    if vs:
        row["vs30"] = float(vs[0])
        row["vs30_source"] = "Bas et al. (2024) database"
    Bridge.append(row)
    brows.append(dict(id=k, edge=row["edge"], chainage_km=row["chainage_km"], piece=b["edge"], lat=mlat, lon=mlon,
                      true_lat=b["y"], true_lon=b["x"],
                      moved_km=round(moved_km, 2), location_source=b["source"],
                      osm_way=b["osm_way"] or "", osm_length_m=b["length_m"] if b["length_m"] is not None else "",
                      road=b["road"] or "", hazus_class=hc, class_from=how, class_basis=basis,
                      paper_site_ids="; ".join(q["site_id"] for q in b["paper"]),
                      paper_names="; ".join(q["name"] for q in b["paper"]),
                      paper_open_closed="; ".join(q["open_closed"] for q in b["paper"]),
                      paper_collapse="; ".join(q["collapse"] for q in b["paper"])))

pd.DataFrame(brows).to_csv(out / "bridges.csv", index=False, encoding="utf-8")
cls_count = Counter(r["hazus_class"] for r in Bridge)
how_count = Counter(r["class_from"] for r in brows)
print("network bridge classes:", dict(cls_count.most_common()), "| from:", dict(how_count))
mv = pd.DataFrame(brows)["moved_km"]
print(f"bridge moved to edge midpoint: median {mv.median():.2f} km, 90% {mv.quantile(0.9):.2f} km, max {mv.max():.2f} km; "
      f"> 1 km: {(mv > 1).sum()}, > 5 km: {(mv > 5).sum()}")
with_bridge = {r["edge"] for r in Bridge}
n_gf_only = sum(1 for r in Edge if not r.get("fixed_open") and r["id"] not in with_bridge)

# ---------------- ground-failure components and support routes ----------------
gcal = json.load(open(net / "gf_calibration.json", encoding="utf-8")) if (net / "gf_calibration.json").is_file() else None
GroundFailure, grows = [], []
for k, g in ground.items():
    row = {"id": k, "lat": g["y"], "lon": g["x"], "edge": on_edge[g["edge"]][0], "chainage_km": chainage(g["edge"], g["y"], g["x"]),
           "kind": g["kind"], "road_class": g["road_class"], "length_km": round(float(g.get("length_m") or 0.0) / 1000.0, 3)}
    if g["kind"] == "liquefaction":
        row.update(x0=float(g["x0"]), liq_class=g["liq_class"], dw_km=round(float(g["dw_km"]), 3))
    elif g["kind"] == "landslide":
        row.update(x0=float(g["x0"]), slope_deg=float(g["slope_deg"]), ac_g=float(g["ac_g"]))
    else:
        row.update(slip_type=str(g.get("slip_type") or ""), fault_id=str(g.get("fault_id") or ""))
    GroundFailure.append(row)
    grows.append({**row, "piece": g["edge"], "road": g.get("road") or "", "highway": g.get("highway"), "f_ref": g.get("f_ref"),
                  "vs30": g.get("vs30"), "dw_km": g.get("dw_km"), "slope_max_deg": g.get("slope_max_deg"),
                  "fault_catalog": g.get("fault_catalog"), "net_slip_rate": g.get("net_slip_rate")})
pd.DataFrame(grows).to_csv(out / "ground_failure.csv", index=False, encoding="utf-8")
# each edge's components in road order (the order of its series table)
for er in erows:
    for col, rows in (("bridges", Bridge), ("ground_failure", GroundFailure)):
        er[col] = " ".join(r["id"] for r in sorted((r for r in rows if r["edge"] == er["id"]), key=lambda r: r["chainage_km"]))
    on = sorted((r for r in Bridge + GroundFailure if r["edge"] == er["id"]), key=lambda r: (r["chainage_km"], r["id"]))
    er["n_components"], er["components_in_road_order"] = len(on), " ".join(r["id"] for r in on)
pd.DataFrame(erows).to_csv(out / "edges.csv", index=False, encoding="utf-8")
n_comp = sorted((er["n_components"] for er in erows if er["n_components"]), reverse=True)
sources = sorted(k for k in nodes if k.startswith("S"))
towns = sorted(k for k in nodes if k.startswith("T"))
Support = [{"id": f"{t}-{s}", "origin": t, "source": s} for t in towns for s in sources] +           [{"id": f"{t}-any", "origin": t} for t in towns]
gcount = Counter(r["kind"] for r in GroundFailure)
print("ground-failure components:", dict(gcount), "| support routes:", len(Support))
cz = gcal["calibration"]["zhu"] if gcal else {}
cj = gcal["calibration"]["jes"] if gcal else {}

network = {
    "$schema": "../../schema/meta/network.schema.json",
    "network": "kahramanmaras-2023",
    "title": "Kahramanmaraş earthquake road network",
    "description": ("Relief access from three entry points (Adana, Ceyhan, Gaziantep) to the 46 towns that felt MMI VII "
                    "or more in the 6 February 2023 Kahramanmaraş earthquakes (Türkiye): roads with their bridges and "
                    "ground-failure components (liquefaction, landslide, fault crossings), and one support route per "
                    "town and source. shakemap.assembly.json binds the Bayesian network."),
    "source": {
        "roads": "OpenStreetMap, Geofabrik turkey-latest.osm.pbf (file dated 2026-11-12); motorway, trunk, primary and secondary roads with their _link ramps",
        "shaking": "USGS ShakeMap grids, events us6000jllz (M7.8 Pazarcık) and us6000jlqa (M7.5 Elbistan): used to choose the destinations (max MMI)",
        "bridge_observations": "Bas et al. (2024) Earthquake Spectra 40(4) 2339-2363, doi:10.1177/87552930241262043; database DesignSafe PRJ-4622, doi:10.17603/ds2-9yr3-vp92",
        "build": "examples/kahramanmaras-2023/build/ (see README.md for the run order)",
        "dem": "Copernicus GLO-90 DEM (3 arc-second), AWS Open Data copernicus-dem-90m: slope and the slope-based Vs30",
        "faults": "GEM Global Active Faults database (github.com/GEMScienceTools/gem-global-active-faults, commit 5681650, CC BY-SA 4.0); in this region its traces are the EMME catalogue's",
        "ground_failure_models": "Zhu et al. (2017) and Nowicki Jessee et al. (2018) coefficients from USGS gfail (code.usgs.gov/ghsc/esi/groundfailure, commit fa7bd28: zhu_2017.py, jessee_2018.py, jessee_2018_slim.py); Hazus liquefaction tables and Jibson (2007) Eq. 6 from the OpenQuake engine (github.com/gem/oq-engine, commit 0f9c75e: openquake/sep); Wells & Coppersmith (1994) Table 2B read from the paper; Allen & Wald (2007, USGS OFR 2007-1357) Table 2 read from the report",
        "ground_failure_calibration": "USGS ground-failure products of the 2023 main shocks: us6000jllz (ground-failure v14, run on ShakeMap v17) and us6000jlqa (v7, on ShakeMap v11), with those ShakeMap versions' PGV",
    },
    "assumptions": {
        "study_area": "Box 35.2-39.0 E, 35.8-38.8 N: the MMI VII area of both main shocks, widened west to include Adana. Türkiye only; the Syrian part of the box has no roads in the data.",
        "road_network": "Today's OpenStreetMap (2026), not the network as of February 2023. Roads: motorway to secondary (Turkish provincial roads), the level most district towns need.",
        "key_destination": "The three sources S1-S3 are the key destinations: relief entry points at the O-52 junctions serving Adana airport and İncirlik (S1), the Ceyhan and Yumurtalık ports (S2) and Gaziantep airport (S3). Travel is undirected, so 'reach the nearest key destination' reads as 'be reached from the nearest source'.",
        "destinations": "T01-T46 are the towns to reach: every OSM place=city/town node in the box with max MMI >= VII in either main shock (USGS ShakeMap). City and its metropolitan districts merged into one node (Antakya, Kahramanmaraş, Gaziantep, Malatya); one Syrian town dropped. Villages are represented by their district town. An assembly should bind the system function to the T nodes only.",
        "routing_nodes": f"J nodes are road junctions: 7 placed by inspection of the OSM map and the rest added where two links from one node shared more than 1 km of road before branching. A junction where only two road pieces meet is not a place the route can split, so it is merged into the edge through it: {len(dropped_junctions)} of them ({', '.join(dropped_junctions)}). The build's shape points (X) mark where components sit along the road and are merged into the edges too; they stay as the points of Edge.geometry. No J node is an origin or a destination.",
        "topology": "Links between S, T and J nodes were chosen by inspection of the OSM map; each link follows the shortest road route, with motorway, trunk, primary and secondary roads weighted 0.8, 0.9, 1.0 and 1.3 per km so main roads are preferred.",
        "component_representation": f"The build splits each road into short pieces, one centred on each component (a bridge or a ground-failure component; a ground-failure component within 250 m of a bridge shares its piece), with connectors between. An edge runs between two nodes where the route can split or ends: a node where three or more pieces meet (a junction), a town or a source. The pieces between are merged into one edge with every component on them: {len(Edge)} edges, the largest carrying {n_comp[0] if n_comp else 0} components. The edge is open only if every component on it is: a series system (edge.open), whose table lists the components in road order (chainage_km, the distance from the edge's source) and has one row per component plus one (composite states), not one per combination of their states.",
        "bridge_representation": "Each bridge's piece runs 0.5 km either side of the bridge centre, or less where the neighbouring bridge (half the gap) or the link end is closer. The pieces only place the bridges along the road; they are merged into the topological edges (component_representation). bridges.csv gives each bridge's piece (build id).",
        "edge_representation": f"Only an edge something can close is a random variable: the {n_random} edges with a bridge or a ground-failure component ({len(with_bridge)} with a bridge, {n_gf_only} with ground-failure components only), numbered E01-E{n_random:02d}. The other {len(Edge) - n_random} edges (C01-C{len(Edge) - n_random:02d}) carry nothing that can fail: they are marked fixed_open, stay in the graph every system function and rsr evaluates, always open, and have no node in the Bayesian network. edges.csv lists each edge's pieces (build ids) and its components in road order.",
        "osm_bridges": "An OSM bridge is a run of consecutive bridge-tagged road segments of one OSM way on the route. A route uses one carriageway of a divided road, so a twin bridge counts once.",
        "paper_bridges": "A Bas et al. road bridge within 100 m of a route is put on that link. If an OSM bridge on the same link is within 150 m along the route, the record is attached to it (same structure); otherwise the record becomes its own bridge, at the route point nearest the paper's coordinates. The remaining records are not on this network (mostly city streets); bridges.csv lists which records each bridge carries.",
        "bridge_location": f"Each bridge sits at the midpoint of its piece's end nodes, as in the EMA and Hebei networks. Because its piece is centred on it, this is within {mv.max()*1000:.0f} m of its true road position (median {mv.median()*1000:.0f} m): the midpoint of the OSM way along the route, or for a paper-only bridge the route point nearest the paper's coordinates. bridges.csv gives true_lat, true_lon and moved_km.",
        "hazus_class": f"Mapped from the Bas et al. record: HAZUS non-California rows; seismic design if built in or after {SEISMIC_FROM} (the paper notes post-2000 bridges should follow modern codes; HAZUS's own non-CA cut-off is 1990; a year that is not a number, one record 'Tarihi?' (historic), is taken as conventional); major bridge if the average span exceeds 150 m (the data give no maximum span); single span -> HWB3/HWB4; girder type -> precast or prestressed girders = simply supported prestressed concrete (HWB17/HWB19), cast-in-place or box concrete = continuous concrete (HWB10/HWB11), post-tensioned box = continuous prestressed (HWB22/HWB23), steel I = simply supported steel (HWB12/HWB14/HWB24), steel box = continuous steel (HWB15/HWB16/HWB26), arch and truss = HWB28. Column count is not used: single-column and wall piers take the multi-column class of their girder type.",
        "hazus_default": f"Bridges with no structural data (OSM only) and records whose girder type was not observed get the most common class among the paper's {len(p)} road bridges: {DEFAULT}.",
        "hazus_multiple_records": "A bridge with several paper records (twin bridges, or several teams) takes the most common class among them.",
        "vs30": "Given only where a Bas et al. record is attached (their estimate: measured at stations or the Zhou 2023 model); the rest are left for python -m src vs30.",
        "length_km": "Length of the routed road between the edge's end nodes: the sum of its pieces.",
        "capacity": "1 per edge, assumed: no capacities are in the data.",
        "avg_speed_kph": "Each piece's speed is assumed by the road class with the most length on it: motorway 100, trunk 80, primary 70, secondary 50 km/h (no measured speeds). An edge's speed is its length over the sum of its pieces' travel times, so its travel time is theirs.",
        "population": "OSM population tag of the town node; for most towns it appears to be the whole district.",
        "coordinates": "WGS84 lat/lon of the routed road graph nodes (S, J) and of the OSM town nodes (T); Edge.geometry is the edge's course through the build's nodes, for drawing only. A component's chainage_km is its position along its edge from the edge's source: a bridge's at its piece's midpoint, a ground-failure component's its location projected on its piece.",
        "gf_placement": f"Ground-failure components are placed by exposure only, never by what happened in 2023. Stations every 100 m along every link; a station is liquefaction-exposed if the Zhu et al. (2017) coverage at a reference shaking (PGV 50 cm/s, M 7.0) is at least 5% inside the model's domain (Vs30 <= 620 m/s, 30-arc-second slope <= 5 deg), and landslide-exposed if the Nowicki Jessee et al. (2018) coverage at PGV 50 cm/s is at least 5% (slope >= 2 deg). Consecutive exposed stations form a stretch, split at 1 km; each stretch is one component. Every crossing of a route with a GEM active-fault trace is a fault component. Result: {gcount.get('liquefaction', 0)} liquefaction, {gcount.get('landslide', 0)} landslide and {gcount.get('fault', 0)} fault components. The 5% screens are a modelling choice that keeps the network a manageable size; a 2% landslide screen would give about 400 stretches.",
        "gf_site_inputs": "Slope: Copernicus GLO-90, mean over 3 x 3 cells (about 270 m) at each station. Vs30: from the slope at 30 arc-seconds by Allen & Wald (2007) Table 2 (active tectonic), log-linear between its breakpoints, clamped to 180-760 m/s. Distance to water: to the nearest OSM river, canal, lake or coastline, standing in for Zhu's min(distance to coast, distance to river). Rainfall, water-table depth (Zhu) and rock type, land cover and wetness index (Jessee) are not mapped per site: each model takes one regional constant for them (gf_calibration). Without the MTA geology map, rock type cannot be resolved locally.",
        "gf_calibration": f"The regional constants are the median, over road stations where the USGS 2023 ground-failure maps are non-zero, of the site term those maps imply (their coverage inverted to the logit, less the shaking terms at the ShakeMap PGV they were run on) minus the part computed here: Zhu C = {cz.get('constant', float('nan')):.3f} over {cz.get('n_stations', 0)} stations (interquartile {cz.get('iqr', [0, 0])[0]:.2f} to {cz.get('iqr', [0, 0])[1]:.2f}), Jessee C = {cj.get('constant', float('nan')):.3f} over {cj.get('n_stations', 0)} stations ({cj.get('iqr', [0, 0])[0]:.2f} to {cj.get('iqr', [0, 0])[1]:.2f}). The USGS maps are model output, not observed failures, so no 2023 outcome enters the prior; the constants also absorb the differences between these simple inputs and USGS's global layers.",
        "liquefaction_model": "Occurrence: Zhu et al. (2017) general model, coverage f = 0.4915 / (1 + 42.4 exp(-9.165 P))^2, P = logistic(x0 + 0.334 ln(PGV MSF)), MSF = 1 / (1 + exp(-2 (M - 6))), PGV clipped at 150 cm/s, zero below PGV 3 cm/s or PGA 0.10 g; x0 = 8.801 - 1.918 ln Vs30 - 0.2054 distance to water (km) + C. A stretch of length L liquefies with P = 1 - (1 - f)^(L / 100 m): 100 m patches taken as independent (assumed). Given it liquefies, PGD = max(lateral spread, settlement), Hazus: lateral spread K_delta(M) a(PGA / PGA_t) inches, a = 12r - 12 (1 < r <= 2), 18r - 24 (2 < r <= 3), 70r - 180 (r > 3), K_delta = 0.0086 M^3 - 0.0914 M^2 + 0.4698 M - 0.9835; PGA_t 0.09/0.12/0.15/0.21/0.26 g and settlement 12/6/2/1/0 in for classes vh/h/m/l/vl. Class from the Zhu coverage at the reference shaking (assumed bands): >= 20% vh, 10-20% h, 5-10% m, 2-5% l, < 2% vl. Hazus applies lateral spread near a free face. Here it applies only to a stretch within 200 m of water (dw_km: the stretch's mean distance to the nearest OSM river, canal, lake or coastline, standing in for the distance to a free face; assumed); a stretch farther away only settles. Applied to every stretch, lateral spread gave 0.5-0.9 m at PGA 0.35-0.45 g (class m, M7.8) on stretches with no free face, enough to close roads that stayed open in 2023.",
        "landslide_model": "Occurrence: Nowicki Jessee et al. (2018), USGS slim form, coverage f = exp(-7.592 + 5.237 P - 3.042 P^2 + 4.035 P^3), P = logistic(x0 + 1.65 ln PGV + 0.01 ln PGV x slope), PGV clipped at 211 cm/s, zero below PGA 0.02 g; x0 = -6.30 + 0.06 slope + C (the interaction coefficient is +0.01, as the USGS code has it). Per stretch as for liquefaction (100 m patches). Given it slides, PGD is lognormal about the Newmark displacement of Jibson (2007) Eq. 6, log10 D(cm) = 0.215 + log10((1 - r)^2.341 r^-1.438), r = ac / PGA (floored at 0.05), sigma 0.51 log10 units (as the paper reports for Eq. 6; checked here only as about 0.5); none where r >= 1. Critical acceleration ac = (FS - 1) sin(slope) with the infinite-slope FS for assumed typical soil (c' 20 kPa, phi' 32 deg, unit weight 19 kN/m3, 3 m slab, dry), at the stretch's steepest station: an assumption made without a geology map.",
        "fault_model": "A crossing is on the scenario's rupture if it lies within 2 km of the top edge of the ShakeMap's rupture.json (its fault planes' shallowest vertices). There the surface ruptures with P = 1 / (1 + exp(12.51 - 2.053 M)) (Wells & Coppersmith 1993, as used by Youngs et al. 2003 and Petersen et al. 2011; these two coefficients were not checked against the primary paper) and offsets the road by a lognormal amount about the offset at the trace: 0.8 of its segment's average offset, the rest taken as spread off the mapped trace (assumed: it puts the Amanos median at 3.0 m, inside the 2.5-3.7 m measured on roads there). For the M7.8 (us6000jllz) the average is the segment's measured one, by the crossed GEM fault: Amanos (ME_TRCS056) 3.8 m, Pazarcık (ME_TRCS055) 6.0 m, Erkenek (ME_TRCS062) 5.3 m, sigma 0.3 in ln units (assumed, about the spread of the road offsets). These are Provost et al. (2024)'s segment averages as a review of this example quotes them, not checked against the paper. Elsewhere, and for any other event, the average is the Wells & Coppersmith (1994) Table 2B average displacement: strike-slip log10 AD = -6.32 + 0.90 M (sigma 0.28), all slip types -4.80 + 0.69 M (sigma 0.36), checked against the paper. Strike-slip is used where the GEM slip type is sinistral or dextral. One offset stands for a whole segment (no along-strike profile). The measured offsets are outcomes of 2023, so at the fault crossings the M7.8 scenario is a hindcast, not a forecast. A crossing off the rupture does not move.",
        "road_fragility": "Liquefaction and landslide components: Hazus 6.1 Table 7-5 roadway curves on PGD (instances/fragility/hazus61-road-pgd.json): HRD1 (motorway, trunk) extensive/complete median 60 in (1.524 m), HRD2 (primary, secondary) 24 in (0.61 m), beta 0.7; the road is closed at extensive/complete, as bridges are (policy.closed-at-extensive). Fault crossings: a curve of their own (instances/fragility/road-fault-offset.json), impassable within 72 h at a median offset of 4.5 m, beta 0.7, either road class. The Hazus curve treats about 1.5 m of any ground movement as closing a major road; in 2023 the offset was mostly horizontal, and roads offset about 3 m sideways mostly stayed usable or were patched within hours. The 4.5 m median is a rough fit to the 2023 crossings (about 3 of the 8 fault-crossing edges with a known outcome closed), so those outcomes are no test of it. Each mechanism's occurrence and displacement models form one variable, its PGD: 0 unless the ground fails (for a landslide: fails and slides, PGA above the critical acceleration), so the ground failed exactly when PGD > 0.",
        "closure_window": "Closed means not passable for relief traffic within the first 72 hours after the shock: the relief window the support routes are about. KGM reopened every road closed on 6 February 2023 by 10 February (as a review of this example reports; not checked), so on a one-week window almost no road closure from ground failure would count. The fault-crossing curve is fitted to this window; the Hazus curves (bridges at extensive damage, roads on liquefaction and landslide PGD) are damage states, taken as lasting through it.",
        "ground_motion_fields": "SA(1.0) at the bridges, PGV and PGA at the liquefaction and landslide components: three independent Gaussian fields from the same ShakeMap, each spatially correlated by Jayaram & Baker (2009): its SA(1.0) set for SA(1.0) and for PGV (it has no PGV set; T = 1 s taken as the proxy), its PGA set for PGA. The correlation between the IMs at one site is ignored.",
        "support_routes": f"Support has one row per town and source (46 x 3), plus one per town with no source (T..-any): {len(Support)} rows. system.reachable judges whether a town is still connected to the source over open edges, whatever the detour, or for the no-source row to at least one source: one system layer per source, so the sources can be compared, and one for any source.",
        "rsr_rules": "Each system node's rules are extracted by rsr (rsr/<Support id>/reachable/). The search draws each random edge open with probability 0.95, independently (rsr/proposal.json): a proposal for the search only, which draws both states of every edge; the rules are exact whatever it is. rsr stops on its own quick estimate of the unclassified probability, which can fall below the target early, so each system's rules are then checked on 1e6 fresh samples from the proposal and, while the unclassified share is 1e-5 or more, rsr resumes from them with a new seed and they are checked again on new samples; the cap is rsr's default 10000 rounds over all runs. Each system's model.json lists its runs and their seeds. The states the rules leave undecided are learnt from: rsr resolves them into new rules, until every state sampled is decided by a rule.",
    },
    "tables": {"Node": Node, "Edge": Edge, "Bridge": Bridge, "GroundFailure": GroundFailure, "Support": Support},
}
json.dump(network, open(out / "network.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {out / 'network.json'}: {len(Node)} nodes, {len(Edge)} edges, {len(Bridge)} bridges")
