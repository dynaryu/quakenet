"""Schematic EQ network: topology by inspection, geometry + bridges from OSM, split so each edge has <=1 bridge."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx, osmnx as ox, geopandas as gpd
from shapely.geometry import LineString, Point

osm, out, paper_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

# ---------- nodes ----------
nodes = {"S1": ("Adana (O-52 Adana Dogu jct)", 35.427, 37.028, "source"),
         "S2": ("Ceyhan (O-52 Ceyhan-Yumurtalik jct)", 35.830, 36.983, "source"),
         "S3": ("Gaziantep (O-52 Havalimani jct)", 37.432, 37.017, "source")}
for _, r in pd.read_csv(osm / "destinations.csv").iterrows():
    nodes[r.id] = (r["name"], r.lon, r.lat, "destination")
junctions = {"J1": ("O-52/O-53 Hatay Ayrim", 35.951, 37.004), "J2": ("Belen road / D825", 36.32, 36.44),
             "J3": ("D400 Duzici turn", 36.45, 37.185), "J4": ("O-52/D835 Narli", 37.16, 37.28),
             "J5": ("D835/D360 Pazarcik turn", 37.16, 37.40), "J6": ("D360/D850 Besni", 37.98, 37.72),
             "J7": ("D850 / Nurhak road", 37.72, 37.86)}
for k, (n, x, y) in junctions.items():
    nodes[k] = (n, x, y, "junction")

# ---------- links chosen by inspection of the OSM maps ----------
links = """S1-T44 S1-S2 S2-T44 S2-T43 S2-J1 T44-T37 J1-T37 J1-T31 T37-T18 T37-T31 T31-T25 T25-T21 T21-T29
T29-T34 T29-T23 T23-J2 J2-T06 J2-T03 T03-T22 T03-T19 T03-T36 T36-T11 T11-T06 T06-T01 T01-T02 T02-T05 T02-T15
T01-T45 T45-T05 T18-J3 J3-T35 J3-T30 J3-T15 T30-T05 T05-T14 T14-T16 T05-J4 J4-T20 J4-J5 J5-T16 J5-T17 T17-T24
T17-T08 T20-S3 S3-T46 T20-T39 T39-T40 T40-J6 T08-J6 J6-T28 J6-T26 T26-T42 T42-T27 T26-T07 T07-T13 T16-T10
T10-T32 T32-T41 T41-T09 T41-T12 T12-J7 J7-T08 J7-T04 T41-T33 T33-T13 T13-T04 T13-T38""".split()

# ---------- routing graph ----------
G = ox.load_graphml(osm / "eq_box_roads_sec_unsimplified.graphml")
U = ox.convert.to_undirected(G)
U = U.subgraph(max(nx.connected_components(U), key=len)).copy()


def first(v):
    return v[0] if isinstance(v, list) else v


def is_bridge(v):
    v = first(v)
    return isinstance(v, str) and v != "no"


fac = {"motorway": 0.8, "trunk": 0.9, "primary": 1.0, "secondary": 1.3}
for u, v, k, d in U.edges(keys=True, data=True):
    d["w"] = d["length"] * fac.get(str(first(d["highway"])).replace("_link", ""), 1.3)
nid = np.array(list(U.nodes))
xy = np.array([[U.nodes[n]["x"], U.nodes[n]["y"]] for n in nid])
big = np.array([any(str(first(d["highway"])).startswith(("motorway", "trunk")) for _, _, d in U.edges(n, data=True)) for n in nid])


def snap(x, y, major=False):
    dd = np.hypot((xy[:, 0] - x) * np.cos(np.radians(y)), xy[:, 1] - y) * 111_195
    if major:
        dd = np.where(big, dd, np.inf)
    i = dd.argmin()
    return nid[i], dd[i]


gnode = {}
for k, (n, x, y, role) in nodes.items():
    gnode[k], dist = snap(x, y, major=(role == "junction"))
    if dist > 1500:
        print(f"  warn: {k} {n} snapped {dist:.0f} m")


def best(u, v):
    return min(U[u][v].values(), key=lambda a: a["w"])


# ---------- route links; add a junction wherever two links from the same node share road ----------
from shapely.geometry import Point as _P


def route(a, b):
    return nx.shortest_path(U, gnode[a], gnode[b], weight="w")


def plen(path, i):  # metres along path up to index i
    return sum(best(u, v)["length"] for u, v in zip(path[:i], path[1:i + 1]))


paths = {lk: route(*lk.split("-")) for lk in links}
jn = 0
changed = True
while changed:
    changed = False
    lines = {lk: gpd.GeoSeries([LineString([(U.nodes[n]["x"], U.nodes[n]["y"]) for n in p])], crs=4326).to_crs(32637).iloc[0]
             for lk, p in paths.items()}
    for la in list(paths):
        for lb in list(paths):
            if la >= lb or la not in paths or lb not in paths:
                continue
            ea, eb = la.split("-"), lb.split("-")
            common = set(ea) & set(eb)
            if len(common) != 1:
                continue
            c = common.pop()
            pa = paths[la] if ea[0] == c else paths[la][::-1]
            pb = paths[lb] if eb[0] == c else paths[lb][::-1]
            lbline = lines[lb]
            pts = gpd.GeoSeries([_P(U.nodes[n]["x"], U.nodes[n]["y"]) for n in pa], crs=4326).to_crs(32637)
            k = 0
            while k + 1 < len(pa) and pts.iloc[k + 1].distance(lbline) < 60:
                k += 1
            if plen(pa, k) < 1000:
                continue
            same = [nm for nm, g in gnode.items() if g == pa[k]]
            if same:
                jid = same[0]
            else:
                jn += 1
                jid = f"J{7 + jn}"
                nodes[jid] = ("branch point (auto)", U.nodes[pa[k]]["x"], U.nodes[pa[k]]["y"], "junction")
                gnode[jid] = pa[k]
            oa = ea[1] if ea[0] == c else ea[0]
            ob = eb[1] if eb[0] == c else eb[0]
            del paths[la], paths[lb]
            have = {frozenset(x.split("-")) for x in paths}
            for nl in (f"{c}-{jid}", f"{jid}-{oa}", f"{jid}-{ob}"):
                u_, v_ = nl.split("-")
                if u_ != v_ and gnode[u_] != gnode[v_] and frozenset((u_, v_)) not in have:
                    paths[nl] = route(u_, v_)
                    have.add(frozenset((u_, v_)))
            print(f"  auto junction {jid}: {la} & {lb} shared {plen(pa, k)/1000:.1f} km from {c}")
            changed = True
            break
        if changed:
            break
links = list(paths)

# the routed links, for build/ground_failure.py to place ground-failure components along
gpd.GeoDataFrame([dict(link=lk, geometry=LineString([(U.nodes[n]["x"], U.nodes[n]["y"]) for n in paths[lk]]))
                  for lk in links], crs=4326).to_file(out / "links.gpkg", layer="links", driver="GPKG")

# ground-failure components from build/ground_failure.py (optional 4th argument): link, kind, s_m, ...
gf_by_link = {}
if len(sys.argv) > 4:
    gfc = pd.read_csv(sys.argv[4])
    for rec in gfc.to_dict("records"):
        gf_by_link.setdefault(rec["link"], []).append(rec)
    print(f"ground-failure components read: {len(gfc)} ({gfc['kind'].value_counts().to_dict()})")
GF_ATTACH_M = 250.0     # a component this close to a bridge shares its edge (in series)

# ---------- paper bridges (Bas et al. 2024): assign each to the nearest link if within 60 m ----------
p = pd.read_csv(paper_csv, header=2, encoding="cp1254")
p.columns = [" ".join(str(c).split()) for c in p.columns]
p = p.dropna(subset=["Latitude (decimal)"])
p = p[p["Pedestrian or Vehicular Bridge?"].astype(str).str.strip() == "Vehicular"].reset_index(drop=True)
pg = gpd.GeoSeries([Point(x, y) for x, y in zip(p["Longitude (decimal)"], p["Latitude (decimal)"])], crs=4326).to_crs(32637)
lline = gpd.GeoSeries([LineString([(U.nodes[n]["x"], U.nodes[n]["y"]) for n in paths[lk]]) for lk in links], crs=4326).to_crs(32637)
paper_on, near_rows = {lk: [] for lk in links}, []
for i, pt in enumerate(pg):
    r = p.iloc[i]
    rec = dict(site_id=str(r["Bridge Site ID"]), name=str(r["Bridge Name"]),
               open_closed=str(r["Bridge Open or Closed at Time of Site Visit? (Open/Closed)"]),
               collapse=str(r["Bridge Collapse or Partial Collapse? (Yes/No)"]),
               lon=float(r["Longitude (decimal)"]), lat=float(r["Latitude (decimal)"]))
    d = lline.distance(pt)
    j = int(d.values.argmin())
    if d.iloc[j] < 100:
        paper_on[links[j]].append(rec)
    else:
        near_rows.append(dict(**rec, dist_to_network_m=round(d.iloc[j]), nearest_link=links[j]))
n_paper_on = sum(len(v) for v in paper_on.values())

# ---------- route links, find bridges, split ----------
out_nodes = {k: dict(name=n, x=round(x, 5), y=round(y, 5), role=r) for k, (n, x, y, r) in nodes.items()}
edges, bridges, link_rows, off_centre = {}, {}, [], []
ecount = bcount = xcount = gcount = 0
ground = {}
pos = {k: (v[1], v[2]) for k, v in nodes.items()}
for lk in links:
    a, b = lk.split("-")
    path = paths[lk]
    coords = [(U.nodes[n]["x"], U.nodes[n]["y"]) for n in path]
    seg = [best(u, v) for u, v in zip(path[:-1], path[1:])]
    cum = np.concatenate([[0], np.cumsum([s["length"] for s in seg])])
    L = cum[-1]
    straight = np.hypot((pos[a][0] - pos[b][0]) * np.cos(np.radians(37.3)), pos[a][1] - pos[b][1]) * 111.195
    near = [k for k, (x, y) in pos.items() if k not in (a, b) and
            min(np.hypot((cx - x) * np.cos(np.radians(y)), cy - y) for cx, cy in coords[::3]) * 111.2 < 1.0]
    refs = []
    for s in seg:
        r = first(s.get("ref")) or str(first(s["highway"]))
        if not refs or refs[-1] != r:
            refs.append(r)
    link_rows.append(dict(link=lk, km=round(L / 1000, 1), detour=round(L / 1000 / max(straight, 0.1), 2),
                          passes_near=",".join(near),
                          roads=" > ".join(dict.fromkeys(r for r in refs if not str(r).endswith("_link")))))
    # bridge runs along the path: consecutive bridge segments of the same OSM way form one bridge
    runs = []
    for i, s in enumerate(seg):
        if is_bridge(s.get("bridge")):
            way = first(s["osmid"])
            if runs and runs[-1]["way"] == way and runs[-1]["i1"] == i:
                runs[-1]["i1"] = i + 1
            else:
                runs.append(dict(way=way, i0=i, i1=i + 1, ref=first(s.get("ref")), hw=str(first(s["highway"])), paper=[]))
    # paper bridges on this link: merge into a run within 150 m along the path, else add as own run
    cxy = np.array(coords)
    for rec in paper_on[lk]:
        k = int(np.argmin(np.hypot((cxy[:, 0] - rec["lon"]) * np.cos(np.radians(rec["lat"])), cxy[:, 1] - rec["lat"])))
        def gap(rr):
            if cum[rr["i0"]] <= cum[k] <= cum[rr["i1"]]:
                return 0.0
            return min(abs(cum[k] - cum[rr["i0"]]), abs(cum[k] - cum[rr["i1"]]))
        cand = sorted(runs, key=gap)
        if cand and gap(cand[0]) < 150:
            cand[0]["paper"].append(rec)
        else:
            k1 = min(k + 1, len(path) - 1)
            k0 = k1 - 1
            runs.append(dict(way=None, i0=k0, i1=k1, k=k, ref=first(seg[k0].get("ref")), hw=str(first(seg[k0]["highway"])), paper=[rec]))
    # ----- one short edge per bridge, centred on it along the road; bridge-free connectors between -----
    HALF = 500.0                                    # metres either side of the bridge centre, at most

    def point_at(s):
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
        t = 0.0 if cum[i + 1] == cum[i] else (s - cum[i]) / (cum[i + 1] - cum[i])
        (x0, y0), (x1, y1) = coords[i], coords[i + 1]
        return x0 + t * (x1 - x0), y0 + t * (y1 - y0)

    def piece(lo, hi):
        inner = [coords[i] for i in range(len(cum)) if lo < cum[i] < hi]
        return [point_at(lo)] + inner + [point_at(hi)]

    for r in runs:
        r["s"] = (cum[r["i0"]] + cum[r["i1"]]) / 2 if r["way"] is not None else cum[r["k"]]
        r["is_bridge"], r["gf"] = True, []
    # ground-failure components: on a bridge's edge if within GF_ATTACH_M of it, else their own short edge
    for comp in sorted(gf_by_link.get(lk, []), key=lambda c: c["s_m"]):
        s = float(np.clip(comp["s_m"], 0.0, L))
        near = min(runs, key=lambda rr: abs(rr["s"] - s), default=None)
        if near is not None and abs(near["s"] - s) <= GF_ATTACH_M:
            near["gf"].append({**comp, "s": s})
            continue
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
        runs.append(dict(way=None, i0=i, i1=i + 1, s=s, ref=first(seg[i].get("ref")), hw=str(first(seg[i]["highway"])),
                         paper=[], is_bridge=False, gf=[{**comp, "s": s}]))
    runs.sort(key=lambda rr: rr["s"])
    spans = []
    for j, r in enumerate(runs):
        d = min(HALF, r["s"], L - r["s"])
        if j > 0:
            d = min(d, (r["s"] - runs[j - 1]["s"]) / 2)
        if j + 1 < len(runs):
            d = min(d, (runs[j + 1]["s"] - r["s"]) / 2)
        lo, hi = r["s"] - d, r["s"] + d
        if d < 5:                                   # bridge at a link end: cannot be centred, take one side
            lo, hi = (0.0, min(2 * HALF, L)) if r["s"] < L / 2 else (max(L - 2 * HALF, 0.0), L)
            if j > 0:
                lo = max(lo, (r["s"] + runs[j - 1]["s"]) / 2)
            if j + 1 < len(runs):
                hi = min(hi, (r["s"] + runs[j + 1]["s"]) / 2)
            off_centre.append(f"{lk}: bridge {r['s']:.0f} m along a {L:.0f} m link")
        spans.append((lo, hi, r))
    pieces, s_prev = [], 0.0                        # (lo, hi, run or None)
    for lo, hi, r in spans:
        if lo - s_prev > 1:
            pieces.append((s_prev, lo, None))
        pieces.append((max(lo, s_prev), hi, r))
        s_prev = hi
    if L - s_prev > 1 or not pieces:
        pieces.append((s_prev, L, None))
    elif pieces:
        lo, _, r = pieces[-1]
        pieces[-1] = (lo, L, r)                     # absorb a sub-metre tail
    if pieces[0][0] > 0:
        pieces[0] = (0.0,) + pieces[0][1:]
    stops = [a]
    for lo, hi, r in pieces[:-1]:
        xcount += 1
        xid = f"X{xcount:03d}"
        x, y = point_at(hi)
        out_nodes[xid] = dict(name=f"{'component end' if r else 'split'} on {a}-{b}", x=round(x, 5), y=round(y, 5), role="split")
        stops.append(xid)
    stops.append(b)
    for j, (lo, hi, r) in enumerate(pieces):
        bid = None
        if r is not None and r["is_bridge"]:
            bcount += 1
            bid = f"B{bcount:03d}"
            x, y = point_at(r["s"])
            bridges[bid] = dict(edge=None, osm_way=f"w{r['way']}" if r["way"] is not None else None, road=r["ref"], highway=r["hw"],
                                length_m=round(cum[r["i1"]] - cum[r["i0"]], 1) if r["way"] is not None else None,
                                x=round(x, 5), y=round(y, 5), source="OSM" if r["way"] is not None else "Bas et al. 2024",
                                paper=[{kk: vv for kk, vv in q.items() if kk not in ("lon", "lat")} for q in r["paper"]])
        ecount += 1
        eid = f"E{ecount:03d}"
        cls_km, ref_km = {}, {}
        for i, s in enumerate(seg):
            ov = min(cum[i + 1], hi) - max(cum[i], lo)
            if ov <= 0:
                continue
            c = str(first(s["highway"])).replace("_link", "")
            cls_km[c] = cls_km.get(c, 0) + ov / 1000
            rf = first(s.get("ref"))
            if isinstance(rf, str):
                ref_km[rf] = ref_km.get(rf, 0) + ov / 1000
        edges[eid] = {"from": stops[j], "to": stops[j + 1], "directed": False, "link": lk,
                      "length_km": round((hi - lo) / 1000, 3), "bridge": bid,
                      "road_class_km": {k: round(v, 3) for k, v in cls_km.items()},
                      "road_refs": [k for k, _ in sorted(ref_km.items(), key=lambda kv: -kv[1])],
                      "geometry": LineString(piece(lo, hi))}
        if bid:
            bridges[bid]["edge"] = eid
        for comp in (r["gf"] if r is not None else []):
            gcount += 1
            x, y = point_at(comp["s"])
            i = int(np.clip(np.searchsorted(cum, comp["s"], side="right") - 1, 0, len(seg) - 1))
            hw = str(first(seg[i]["highway"])).replace("_link", "")
            ground[f"G{gcount:03d}"] = {**{k: v for k, v in comp.items() if k not in ("link", "s_m", "s")},
                                        "edge": eid, "x": round(x, 5), "y": round(y, 5),
                                        "road": first(seg[i].get("ref")), "highway": hw,
                                        "road_class": "HRD1" if hw in ("motorway", "trunk") else "HRD2"}

print(f"bridges that could not be centred (at a link end): {len(off_centre)}")
for m in off_centre:
    print('  ', m)

# ---------- paper bridge summary ----------
n_osm_with_paper = sum(1 for v in bridges.values() if v["source"] == "OSM" and v["paper"])
n_paper_only = sum(1 for v in bridges.values() if v["source"] != "OSM")
print(f"paper vehicular bridges: {len(p)}; on network {n_paper_on} "
      f"(merged into {n_osm_with_paper} OSM bridges, {n_paper_only} added as paper-only bridges); not on network {len(near_rows)}")
pd.DataFrame(near_rows).sort_values("dist_to_network_m").to_csv(out / "paper_bridges_not_on_network.csv", index=False, encoding="utf-8")

# ---------- save ----------
json.dump(out_nodes, open(out / "nodes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "geometry"} for k, v in edges.items()},
          open(out / "edges.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(bridges, open(out / "bridges.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(ground, open(out / "ground.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ground-failure components placed:", len(ground))
gpd.GeoDataFrame([dict(eid=k, **v) for k, v in edges.items()], crs=4326).to_file(out / "network.gpkg", layer="edges", driver="GPKG")
lr = pd.DataFrame(link_rows)
lr.to_csv(out / "links_check.csv", index=False, encoding="utf-8")
roles = pd.Series([v["role"] for v in out_nodes.values()]).value_counts()
print("nodes:", roles.to_dict(), "| edges:", len(edges), "| bridges:", len(bridges),
      "| edges with a bridge:", sum(e["bridge"] is not None for e in edges.values()))
print("\nlinks to check (detour > 1.6 or passing within 1 km of another node):")
print(lr[(lr.detour > 1.6) | (lr.passes_near != "")].to_string(index=False))
# links sharing road with each other (> 2 km overlap)
lg = gpd.GeoDataFrame([dict(link=v["link"], geometry=v["geometry"]) for v in edges.values()], crs=4326).to_crs(32637)
lg = lg.dissolve("link").reset_index()
buf = lg.geometry.buffer(30)
print("\nlinks overlapping > 2 km:")
for i in range(len(lg)):
    for j in range(i + 1, len(lg)):
        ov = lg.geometry[i].intersection(buf[j]).length
        if ov > 2000:
            print(f"  {lg.link[i]} & {lg.link[j]}: {ov/1000:.1f} km")
