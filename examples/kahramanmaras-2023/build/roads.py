"""Motorway..secondary roads (+links) in the EQ box -> graph + bridge lines."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, osmium, osmnx as ox
from shapely.geometry import LineString

pbf, out = sys.argv[1], Path(sys.argv[2])
W, S_, E, N = 35.2, 35.8, 39.0, 38.8
cls = {"motorway","motorway_link","trunk","trunk_link","primary","primary_link","secondary","secondary_link"}
xml = out / "eq_box_roads_sec.osm"
inbox = lambda x, y: W <= x <= E and S_ <= y <= N
bridges = []
with osmium.BackReferenceWriter(str(xml), ref_src=pbf, overwrite=True) as wr:
    for w in osmium.FileProcessor(pbf, osmium.osm.NODE | osmium.osm.WAY).with_locations() \
            .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY)).with_filter(osmium.filter.KeyFilter("highway")):
        if w.tags["highway"] not in cls: continue
        pts = [(n.lon, n.lat) for n in w.nodes if n.location.valid()]
        if not any(inbox(*p) for p in pts): continue
        wr.add_way(w)
        if w.tags.get("bridge", "no") != "no" and len(pts) > 1:
            bridges.append(dict(osm=f"w{w.id}", highway=w.tags["highway"], ref=w.tags.get("ref"), name=w.tags.get("name"),
                                bridge=w.tags.get("bridge"), length_m=round(osmium.geom.haversine_distance(w.nodes), 1),
                                node_ids=",".join(str(n.ref) for n in w.nodes), geometry=LineString(pts)))
ox.settings.useful_tags_way += ["bridge", "tunnel", "lanes", "maxspeed", "ref"]
G = ox.graph_from_xml(xml, simplify=False, retain_all=True)
G = ox.truncate.truncate_graph_bbox(G, (W, S_, E, N), truncate_by_edge=True)
ox.save_graphml(G, out / "eq_box_roads_sec_unsimplified.graphml")
gpd.GeoDataFrame(bridges, crs=4326).to_file(out / "eq_box_major_roads.gpkg", layer="bridges_sec", driver="GPKG")
print(f"graph {len(G.nodes)} nodes {len(G.edges)} edges; bridges (motorway..secondary): {len(bridges)}")
xy = np.array([[d["x"], d["y"]] for _, d in G.nodes(data=True)])
d = pd.read_csv(out / "destinations.csv")
far = []
for _, r in d.iterrows():
    k = np.hypot((xy[:, 0]-r.lon)*np.cos(np.radians(r.lat)), xy[:, 1]-r.lat).min()*111.2
    if k > 1.0: far.append((r.id, r["name"], round(k, 1)))
print("destinations >1 km from motorway..secondary:", far)
