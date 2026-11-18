"""Step 1: destination towns = OSM cities/towns in the box with max MMI >= VII in either main shock.

usage: python destinations.py <turkey-latest.osm.pbf> <dir with us6000jllz.grid.xml, us6000jlqa.grid.xml> <work dir>
writes <work dir>/destinations.csv
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import osmium
import pandas as pd

pbf, usgs, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
W, S, E, N = 35.2, 35.8, 39.0, 38.8

rows = []
for o in osmium.FileProcessor(pbf, osmium.osm.NODE).with_filter(osmium.filter.KeyFilter("place")):
    if o.tags["place"] in ("city", "town") and S <= o.lat <= N and W <= o.lon <= E:
        rows.append(dict(name=o.tags.get("name"), place=o.tags["place"], lat=round(o.lat, 4), lon=round(o.lon, 4),
                         population=pd.to_numeric(o.tags.get("population"), errors="coerce"), osm=f"n{o.id}"))
d = pd.DataFrame(rows)

ns = {"s": "http://earthquake.usgs.gov/eqcenter/shakemap"}
mmi = np.zeros(len(d))
for f in ["us6000jllz.grid.xml", "us6000jlqa.grid.xml"]:
    r = ET.parse(usgs / f).getroot()
    flds = {e.attrib["name"]: int(e.attrib["index"]) - 1 for e in r.findall("s:grid_field", ns)}
    a = np.array(r.find("s:grid_data", ns).text.split(), float).reshape(-1, len(flds))
    gx, gy, gm = a[:, flds["LON"]], a[:, flds["LAT"]], a[:, flds["MMI"]]
    for i, t in d.iterrows():
        j = np.argmin((gx - t.lon) ** 2 + (gy - t.lat) ** 2)
        if abs(gx[j] - t.lon) < 0.02 and abs(gy[j] - t.lat) < 0.02:
            mmi[i] = max(mmi[i], gm[j])
d["max_mmi"] = mmi.round(1)
d = d[d.max_mmi >= 7].copy()

# one node per city: merge metropolitan districts into the city, keeping the strongest shaking
merge = {"Hatay": "Antakya", "Defne": "Antakya", "Onikişubat": "Kahramanmaraş", "Dulkadiroğlu": "Kahramanmaraş",
         "Şahinbey": "Gaziantep", "Şehitkamil": "Gaziantep", "Yeşilyurt": "Malatya", "Battalgazi": "Malatya"}
m = d.set_index("name")["max_mmi"]
for k, v in merge.items():
    if k in m and v in m:
        d.loc[d.name == v, "max_mmi"] = max(m[v], m[k])
d = d[~d.name.isin(merge) & (d.name != "حارم")].copy()       # drop the merged districts and the one Syrian town
d.loc[d.name == "Ceyhan", "population"] = 156081               # OSM tag reads "156.081" (thousands separator)
d = d.sort_values(["max_mmi", "name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
d["id"] = [f"T{i + 1:02d}" for i in range(len(d))]
d["population"] = d["population"].astype("Int64")
d[["id", "name", "lat", "lon", "population", "max_mmi", "osm"]].to_csv(out / "destinations.csv", index=False, encoding="utf-8")
print(len(d), "destinations")
