"""Draw the Kahramanmaraş network straight from network.json.

usage: python examples/kahramanmaras-2023/plot_network.py [network.json] [out.png]
"""
import json
import sys
from collections import Counter
from math import cos, radians
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

here = Path(__file__).parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "network.json"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "network.png"
net = json.loads(src.read_text(encoding="utf-8"))
nodes = {n["id"]: n for n in net["tables"]["Node"]}
edges = net["tables"]["Edge"]
bridges = net["tables"]["Bridge"]

sources = [n for n in nodes.values() if n["key_destination"]]
towns = [n for n in nodes.values() if n["id"].startswith("T")]
junctions = [n for n in nodes.values() if n["id"].startswith("J")]

fig, ax = plt.subplots(figsize=(14, 12))
for e in edges:                                   # along its course where it has one; ids at its middle
    line = e.get("geometry") or [[nodes[e["source"]]["lat"], nodes[e["source"]]["lon"]],
                                 [nodes[e["target"]]["lat"], nodes[e["target"]]["lon"]]]
    fixed = e.get("fixed_open", False)
    ax.plot([q[1] for q in line], [q[0] for q in line], color="#c4ccd6" if fixed else "#8a9bb3", lw=1.0 if fixed else 1.6,
            zorder=1, solid_capstyle="round", solid_joinstyle="round")
    mid = line[len(line) // 2] if len(line) > 2 else [(line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2]
    ax.annotate(e["id"], (mid[1], mid[0]), fontsize=6, color="#4a5a70" if not fixed else "#9aa5b3", style="italic",
                ha="center", va="center", zorder=2, bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.6))

counts = Counter(b["hazus_class"] for b in bridges)
classes = [c for c, _ in counts.most_common()]
palette = ["#4c78a8", "#e45756", "#54a24b", "#b279a2", "#f58518", "#72b7b2", "#9d755d", "#bab0ac"]
colour = {c: palette[i % len(palette)] for i, c in enumerate(classes)}
for c in classes:
    bs = [b for b in bridges if b["hazus_class"] == c]
    ax.scatter([b["lon"] for b in bs], [b["lat"] for b in bs], s=14 if c == classes[0] else 30,
               color=colour[c], edgecolor="white", linewidth=0.4, zorder=3)

ground = net["tables"].get("GroundFailure", [])
gstyle = {"landslide": ("^", "#8c564b"), "liquefaction": ("D", "#17becf"), "fault": ("_", "#000000")}
for kind, (marker, col) in gstyle.items():
    gs = [g for g in ground if g["kind"] == kind]
    if gs:
        ax.scatter([g["lon"] for g in gs], [g["lat"] for g in gs], marker=marker, s=46 if kind == "fault" else 26,
                   color=col, linewidths=2.2 if kind == "fault" else 0.4, edgecolor=None if kind == "fault" else "white",
                   zorder=3.5)
ax.scatter([n["lon"] for n in junctions], [n["lat"] for n in junctions], marker="s", s=18,
           facecolor="white", edgecolor="#333", linewidth=0.8, zorder=4)
ax.scatter([n["lon"] for n in towns], [n["lat"] for n in towns], s=60, color="#e6550d",
           edgecolor="black", linewidth=0.7, zorder=5)
ax.scatter([n["lon"] for n in sources], [n["lat"] for n in sources], marker="*", s=480, color="gold",
           edgecolor="black", linewidth=1.0, zorder=6)
for n in towns + sources:
    label = f"{n['id']} {n['name'].split(' (')[0]}"
    ax.annotate(label, (n["lon"], n["lat"]), xytext=(4, 4), textcoords="offset points", zorder=4.5,
                fontsize=9 if n in sources else 7, weight="bold" if n in sources else "normal",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

handles = [Line2D([], [], color="#8a9bb3", lw=1.6, label=f"edge that can close ({len(edges) - sum(bool(e.get("fixed_open")) for e in edges)})"),
           Line2D([], [], color="#c4ccd6", lw=1.0, label=f"edge that cannot close ({sum(bool(e.get("fixed_open")) for e in edges)})"),
           Line2D([], [], marker="*", ls="", color="gold", mec="black", ms=15, label=f"source / key destination ({len(sources)})"),
           Line2D([], [], marker="o", ls="", color="#e6550d", mec="black", ms=8, label=f"destination town ({len(towns)})"),
           Line2D([], [], marker="s", ls="", mfc="white", mec="#333", ms=5, label=f"junction ({len(junctions)})")]
handles += [Line2D([], [], marker="o", ls="", color=colour[c], mec="white", ms=6, label=f"bridge {c} ({counts[c]})")
            for c in classes]
handles += [Line2D([], [], marker=m, ls="", color=col, mec="white" if k != "fault" else col, ms=7, mew=2 if k == "fault" else 0.5,
                   label=f"{k}{' crossing' if k == 'fault' else ''} ({sum(g['kind'] == k for g in ground)})")
            for k, (m, col) in gstyle.items() if any(g["kind"] == k for g in ground)]
ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.95)
lat0 = sum(n["lat"] for n in nodes.values()) / len(nodes)
ax.set_aspect(1 / cos(radians(lat0)))
ax.set_xlabel("longitude (°E)")
ax.set_ylabel("latitude (°N)")
ax.set_title(f"{net['title']}: {len(nodes)} nodes, {len(edges)} edges, {len(bridges)} bridges, {len(ground)} ground-failure components")
ax.grid(alpha=0.25)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
