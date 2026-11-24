"""Before/after figure: what the multi party marking adds to the co-word analysis."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, sys
out = sys.argv[1]
INK, INK2, MUTED, BLUE, GREY, SURF = "#0b0b0b", "#52514e", "#8a8984", "#2a78d6", "#c9c8c3", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.edgecolor": MUTED})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURF, gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.55})

rows = [("Multi rows left out (current default)", 17, GREY),
        ("Every multi row counted as buyer and seller", 138, GREY),
        ("Malak's pair marking", 62, BLUE),
        ("Malak's pair marking, links from different snippets only", 56, BLUE)]
y = range(len(rows))[::-1]
for yi, (lab, v, c) in zip(y, rows):
    a1.barh(yi, v, color=c, height=0.55)
    a1.text(v + 2, yi, str(v), va="center", color=INK)
a1.set_yticks(list(y)); a1.set_yticklabels([r[0] for r in rows])
a1.set_xlabel("buyer × brand seller pairs with association ratio ≥ 1.5", color=INK2)
a1.set_title("A. Strong cross-level links under each way of handling multi", loc="left", fontweight="bold", fontsize=11)

sides = ["buyer", "brand seller", "platform", "community", "market (unattributed)"]
before = [1334, 292, 95, 63, 0]; after = [1570, 493, 168, 115, 456]
import numpy as np
yy = np.arange(len(sides))[::-1]; h = 0.36
a2.barh(yy + h/2 + 0.02, before, height=h, color=GREY, label="multi rows left out")
a2.barh(yy - h/2 - 0.02, after, height=h, color=BLUE, label="with Malak's pair marking")
for yi, b, f in zip(yy, before, after):
    a2.text(b + 15, yi + h/2 + 0.02, str(b), va="center", fontsize=8.5, color=INK2)
    pct = f"  (+{(f-b)/b:.0%})" if b else "  (new)"
    a2.text(f + 15, yi - h/2 - 0.02, f"{f}{pct}", va="center", fontsize=8.5, color=INK)
a2.set_yticks(yy); a2.set_yticklabels(sides)
a2.set_xlabel("snippets that count toward each side", color=INK2)
a2.set_title("B. Evidence available on each side", loc="left", fontweight="bold", fontsize=11)
a2.legend(frameon=False, loc="lower right", fontsize=9)
for ax in (a1, a2):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis="x", color="#ecebe8", lw=0.8); ax.set_axisbelow(True)
a2.set_xlim(0, 1850); a1.set_xlim(0, 160)
fig.text(0.01, -0.06, "Leaving multi out hides most links; counting all of it as buyer and seller invents links. The marking sits between, and survives the stricter count almost unchanged (56 of 62).\n"
         "Stage 3 review sheet before Ahmed's adjudication; 833 multi rows. Exploratory and descriptive; no statistical test. "
         "Settings otherwise as the script defaults (engaged baseline, at least 5 papers per item and 5 shared papers per pair).",
         fontsize=7.5, color=MUTED)
fig.savefig(out, dpi=220, facecolor=SURF, bbox_inches="tight")
