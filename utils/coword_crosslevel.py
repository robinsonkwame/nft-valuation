"""Cross-level co-word analysis of the Stage 3 coding (descriptive).

Builds a paper-by-item matrix where an item is a stakeholder side plus a code (for example
'buyer:G1' or 'seller:G3'), or, with --assignments, a side plus a board category. For every
left-side item x right-side item pair it reports the papers coding each item, the papers coding
both (co-occurrence), the co-occurrence expected from the two items' frequencies alone, and the
association ratio (observed / expected). A ratio of 1 means the pair shares as many papers as
their frequencies predict; 2 means twice as many. The ratio is proportional to van Eck and
Waltman's (2009) association strength. With --baseline engaged (default) the expectation is
computed only among papers that code both sides at all.

No statistical test is applied by default: the figure is descriptive. --test adds a curveball
permutation screen (p and Benjamini-Hochberg q columns) to the CSV only. The test always uses
paper-level co-occurrence.

Changes from the first version (Nov 2026)
  --cooc-scope cross-excerpt  A multi snippet split into two sides (for example buyer_seller)
      puts all of its codes on both sides, so one sentence coded A2 and A5 creates a buyer A2 x
      seller A5 link by itself. cross-excerpt counts a paper for a pair only when the two items
      are supported by at least two different snippets in that paper, which removes those
      one-sentence links. The expected count is not adjusted, so ratios in this mode are
      conservative (slightly low). Both counts are always written to the CSV.
  Unattributed multi rows     In --multi-mode pairs, rows marked unattributed now go to a
      'market' side (usable as --left market or --right market) instead of vanishing, and the
      script prints how many multi rows went where.
  Counts in the matrix        Labelled cells show the ratio and, under it, the shared papers.
  --all-pairs                 Runs every pair of sides and writes summary_all_pairs.csv.
  --assignments               Reads the board's assignments.csv export and uses side plus
      category (or shared label, --item-level label) as items, as the board spec asks.
  run_settings.json           Every output folder records the settings and input files used.

Outputs (in --out): census_papers_by_code_and_side.csv, coword_cells.csv (with paper lists),
coword_figure.png, coword_figure.pdf, run_settings.json.

Examples
  python3 coword_crosslevel.py --review "stage 3 review sheet.xlsx"
  python3 coword_crosslevel.py --review "stage 3 review sheet.xlsx" --mapping coword_mapping_template.csv \\
      --multi-mode pairs --multi-file "Stage 3 multi party marking - Malak.xlsx" --cooc-scope cross-excerpt
  python3 coword_crosslevel.py --review "stage 3 review sheet.xlsx" --multi-mode pairs \\
      --multi-file "Stage 3 multi party marking - Malak.xlsx" --all-pairs --out out_all
  python3 coword_crosslevel.py --assignments v1-pilot_2026-11-24_assignments.csv --item-level category

Mapping file (optional): CSV with columns 'code' and 'category'. Every code token is replaced
by its category before analysis. Leave category blank to drop a code. Unmapped codebook codes are
kept as they are; unmapped proposed new codes ('NEW: ...') are dropped unless --keep-new is given.
"""
import argparse, datetime, itertools, json, os, re, sys
import numpy as np, pandas as pd

NAMES = {"A1": "efficiency", "A2": "excellence", "A3": "status", "A4": "esteem", "A5": "play", "A6": "aesthetics",
         "B1": "functional value", "B2": "social value", "B3": "emotional value", "B4": "epistemic value",
         "B5": "conditional value", "C1": "extended self", "C2": "identity expression", "D1": "co-creation",
         "D2": "service ecosystem", "E1": "community membership", "E2": "shared rituals", "F1": "status signaling",
         "G1": "scarcity perception", "G2": "provenance", "G3": "royalty", "G4": "utility", "G5": "speculation"}
SIDES = {"buyer": "buyer", "brand_seller": "seller", "seller": "seller", "platform": "platform", "community": "community"}
PAIR_COLS = {"buyer_seller": ("buyer", "seller"), "seller_platform": ("seller", "platform"),
             "buyer_platform": ("buyer", "platform"), "buyer_community": ("buyer", "community"),
             "seller_community": ("seller", "community"), "platform_community": ("platform", "community")}
SIDE_TITLE = {"buyer": "BUYER", "seller": "BRAND-SELLER", "platform": "PLATFORM", "community": "COMMUNITY",
              "market": "MARKET (UNATTRIBUTED)"}
ALL_SIDES = ["buyer", "seller", "platform", "community", "market"]


class NoItems(Exception):
    pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--review", help="stage 3 review sheet (.xlsx, sheet 'Review')")
    src.add_argument("--assignments", help="the board's assignments.csv export (category-level items)")
    ap.add_argument("--item-level", choices=["category", "label"], default="category",
                    help="with --assignments: items are side + category name (category) or side + shared label (label)")
    ap.add_argument("--multi-mode", choices=["both", "exclude", "pairs"], default="exclude",
                    help="multi rows: drop them (exclude, default), count as buyer and seller (both), or use the pair marking (pairs)")
    ap.add_argument("--multi-file", help="the marked multi party workbook (needed for --multi-mode pairs)")
    ap.add_argument("--mapping", help="CSV with columns code,category")
    ap.add_argument("--keep-new", action="store_true", help="keep unmapped proposed new codes as their own items")
    ap.add_argument("--left", default="buyer", choices=ALL_SIDES)
    ap.add_argument("--right", default="seller", choices=ALL_SIDES)
    ap.add_argument("--all-pairs", action="store_true", help="run every pair of sides into subfolders of --out")
    ap.add_argument("--cooc-scope", choices=["paper", "cross-excerpt"], default="paper",
                    help="paper (default): any paper with both items counts. cross-excerpt: the two items must come "
                         "from at least two different snippets in the paper (removes links made by one multi snippet)")
    ap.add_argument("--min-papers", type=int, default=5, help="an item needs at least this many papers to be shown")
    ap.add_argument("--min-cooc", type=int, default=5, help="a pair needs at least this many shared papers to be shown")
    ap.add_argument("--baseline", choices=["engaged", "all"], default="engaged",
                    help="expected co-occurrence computed among papers that code both sides at all (engaged, default) "
                         "or among all papers (all)")
    ap.add_argument("--min-ratio", type=float, default=1.5, help="pairs at or above this ratio get an edge and a label")
    ap.add_argument("--include-same-code", action="store_true",
                    help="also show pairs with the same code on both sides (off by default; multi rows create them mechanically)")
    ap.add_argument("--test", action="store_true", help="add a curveball permutation screen to the CSV (not the figure)")
    ap.add_argument("--perms", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out", default="coword_out")
    a = ap.parse_args(argv)
    if a.left == a.right and not a.all_pairs:
        ap.error("--left and --right must differ")
    if a.multi_mode == "pairs" and a.review and not a.multi_file:
        ap.error("--multi-mode pairs needs --multi-file")
    return a


def load_mapping(path):
    if not path:
        return {}
    m = pd.read_csv(path, dtype=str).fillna("")
    return {r["code"].strip(): r["category"].strip() for _, r in m.iterrows()}


def tokens(cell):
    if pd.isna(cell):
        return []
    return [t.strip() for t in str(cell).split(",") if t.strip() and t.strip() != "(no codes)"]


def map_token(t, mapping, keep_new):
    if t in mapping:
        return mapping[t] or None
    if t.startswith("NEW:"):
        return t if keep_new else None
    return t


def label(item_code):
    return f"{item_code} {NAMES[item_code]}" if item_code in NAMES else item_code


def yes(v):
    return str(v).strip().lower() in ("y", "yes", "true", "1", "discussed")


def read_marking(path):
    """multi party workbook -> {paper|highlight_id: (sides, kind)}; kind = pairs, unattributed, single, blank."""
    mk = pd.read_excel(path, "Multi_rows").fillna("")
    out = {}
    for _, r in mk.iterrows():
        ps = [PAIR_COLS[c] for c in PAIR_COLS if str(r.get(c, "")).strip().lower() == "discussed"]
        if ps:
            sides, kind = sorted({s for p in ps for s in p}), "pairs"
        elif yes(r.get("unattributed", "")):
            sides, kind = ["market"], "unattributed"
        elif str(r.get("other_pair", "")).strip():
            sides, kind = [], "other pair only"
        else:
            sides, kind = [], "no pair (one party)"
        out[f"{r['paper']}|{r['highlight_id']}"] = (sides, kind)
    return out


def load_rows(a, mapping, diag):
    df = pd.read_excel(a.review, "Review")
    use_ai = df["choice"].astype(str).str.strip().str.lower().eq("ai")
    df["codes_final"] = np.where(use_ai, df["ai"], df["human"])
    df["side_final"] = np.where(use_ai, df["ai_stakeholder_side"], df["human_stakeholder_side"])
    df = df[df["codes_final"].notna()].copy()
    df["key"] = df["paper"].astype(str) + "|" + df["highlight_id"].astype(str)
    marks = read_marking(a.multi_file) if a.multi_mode == "pairs" else {}

    recs, multi = [], {}
    for _, r in df.iterrows():
        side = str(r["side_final"]).strip()
        if side == "multi":
            if a.multi_mode == "exclude":
                multi["excluded"] = multi.get("excluded", 0) + 1
                continue
            if a.multi_mode == "both":
                sides, kind = ["buyer", "seller"], "both"
            else:
                sides, kind = marks.get(r["key"], ([], "not in marking file"))
            multi[kind] = multi.get(kind, 0) + 1
            if not sides:
                continue
        elif side in SIDES:
            sides = [SIDES[side]]
        else:
            continue
        for t in tokens(r["codes_final"]):
            c = map_token(t, mapping, a.keep_new)
            if c:
                for s in sides:
                    recs.append((r["paper"], f"{s}:{c}", r["key"]))
    diag["multi_rows"] = multi
    return pd.DataFrame(recs, columns=["paper", "item", "key"]).drop_duplicates()


def parse_pairs(v):
    v = "" if pd.isna(v) else str(v).strip()
    if not v:
        return []
    if v.startswith("["):
        try:
            return [str(x) for x in json.loads(v)]
        except ValueError:
            pass
    return [x.strip() for x in re.split(r"[;,|]", v) if x.strip()]


def load_assignments(a, diag):
    df = pd.read_csv(a.assignments, dtype=str).fillna("")
    need = {"snippet_id", "paper_id", "side"} | ({"category_name", "category_id"} if a.item_level == "category" else {"shared_label"})
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"assignments file is missing columns: {', '.join(miss)}")
    recs, skipped, multi = [], 0, {}
    for _, r in df.iterrows():
        if a.item_level == "category":
            cat = r["category_name"].strip() or r["category_id"].strip()
        else:
            cat = r["shared_label"].strip()
        if not cat:
            skipped += 1
            continue
        side = r["side"].strip()
        if side == "multi":
            if a.multi_mode == "exclude":
                multi["multi cards excluded"] = multi.get("multi cards excluded", 0) + 1
                continue
            ps = parse_pairs(r.get("multi_pairs", ""))
            if a.multi_mode == "both":
                sides = ["buyer", "seller"]
            elif ps == ["unattributed"]:
                sides = ["market"]
            else:
                sides = sorted({s for p in ps if p in PAIR_COLS for s in PAIR_COLS[p]})
            kind = "multi cards placed" if sides else "multi cards with no pair"
            multi[kind] = multi.get(kind, 0) + 1
        else:
            sides = [SIDES[side]] if side in SIDES else []
        for s in sides:
            recs.append((r["paper_id"], f"{s}:{cat}", r["snippet_id"]))
    diag["multi_rows"] = multi
    diag["cards_without_category_skipped"] = skipped
    return pd.DataFrame(recs, columns=["paper", "item", "key"]).drop_duplicates()


def curveball(rows, rng, swaps):
    n = len(rows)
    for _ in range(swaps):
        i, j = rng.choice(n, 2, replace=False)
        A, B = rows[i], rows[j]
        ua, ub = list(A - B), list(B - A)
        if not ua or not ub:
            continue
        pool = ua + ub
        rng.shuffle(pool)
        k = len(ua)
        common = A & B
        rows[i] = common | set(pool[:k])
        rows[j] = common | set(pool[k:])


def bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    m = len(pv)
    adj = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    res = np.empty(m)
    res[o] = np.minimum(adj, 1)
    q[ok] = res
    return q


def build(rec, a, left, right):
    papers = sorted(rec["paper"].unique())
    items = sorted(rec["item"].unique())
    pi = {p: i for i, p in enumerate(papers)}
    ii = {t: j for j, t in enumerate(items)}
    M = np.zeros((len(papers), len(items)), dtype=np.int8)
    for p, t in rec[["paper", "item"]].itertuples(index=False):
        M[pi[p], ii[t]] = 1
    N = len(papers)
    n = M.sum(0)
    L = [t for t in items if t.startswith(left + ":") and n[ii[t]] >= a.min_papers]
    R = [t for t in items if t.startswith(right + ":") and n[ii[t]] >= a.min_papers]
    if not L or not R:
        raise NoItems(f"no {left} or {right} items with at least {a.min_papers} papers; lower --min-papers")
    li, ri = [ii[t] for t in L], [ii[t] for t in R]
    C_paper = M[:, li].T.astype(np.int32) @ M[:, ri].astype(np.int32)

    support = rec.groupby(["paper", "item"])["key"].apply(set).to_dict()
    C_cross = np.zeros_like(C_paper)
    both_lists, within = {}, {}
    for x, lt in enumerate(L):
        for y, rt in enumerate(R):
            both = [papers[i] for i in np.flatnonzero(M[:, ii[lt]] & M[:, ii[rt]])]
            # a paper counts across excerpts when the two items rest on at least two different snippets
            cross = [q for q in both if len(support[(q, lt)] | support[(q, rt)]) >= 2]
            C_cross[x, y] = len(cross)
            within[(x, y)] = sum(1 for q in both if support[(q, lt)] & support[(q, rt)])
            both_lists[(x, y)] = (both, cross)
    C = C_cross if a.cooc_scope == "cross-excerpt" else C_paper

    side_l = np.array([t.startswith(left + ":") for t in items])
    side_r = np.array([t.startswith(right + ":") for t in items])
    if a.baseline == "engaged":
        eng = (M[:, side_l].sum(1) > 0) & (M[:, side_r].sum(1) > 0)
        Ne = int(eng.sum())
        E = np.outer(M[eng][:, li].sum(0), M[eng][:, ri].sum(0)) / Ne if Ne else np.zeros(C.shape)
    else:
        Ne = N
        E = np.outer(n[li], n[ri]) / N
    same = np.array([[lt.split(":", 1)[1] == rt.split(":", 1)[1] for rt in R] for lt in L])
    shown = (C >= a.min_cooc) & (~same | a.include_same_code)

    pcols = {}
    if a.test:
        rng = np.random.default_rng(a.seed)
        rows = [set(np.flatnonzero(r)) for r in M]
        curveball(rows, rng, N * 10)
        ge = np.zeros(C.shape)
        for _ in range(a.perms):
            curveball(rows, rng, N * 3)
            X = np.zeros_like(M)
            for r_, s_ in enumerate(rows):
                X[r_, list(s_)] = 1
            ge += (X[:, li].T.astype(np.int32) @ X[:, ri].astype(np.int32)) >= C_paper
        p = (ge + 1) / (a.perms + 1)
        pcols = {"p_curveball_paper_level": p,
                 "q_curveball_paper_level": bh(np.where(shown, p, np.nan).ravel()).reshape(C.shape)}

    out = []
    for x, lt in enumerate(L):
        for y, rt in enumerate(R):
            both, cross = both_lists[(x, y)]
            e = E[x, y]
            row = dict(left=lt, right=rt, n_left=int(n[ii[lt]]), n_right=int(n[ii[rt]]),
                       cooc=int(C[x, y]), cooc_paper=int(C_paper[x, y]), cooc_cross_excerpt=int(C_cross[x, y]),
                       expected=round(e, 2), assoc_ratio=round(C[x, y] / e, 3) if e else np.nan,
                       same_code=bool(same[x, y]), shown=bool(shown[x, y]),
                       papers_within_one_excerpt=within[(x, y)], papers="; ".join(both),
                       papers_cross_excerpt="; ".join(cross))
            for k, v in pcols.items():
                row[k] = v[x, y]
            out.append(row)
    cells = pd.DataFrame(out)
    census = pd.Series(n, index=items).rename("papers").reset_index().rename(columns={"index": "item"})
    census[["side", "code"]] = census["item"].str.split(":", n=1, expand=True)
    census = census.pivot_table(index="code", columns="side", values="papers", fill_value=0).astype(int)
    return cells, census, L, R, n, ii, N, Ne


def draw(cells, L, R, n, ii, N, Ne, a, left, right, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    from matplotlib.lines import Line2D

    INK, INK2, MUTED, BLUE, SURF = "#0b0b0b", "#52514e", "#8a8984", "#2a78d6", "#ffffff"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})
    nl, nr = len(L), len(R)
    h = max(7.5, 0.36 * max(nl, nr) + 3)
    fig = plt.figure(figsize=(17, h), facecolor=SURF)
    axA = fig.add_axes([0.14, 0.17, 0.27, 0.71])
    axB = fig.add_axes([0.70, 0.27, 0.25, 0.61])
    lab = lambda t: label(t.split(":", 1)[1])
    cv = cells.set_index(["left", "right"])

    yl = np.linspace(1, 0, nl) if nl > 1 else np.array([0.5])
    yr = np.linspace(1, 0, nr) if nr > 1 else np.array([0.5])
    axA.set_xlim(-0.1, 1.1); axA.set_ylim(-0.06, 1.08); axA.axis("off")
    for x, lt in enumerate(L):
        for y, rt in enumerate(R):
            c = cv.loc[(lt, rt)]
            if c.shown and c.assoc_ratio >= a.min_ratio:
                axA.plot([0, 1], [yl[x], yr[y]], color=BLUE, lw=min(0.5 + 1.5 * (c.assoc_ratio - 1), 6), alpha=.8,
                         solid_capstyle="round", zorder=2)
    for x, lt in enumerate(L):
        axA.scatter(0, yl[x], s=30 + n[ii[lt]] * 1.5, color=SURF, edgecolor=INK, lw=1.2, zorder=3)
        axA.text(-0.06, yl[x], f"{lab(lt)}  ({n[ii[lt]]})", ha="right", va="center")
    for y, rt in enumerate(R):
        axA.scatter(1, yr[y], s=30 + n[ii[rt]] * 1.5, color=INK, edgecolor=INK, lw=1.2, zorder=3)
        axA.text(1.06, yr[y], f"({n[ii[rt]]})  {lab(rt)}", ha="left", va="center")
    axA.text(0, 1.07, f"{SIDE_TITLE[left]}-SIDE ITEMS", ha="center", fontweight="bold", color=INK2)
    axA.text(1, 1.07, f"{SIDE_TITLE[right]}-SIDE ITEMS", ha="center", fontweight="bold", color=INK2)
    axA.legend(handles=[Line2D([], [], color=BLUE, lw=1.3, label=f"Association ratio {a.min_ratio:g}"),
                        Line2D([], [], color=BLUE, lw=4.3, label=f"Association ratio {a.min_ratio + 2:g}"),
                        Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK, label="Node size = papers coded (n)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8.5)

    piv = lambda col: cells.pivot(index="left", columns="right", values=col).reindex(index=L, columns=R).values
    Rm, SH, SC, CO = piv("assoc_ratio").astype(float), piv("shown").astype(bool), piv("same_code").astype(bool), piv("cooc")
    Lg = np.where(SH, np.log2(np.clip(Rm, 0.25, 4)), np.nan)
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f2b6b4", "#f0efec", "#9ec5f4", "#2a78d6", "#184f95"])
    cmap.set_bad("#f7f7f5")
    im = axB.imshow(np.ma.masked_invalid(Lg), cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2), aspect="auto")
    axB.set_xticks(range(nr))
    axB.set_xticklabels([lab(t) for t in R], rotation=40 if nr <= 12 else 60, ha="right",
                        fontsize=9 if nr <= 12 else 7.5, rotation_mode="anchor")
    axB.set_yticks(range(nl)); axB.set_yticklabels([lab(t) for t in L], fontsize=9 if nl <= 12 else 8)
    axB.set_xticks(np.arange(-.5, nr), minor=True); axB.set_yticks(np.arange(-.5, nl), minor=True)
    axB.grid(which="minor", color=SURF, lw=2); axB.tick_params(which="both", length=0)
    for s_ in axB.spines.values():
        s_.set_visible(False)
    big = nr <= 12
    for i in range(nl):
        for j in range(nr):
            if SC[i, j] and not SH[i, j]:
                axB.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", ec=MUTED, lw=0))
            if SH[i, j] and Rm[i, j] >= a.min_ratio:
                col = "white" if Lg[i, j] > 0.9 else INK
                axB.text(j, i - 0.13, f"{Rm[i, j]:.1f}", ha="center", va="center", fontsize=7.5 if big else 6,
                         fontweight="bold", color=col)
                axB.text(j, i + 0.24, f"{int(CO[i, j])}", ha="center", va="center", fontsize=6 if big else 4.8, color=col)
    axB.set_xlabel(f"{SIDE_TITLE[right].lower()}-side item (columns); {SIDE_TITLE[left].lower()}-side item (rows)",
                   color=INK2, labelpad=8)
    cb = fig.colorbar(im, ax=axB, fraction=0.04, pad=0.02, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["¼×", "½×", "1×", "2×", "4×"]); cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=INK2)

    fig.text(0.02, 0.95, "A. Cross-level co-word map", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.915, f"Edges: pairs with association ratio of at least {a.min_ratio:g}; width = ratio", color=INK2)
    fig.text(0.56, 0.95, f"B. {SIDE_TITLE[left].title()} × {SIDE_TITLE[right].lower()} association matrix",
             fontsize=12, fontweight="bold")
    fig.text(0.56, 0.915, f"Cell colour = association ratio; labelled cells (at least {a.min_ratio:g}) show ratio over shared papers",
             color=INK2)
    scope = ("Co-occurrence counts a paper only when the two items come from different snippets (cross-excerpt); "
             "expected is not adjusted, so ratios are conservative. " if a.cooc_scope == "cross-excerpt" else "")
    fig.text(0.02, 0.01, f"{N} papers. Unit of analysis = paper. Association ratio = observed ÷ expected paper-level "
             f"co-occurrence. " + (f"Expected is computed among the {Ne} papers that code both sides: (papers with left item × "
             f"papers with right item) ÷ {Ne}." if a.baseline == "engaged" else f"Expected = (papers with left item × "
             f"papers with right item) ÷ {N}.") + " Descriptive; no statistical test.\n" + scope +
             f"Multi rows: {a.multi_mode}. Items shown: at least {a.min_papers} papers. Pairs shown: at "
             f"least {a.min_cooc} shared papers (blank cells have fewer)."
             + ("" if a.include_same_code else " Hatched cells: same item on both sides, not shown."),
             fontsize=8, color=MUTED)
    fig.savefig(path + ".png", dpi=300, facecolor=SURF)
    fig.savefig(path + ".pdf", facecolor=SURF)
    plt.close(fig)


def run_one(rec, a, left, right, out, diag):
    os.makedirs(out, exist_ok=True)
    cells, census, L, R, n, ii, N, Ne = build(rec, a, left, right)
    census.to_csv(os.path.join(out, "census_papers_by_code_and_side.csv"))
    cells.to_csv(os.path.join(out, "coword_cells.csv"), index=False)
    draw(cells, L, R, n, ii, N, Ne, a, left, right, os.path.join(out, "coword_figure"))
    strong = cells[cells.shown & (cells.assoc_ratio >= a.min_ratio)].sort_values("assoc_ratio", ascending=False)
    settings = {k: v for k, v in vars(a).items()}
    settings.update(left=left, right=right, run_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    papers=int(N), engaged_papers=int(Ne), diagnostics=diag,
                    pairs_shown=int(cells.shown.sum()), pairs_at_or_above_min_ratio=int(len(strong)))
    with open(os.path.join(out, "run_settings.json"), "w") as f:
        json.dump(settings, f, indent=2, default=str)
    print(f"{left} x {right}: {int(cells.shown.sum())} pairs shown, {len(strong)} at ratio >= {a.min_ratio:g}. Outputs in {out}/")
    return cells, strong


def main(argv=None):
    a = parse_args(argv)
    diag = {}
    rec = load_assignments(a, diag) if a.assignments else load_rows(a, load_mapping(a.mapping), diag)
    print(f"{rec['paper'].nunique()} papers, {rec['item'].nunique()} side:item items")
    if diag.get("multi_rows"):
        print("multi rows: " + ", ".join(f"{k} {v}" for k, v in sorted(diag["multi_rows"].items())))
    if diag.get("cards_without_category_skipped"):
        print(f"cards without a category skipped: {diag['cards_without_category_skipped']}")
    if not a.all_pairs:
        try:
            run_one(rec, a, a.left, a.right, a.out, diag)
        except NoItems as e:
            sys.exit(str(e))
        return
    present = [s for s in ALL_SIDES if rec["item"].str.startswith(s + ":").any()]
    summary = []
    for left, right in itertools.combinations(present, 2):
        sub = os.path.join(a.out, f"{left}_x_{right}")
        try:
            cells, strong = run_one(rec, a, left, right, sub, diag)
        except NoItems as e:
            print(f"{left} x {right}: skipped ({e})")
            summary.append(dict(left=left, right=right, status="skipped", note=str(e)))
            continue
        top = "; ".join(f"{r.left.split(':', 1)[1]} x {r.right.split(':', 1)[1]} ({r.assoc_ratio:.1f}, {r.cooc} papers)"
                        for r in strong.head(5).itertuples())
        summary.append(dict(left=left, right=right, status="ok", left_items=cells.left.nunique(),
                            right_items=cells.right.nunique(), pairs_shown=int(cells.shown.sum()),
                            strong_pairs=len(strong), top_pairs=top))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(a.out, "summary_all_pairs.csv"), index=False)
    print(f"summary written to {a.out}/summary_all_pairs.csv")


if __name__ == "__main__":
    main()
