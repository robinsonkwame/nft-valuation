"""Build the board's import files (snippets.json and codebook.json) from the Stage 3 review sheet
and the multi party marking workbook, following the Snippet Clustering Board Specification,
section "Input data: the snippet file".

    python3 build_snippets_json.py --review "stage 3 review sheet.xlsx" \
        --multi-file "Stage 3 multi party marking - Malak.xlsx" \
        --codebook codebook_v1.md --mapping coword_mapping_template.csv --out board_input

What it does
  * Applies the review sheet's `choice` column: `human` keeps the coder's labels, `ai` keeps the
    AI's. Rows with no final codes (deleted highlights) are left out.
  * One record per kept snippet. snippet_id = paper | highlight_id.
  * side `brand_seller`, `buyer`, `platform`, `community` or `multi`. For multi rows, multi_pairs
    comes from the marking workbook: the pair columns marked `discussed`, or ["unattributed"].
  * Proposed codes that the mapping file sends to "HEDGE hedging" become "NEW: hedging" (the one
    committed new code); other proposed codes stay as "NEW: <label>".
  * cite is built from the paper folder name (e.g. anwar_and_raza_2024_... -> "Anwar and Raza 2024").
    It is a best guess; check it before the pilot.
  * Validates as the spec says and writes a readable report. The files are written only when there
    are no errors (use --force to write anyway). Warnings do not block.

Outputs in --out: snippets.json, codebook.json, build_report.txt.
"""
import argparse, collections, datetime, json, os, re, sys
import pandas as pd

SIDES = {"buyer", "brand_seller", "platform", "community", "multi"}
STANCES = {"documents", "proposes", "critiques", "measures"}
PAIRS = ["buyer_seller", "seller_platform", "buyer_platform", "buyer_community", "seller_community", "platform_community"]


def parse_codebook(path):
    text = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"^### ([A-G]\d)\. (.+?)\n(.*?)(?=^### |^## |\Z)", text, re.S | re.M):
        cid, name, body = m.group(1), m.group(2).strip(), m.group(3)
        d = re.search(r"\*\*Definition\.\*\*\s*(.+)", body)
        definition = ""
        if d:
            definition = re.split(r"(?<=\.)\s", d.group(1).strip(), maxsplit=1)[0]
        out.append({"id": cid, "name": name, "definition": definition})
    return out


def cite_from_paper(pid):
    """Best guess at a short citation from the paper folder name. Returns (cite, guessed_without_year)."""
    m = re.match(r"^(.*?)_?((?:19|20)\d\d)(?!\d)", pid)
    if m and m.group(1):
        authors, year = m.group(1), m.group(2)
    elif m:
        return f"Anon. {m.group(2)}", True
    else:
        authors, year = (pid.split("_et_al")[0] + "_et_al") if "_et_al" in pid else pid.split("_")[0], "n.d."
    etal = authors.endswith("_et_al")
    authors = authors[:-6] if etal else authors
    words = [w if w == "and" else w.capitalize() for w in authors.split("_") if w]
    return " ".join(words) + (" et al." if etal else "") + f" {year}", year == "n.d."


def read_marking(path):
    mk = pd.read_excel(path, "Multi_rows").fillna("")
    out = {}
    for _, r in mk.iterrows():
        pairs = [p for p in PAIRS if str(r.get(p, "")).strip().lower() == "discussed"]
        if not pairs and str(r.get("unattributed", "")).strip().upper() == "Y":
            pairs = ["unattributed"]
        out[f"{r['paper']}|{r['highlight_id']}"] = {"pairs": pairs, "other_pair": str(r.get("other_pair", "")).strip(),
                                                    "question": str(r.get("question", "")).strip().upper() == "Y"}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", required=True)
    ap.add_argument("--multi-file", required=True)
    ap.add_argument("--codebook", required=True, help="codebook_v1.md")
    ap.add_argument("--mapping", help="coword_mapping_template.csv, to fold hedging variants into 'NEW: hedging'")
    ap.add_argument("--dataset-version", default=None)
    ap.add_argument("--created-by", default="malak")
    ap.add_argument("--out", default="board_input")
    ap.add_argument("--force", action="store_true", help="write the files even when validation finds errors")
    a = ap.parse_args()

    codebook = parse_codebook(a.codebook)
    ids = {c["id"] for c in codebook}
    hedge = set()
    if a.mapping:
        m = pd.read_csv(a.mapping, dtype=str).fillna("")
        hedge = {r["code"].strip() for _, r in m.iterrows() if r["category"].strip().upper().startswith("HEDGE")}
    marks = read_marking(a.multi_file)

    df = pd.read_excel(a.review, "Review")
    use_ai = df["choice"].astype(str).str.strip().str.lower().eq("ai")
    pick = lambda h, ai_: df[ai_].where(use_ai, df[h])
    df["codes_f"], df["side_f"], df["stance_f"] = pick("human", "ai"), pick("human_stakeholder_side", "ai_stakeholder_side"), \
        pick("human_stance", "ai_stance")
    n_total = len(df)
    df = df[df["codes_f"].notna() & (df["codes_f"].astype(str).str.strip() != "")]
    source_version = os.path.splitext(os.path.basename(a.review))[0].replace(" ", "_")
    dataset_version = a.dataset_version or f"{source_version}_{datetime.date.today().isoformat()}"

    errors, warnings, records = [], collections.defaultdict(list), []
    for _, r in df.iterrows():
        sid = f"{r['paper']}|{r['highlight_id']}"
        codes = []
        for t in [x.strip() for x in str(r["codes_f"]).split(",") if x.strip() and x.strip() != "(no codes)"]:
            if t in hedge:
                t = "NEW: hedging"
            if t not in codes:
                codes.append(t)
        side = str(r["side_f"]).strip()
        if cite_from_paper(str(r["paper"]))[1]:
            warnings["cite guessed without a year (fix by hand; folder name has no year or is not author_year)"].append(str(r["paper"]))
        rec = {"snippet_id": sid, "paper_id": str(r["paper"]), "highlight_id": str(r["highlight_id"]),
               "cite": cite_from_paper(str(r["paper"]))[0], "text": str(r["Snippet"]).strip(), "codes": codes,
               "side": side, "multi_pairs": [], "stance": str(r["stance_f"]).strip(),
               "reviewer": str(r["coder"]).strip(), "source_version": source_version}
        if side == "multi":
            mk = marks.get(sid)
            if mk is None:
                errors.append(f"{sid}: multi row not found in the marking workbook")
            else:
                rec["multi_pairs"] = mk["pairs"]
                if not mk["pairs"]:
                    warnings["multi row with no pair (only one party, or only an outside party)"].append(sid)
                if mk["question"]:
                    warnings["multi row still flagged as a question for Kwame"].append(sid)
        records.append(rec)

    # validation, as in the spec
    seen = collections.Counter(r["snippet_id"] for r in records)
    errors += [f"{k}: snippet_id repeats {v} times" for k, v in seen.items() if v > 1]
    for r in records:
        for f in ("snippet_id", "paper_id", "highlight_id", "cite", "text", "side", "stance", "reviewer", "source_version"):
            if not r[f] or r[f] == "nan":
                errors.append(f"{r['snippet_id']}: required field '{f}' is empty")
        if not r["codes"]:
            errors.append(f"{r['snippet_id']}: no codes")
        if r["side"] not in SIDES:
            errors.append(f"{r['snippet_id']}: side '{r['side']}' is not one of {sorted(SIDES)}")
        if r["stance"] not in STANCES:
            errors.append(f"{r['snippet_id']}: stance '{r['stance']}' is not one of {sorted(STANCES)}")
        if not 1 <= len(r["text"]) <= 2000:
            errors.append(f"{r['snippet_id']}: text is {len(r['text'])} characters (allowed 1 to 2000)")
        for c in r["codes"]:
            if c not in ids and not c.startswith("NEW: "):
                errors.append(f"{r['snippet_id']}: code '{c}' is neither a codebook ID nor a NEW: label")

    os.makedirs(a.out, exist_ok=True)
    sides = collections.Counter(r["side"] for r in records)
    n_cards = sum(len(r["codes"]) for r in records)
    cells = collections.Counter((c, r["side"]) for r in records for c in r["codes"])
    new_codes = collections.Counter(c for r in records for c in r["codes"] if c.startswith("NEW: "))
    lines = [f"build_snippets_json report, {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}",
             f"review sheet: {a.review}", f"marking workbook: {a.multi_file}", "",
             f"snippets: {len(records)}  (review sheet rows left out because they had no final codes, i.e. deleted: {n_total - len(records)})",
             f"cards (one per snippet per code): {n_cards}",
             f"snippets with more than one code: {sum(len(r['codes']) > 1 for r in records)}",
             f"non-empty cells (code x side): {len(cells)}; cells with fewer than five cards: {sum(v < 5 for v in cells.values())}",
             "sides: " + ", ".join(f"{k} {v}" for k, v in sides.most_common()),
             "largest cells: " + ", ".join(f"{s} {c} ({v})" for (c, s), v in cells.most_common(6)),
             f"proposed codes: 'NEW: hedging' on {new_codes.get('NEW: hedging', 0)} snippets; "
             f"{len(new_codes) - ('NEW: hedging' in new_codes)} other NEW labels on "
             f"{sum(v for k, v in new_codes.items() if k != 'NEW: hedging')} snippets (these go to the 'Other new codes' column)",
             "", f"ERRORS ({len(errors)}): the board would reject the file"] + [f"  {e}" for e in errors[:200]]
    lines += ["", "WARNINGS (do not block import; decide before the pilot)"]
    for k, v in warnings.items():
        v = list(dict.fromkeys(v))
        lines.append(f"  {k}: {len(v)}")
        lines += [f"    {x}" for x in v[:15]] + (["    ..."] if len(v) > 15 else [])
    report = "\n".join(lines)
    with open(os.path.join(a.out, "build_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    if errors and not a.force:
        sys.exit(f"\nNot written: {len(errors)} errors. Fix them or use --force.")
    header = {"dataset_version": dataset_version,
              "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "created_by": a.created_by}
    with open(os.path.join(a.out, "snippets.json"), "w", encoding="utf-8") as f:
        json.dump({"header": header, "snippets": records}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "codebook.json"), "w", encoding="utf-8") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=1)
    print(f"\nWrote {len(records)} snippets to {a.out}/snippets.json and {len(codebook)} codes to {a.out}/codebook.json")


if __name__ == "__main__":
    main()
