"""Blind human check of the LLM judges on LoCoMo (paper: "What else moves the score").

    python3 human-check/human_check.py score human-check/sheet-nikhil.csv [human-check/sheet-amitoj.csv]

`make` rebuilt the sample from our run files and is kept for the record; `score` needs only the files
in this folder.

Sample: gpt-4o-mini answers with the short prompt (the headline setting), Becos and the whole
conversation. 120 questions where the three judges (gpt-4o-mini, gpt-4.1-mini, DeepSeek) do not all
agree, 60 per system, plus 30 where all three agree, 15 per system, as a check on unanimous verdicts.
The sheet shows only the question, the gold answer and the model's answer, in random order, so a
grader can see neither the system nor any judge's verdict. key.json maps item numbers back.

Graders fill the `grade` column with C (correct), W (wrong) or ? (cannot tell). Rubric in the sheet's
README.md: the answer is correct if it states the gold answer's key fact and contradicts nothing in it.
"""
from __future__ import annotations

import csv
import os
import json
import random
import sys
from pathlib import Path

RUN = Path(os.environ.get("LOCOMO_RUN", "runs/locomo"))  # only `make` reads it
OUT = Path(__file__).resolve().parent
ARMS = {"becos": "becos-qfc30rq", "whole": "full"}
JUDGES = ["gpt-4o-mini", "gpt-4.1-mini", "deepseek-v4-flash"]
N_DIS, N_AGREE, SEED = 60, 15, 20261112


def jsonl(p: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def questions() -> dict[str, dict]:
    data = json.loads((RUN / "data/locomo10.json").read_text())
    out = {}
    for conv in data:
        for i, qa in enumerate(conv["qa"]):
            if qa.get("category") == 5:
                continue
            out[f"{conv['sample_id']}-q{i:03d}"] = {"question": qa["question"], "gold": str(qa["answer"]),
                                                    "category": qa["category"]}
    return out


def make() -> None:
    qs, rng, items = questions(), random.Random(SEED), []
    for system, arm in ARMS.items():
        answers = jsonl(RUN / f"answers-{arm}.jsonl")
        verdicts = {j: jsonl(RUN / f"judge-{arm}-{j}-r1.jsonl") for j in JUDGES}
        ids = sorted(i for i in qs if i in answers and all(i in verdicts[j] for j in JUDGES))
        assert len(ids) == 1540, (system, len(ids))
        labels = {i: {j: verdicts[j][i]["label"] for j in JUDGES} for i in ids}
        dis = [i for i in ids if len(set(labels[i].values())) > 1]
        agree = [i for i in ids if len(set(labels[i].values())) == 1]
        print(f"{system}: {len(dis)} questions where the judges disagree, {len(agree)} unanimous")
        for group, pool, n in (("disagree", dis, N_DIS), ("agree", agree, N_AGREE)):
            for i in rng.sample(pool, n):
                items.append({"id": i, "system": system, "group": group, "judges": labels[i],
                              "answer": answers[i]["response"].strip(), **qs[i]})
    rng.shuffle(items)
    OUT.mkdir(exist_ok=True)
    with (OUT / "sheet.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "question", "gold answer", "model answer", "grade (C/W/?)", "note"])
        for n, it in enumerate(items, 1):
            w.writerow([n, it["question"], it["gold"], it["answer"], "", ""])
    key = {n: {k: it[k] for k in ("id", "system", "group", "category", "judges")} for n, it in enumerate(items, 1)}
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {OUT/'sheet.csv'} ({len(items)} items) and key.json")


def subset(n_total: int = 40) -> None:
    """A second grader's sheet: the same item numbers, stratified like the full sample (16+16 + 4+4)."""
    key = {int(k): v for k, v in json.loads((OUT / "key.json").read_text()).items()}
    rng, picked = random.Random(SEED + 1), []
    share = {"disagree": 16, "agree": 4}
    for system in ARMS:
        for group, k in share.items():
            pool = sorted(n for n, v in key.items() if v["system"] == system and v["group"] == group)
            picked += rng.sample(pool, k)
    assert len(picked) == n_total
    with (OUT / "sheet.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    keep = set(picked)
    with (OUT / "sheet-amitoj.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(rows[0])
        w.writerows(r for r in rows[1:] if int(r[0]) in keep)
    print(f"wrote {OUT/'sheet-amitoj.csv'} ({len(keep)} items, same item numbers as sheet.csv)")


def read_grades(p: Path) -> dict[int, str]:
    out = {}
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            g = (row.get("grade (C/W/?)") or "").strip().upper()[:1]
            if g in ("C", "W", "?"):
                out[int(row["item"])] = g
    return out


def score(paths: list[str]) -> None:
    key = {int(k): v for k, v in json.loads((OUT / "key.json").read_text()).items()}
    graders = [read_grades(Path(p)) for p in paths]
    for p, g in zip(paths, graders):
        print(f"{Path(p).name}: {len(g)} of {len(key)} graded")
    if len(graders) == 2:
        both = [n for n in key if graders[0].get(n) in ("C", "W") and graders[1].get(n) in ("C", "W")]
        a = sum(graders[0][n] == graders[1][n] for n in both)
        p1 = sum(graders[0][n] == "C" for n in both) / len(both)
        p2 = sum(graders[1][n] == "C" for n in both) / len(both)
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        po = a / len(both)
        print(f"graders agree on {a}/{len(both)} ({100*po:.1f}%), Cohen's kappa {(po-pe)/(1-pe):.2f}")
    # human verdict: from every grader who graded the item C or W, used only when they all agree
    # (the second sheet may cover a subset; items only the first grader saw use that grade)
    human = {}
    for n in key:
        gs = [g.get(n) for g in graders if g.get(n) in ("C", "W")]
        if gs and len(set(gs)) == 1 and n in graders[0]:
            human[n] = "CORRECT" if gs[0] == "C" else "WRONG"
    print(f"items with a usable human verdict: {len(human)}")
    for group in ("disagree", "agree"):
        ns = [n for n in human if key[n]["group"] == group]
        print(f"\n{group} ({len(ns)} items)")
        for j in JUDGES:
            m = sum(key[n]["judges"][j] == human[n] for n in ns)
            print(f"  {j:18s} matches the human on {m}/{len(ns)} ({100*m/max(len(ns),1):.0f}%)")
        for s in ARMS:
            ss = [n for n in ns if key[n]["system"] == s]
            c = sum(human[n] == "CORRECT" for n in ss)
            print(f"  {s:6s}: human says correct on {c}/{len(ss)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["make"]:
        make()
    elif sys.argv[1:2] == ["subset"]:
        subset()
    elif sys.argv[1:2] == ["score"] and len(sys.argv) in (3, 4):
        score(sys.argv[2:])
    else:
        sys.exit(__doc__)
