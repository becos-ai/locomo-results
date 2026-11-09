#!/usr/bin/env python3
"""LoCoMo answer / judge / score, as used in "What Moves a LoCoMo Score?" (becos, 2026).

Standalone: Python 3.10+, standard library only. Needs LoCoMo's locomo10.json
(https://github.com/snap-research/locomo, CC BY-NC 4.0) and an OpenAI API key in OPENAI_API_KEY.

    # answer with the whole conversation, the no-memory floor, or your own system's context
    python3 locomo_eval.py answer --locomo locomo10.json --context whole --model gpt-4o-mini --out answers.jsonl
    python3 locomo_eval.py answer --locomo locomo10.json --context none  --model gpt-4o-mini --out answers.jsonl
    python3 locomo_eval.py answer --locomo locomo10.json --context file:my_contexts.jsonl --out answers.jsonl
    #   my_contexts.jsonl: one {"id": "conv-26-q000", "context": "..."} per question
    #   --prompt short (default, Nemori's) or 7step (ours)

    # judge answers (ours from data/, or your own)
    python3 locomo_eval.py judge --locomo locomo10.json --answers answers.jsonl --judge gpt-4o-mini --out verdicts.jsonl

    # score verdicts: overall and per category
    python3 locomo_eval.py score --locomo locomo10.json --verdicts verdicts.jsonl

Questions: categories 1-4 of LoCoMo (1,540); id = "<sample_id>-q<index in the sample's qa list, 3 digits>".
Category ids: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop. Every step resumes: rows already in
--out are skipped. Answer and judge calls use temperature 0.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

# Nemori's answer prompt, verbatim (MIT licence; github.com/nemori-ai/nemori @ d2a6dff,
# evaluation/locomo/search.py). {episodic} gets the context; {semantic} is left empty.
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Episodic Memories:
    {episodic}

    Semantic Memories:
    {semantic}

    Question: {question}

    Answer:
    """

# Our 7-step prompt: Nemori's prompt with instruction 8 replaced. Only the text after "FINAL ANSWER:"
# is saved as the answer and judged.
ANSWER_PROMPT_7STEP = ANSWER_PROMPT.replace(
    "8. The answer should be less than 5-6 words.",
    """8. Before answering, reason through these steps and write them out:
       Step 1 - Relevant memories: list every memory that bears on the question, with its timestamp.
       Step 2 - Key details: pull out the names, places, numbers, dates and frequencies they give.
       Step 3 - Connections: link the same people, things and events across memories; note what
                follows from them together.
       Step 4 - Time: turn every relative time reference into a date, month or year using the
                memory's own timestamp.
       Step 5 - Conflicts: if memories disagree, say which one wins and why (most recent first).
       Step 6 - Checklist: confirm the answer keeps every specific name, number and detail asked for.
       Step 7 - Answer: say how you will phrase the answer.
       Then write "FINAL ANSWER:" followed by a concise answer that keeps all specific details.""")
assert ANSWER_PROMPT_7STEP != ANSWER_PROMPT

# Nemori's judge, verbatim (MIT licence; github.com/nemori-ai/nemori @ d2a6dff,
# evaluation/locomo/metrics/llm_judge.py).
JUDGE_SYSTEM = "You are an expert grader that determines if answers to questions match a gold standard answer."
JUDGE_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a ’gold’ (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

_lock = threading.Lock()


def load_questions(locomo: Path) -> dict[str, dict]:
    out = {}
    for s in json.loads(locomo.read_text()):
        for i, q in enumerate(s["qa"]):
            if q.get("category") in CATS:
                out[f"{s['sample_id']}-q{i:03d}"] = {"sample_id": s["sample_id"], "category": q["category"],
                                                     "question": q["question"], "answer": str(q["answer"])}
    return out


def whole_conversation(sample: dict) -> str:
    """One line per message, each session's date as a header line (the released data's own text)."""
    c, out = sample["conversation"], []
    for n in sorted(int(m.group(1)) for k in c if (m := re.fullmatch(r"session_(\d+)", k))):
        if not c[f"session_{n}"]:
            continue
        out.append(f"\n[{c[f'session_{n}_date_time']}]")
        for t in c[f"session_{n}"]:
            line = f"{t['speaker']}: {t['text']}"
            if t.get("blip_caption"):
                line += f" [shares a photo: {t['blip_caption']}]"
            out.append(line)
    return "\n".join(out).strip()


def final_answer(text: str) -> str:
    m = list(re.finditer(r"final answer\s*[:\-]\s*", text, re.I))
    if m:
        return text[m[-1].end():].strip().strip("*").strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines[-1] if lines else text.strip()


def chat(model: str, messages: list, json_mode: bool) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENAI_API_KEY")
    body = {"model": model, "messages": messages, "temperature": 0.0}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(6):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if attempt < 5 and code in (None, 429, 500, 502, 503):
                time.sleep(2 ** attempt * 2)
                continue
            raise SystemExit(f"OpenAI error: {e}")
    raise SystemExit("OpenAI: too many retries")


def read_jsonl(p: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, p.read_text().splitlines())} if p.exists() else {}


def append(p: Path, row: dict) -> None:
    with _lock, open(p, "a", encoding="utf8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_all(fn, items, workers: int, what: str) -> None:
    with cf.ThreadPoolExecutor(workers) as ex:
        for n, fut in enumerate(cf.as_completed([ex.submit(fn, x) for x in items]), 1):
            fut.result()
            if n % 50 == 0 or n == len(items):
                print(f"{what}: {n}/{len(items)}", flush=True)


def pick(qs: dict, conv: str | None, limit: int | None) -> list[str]:
    ids = [i for i, q in qs.items() if conv in (None, q["sample_id"])]
    return ids[:limit] if limit else ids


def cmd_answer(a) -> None:
    qs = load_questions(a.locomo)
    samples = {s["sample_id"]: s for s in json.loads(a.locomo.read_text())}
    contexts = {}
    if a.context.startswith("file:"):
        contexts = {i: r["context"] for i, r in read_jsonl(Path(a.context[5:])).items()}
    elif a.context not in ("whole", "none"):
        raise SystemExit("--context is whole, none or file:<path>")
    whole = {}
    done = read_jsonl(a.out)
    todo = [i for i in pick(qs, a.conv, a.limit) if i not in done]
    prompt = ANSWER_PROMPT_7STEP if a.prompt == "7step" else ANSWER_PROMPT

    def one(i: str) -> None:
        q = qs[i]
        if a.context == "whole":
            with _lock:
                if q["sample_id"] not in whole:
                    whole[q["sample_id"]] = whole_conversation(samples[q["sample_id"]])
            ctx = whole[q["sample_id"]]
        elif a.context == "none":
            ctx = ""
        else:
            if i not in contexts:
                raise SystemExit(f"no context for {i}")
            ctx = contexts[i]
        raw = chat(a.model, [{"role": "user", "content": prompt.format(episodic=ctx, semantic="", question=q["question"])}],
                   json_mode=False)
        row = {"id": i, "model": a.model, "prompt": a.prompt, "context_chars": len(ctx),
               "answer": final_answer(raw) if a.prompt == "7step" else raw}
        append(a.out, row)

    run_all(one, todo, a.workers, "answer")


def cmd_judge(a) -> None:
    qs = load_questions(a.locomo)
    answers = read_jsonl(a.answers)
    done = read_jsonl(a.out)
    todo = [i for i in pick(qs, a.conv, a.limit) if i in answers and i not in done]

    def one(i: str) -> None:
        q = qs[i]
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_PROMPT.format(question=q["question"], gold_answer=q["answer"],
                                                               generated_answer=answers[i]["answer"])}]
        raw = chat(a.judge, msgs, json_mode=True)
        try:
            label = json.loads(raw).get("label")
        except json.JSONDecodeError:
            label = None
        append(a.out, {"id": i, "judge": a.judge, "label": label})

    run_all(one, todo, a.workers, "judge")


def cmd_score(a) -> None:
    qs = load_questions(a.locomo)
    v = read_jsonl(a.verdicts)
    ids = [i for i in pick(qs, a.conv, None) if i in v]
    if not ids:
        raise SystemExit("no verdicts for these questions")
    missing = len(pick(qs, a.conv, None)) - len(ids)
    rows = [("overall", ids)] + [(name, [i for i in ids if qs[i]["category"] == c]) for c, name in CATS.items()]
    for name, sel in rows:
        if sel:
            ok = sum(v[i]["label"] == "CORRECT" for i in sel)
            print(f"{name:12} {100 * ok / len(sel):5.1f}  ({ok}/{len(sel)})")
    if missing:
        print(f"note: {missing} questions have no verdict and are left out")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("answer", "judge", "score"):
        p = sub.add_parser(name)
        p.add_argument("--locomo", type=Path, required=True, help="path to LoCoMo's locomo10.json")
        p.add_argument("--conv", help="one conversation, e.g. conv-26")
        if name != "score":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--limit", type=int, help="first N questions only (for a dry run)")
    sub.choices["answer"].add_argument("--context", required=True, help="whole | none | file:<contexts.jsonl>")
    sub.choices["answer"].add_argument("--model", default="gpt-4o-mini")
    sub.choices["answer"].add_argument("--prompt", choices=["short", "7step"], default="short")
    sub.choices["judge"].add_argument("--answers", type=Path, required=True)
    sub.choices["judge"].add_argument("--judge", default="gpt-4o-mini")
    sub.choices["score"].add_argument("--verdicts", type=Path, required=True)
    a = ap.parse_args()
    {"answer": cmd_answer, "judge": cmd_judge, "score": cmd_score}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
