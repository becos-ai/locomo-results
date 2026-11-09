# LoCoMo results for becos

https://github.com/becos-ai/locomo-results

Answers, judge verdicts and the evaluation script behind *What Moves a LoCoMo Score? Retrieval,
Answer Prompts and Judges in a Long-Term Conversational Memory Benchmark* (Nikhil Kumar Jha and Amitoj
Cheema, becos, 2026).

## What is here
- `locomo_eval.py`: answer, judge and score LoCoMo questions. Python 3.10+, standard library only.
- `data/answers/<system>_<prompt>_<answer model>.jsonl`: one `{"id", "answer"}` per question, 1,540
  questions each.
- `data/verdicts/<system>_<prompt>_<answer model>__<judge>.jsonl`: one `{"id", "judge", "label"}` per
  question (`CORRECT` or `WRONG`).

Systems: `becos` (context retrieved by becos), `whole` (the whole conversation), `none` (no context).
Prompts: `short` (Nemori's, answers under 5–6 words) and `7step` (ours; only the final answer is
judged). Answer models: gpt-4o-mini, gpt-4.1-mini. Judges: gpt-4o-mini, gpt-4.1-mini,
deepseek-v4-flash (the no-memory run has no DeepSeek verdicts).

## Use
Get `locomo10.json` from https://github.com/snap-research/locomo, then:

```
# reproduce a table row from our verdicts (no API calls)
python3 locomo_eval.py score --locomo locomo10.json --verdicts data/verdicts/becos_short_gpt-4o-mini__gpt-4o-mini.jsonl

# re-judge our answers with any OpenAI judge
export OPENAI_API_KEY=...
python3 locomo_eval.py judge --locomo locomo10.json --answers data/answers/becos_short_gpt-4o-mini.jsonl --judge gpt-4o-mini --out v.jsonl

# reproduce the whole-conversation and no-memory baselines
python3 locomo_eval.py answer --locomo locomo10.json --context whole --model gpt-4o-mini --prompt short --out a.jsonl

# score your own memory system the same way: one {"id", "context"} per question
python3 locomo_eval.py answer --locomo locomo10.json --context file:my_contexts.jsonl --out a.jsonl
```

`--conv conv-26` runs one conversation and `--limit 5` the first five questions (dry run). Every
step resumes from its `--out` file. Rough cost with gpt-4o-mini: about $5 for the whole-conversation
baseline over all 1,540 questions (about 21,000 prompt tokens each), under $0.50 to judge 1,540 answers.

Question ids are `<sample_id>-q<index in that sample's qa list>`; the adversarial category (5) is left
out, as in prior work. Category ids: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop.

becos itself is not part of this release. Researchers who want to run becos on their own data can
request API access at hello@becos.ai.

## Licences
- `locomo_eval.py`: MIT. It contains Nemori's answer and judge prompts, reproduced verbatim from
  nemori-ai/nemori at commit d2a6dff (`evaluation/locomo/search.py`, `metrics/llm_judge.py`), MIT
  licence; their notice is in `NOTICE-nemori.txt`. Our licence is in `LICENSE`.
- `data/`: CC BY-NC 4.0, because the answers respond to LoCoMo's questions (LoCoMo is CC BY-NC 4.0,
  Maharana et al., 2024). Use for non-commercial research and credit LoCoMo.
