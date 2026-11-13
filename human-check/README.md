# Human check of the LLM judges

The paper's "A human check of the judges" paragraph comes from these files.

- `sheet.csv`: 150 questions with the gold answer and one model's answer (gpt-4o-mini answers, short
  prompt), in random order. 60 Becos and 60 whole-conversation answers where the three judges
  (gpt-4o-mini, gpt-4.1-mini, DeepSeek) do not all agree, and 15 + 15 where they all agree.
- `key.json`: for each item, the question id, the system, whether the judges agreed, the LoCoMo
  category and each judge's verdict. Graders did not see it.
- `sheet-nikhil.csv`: one author's grades, made blind to system and judge: C (correct), W (wrong) or
  ? (cannot tell).
- `human_check.py score`: compares the grades with each judge.

```
python3 human-check/human_check.py score human-check/sheet-nikhil.csv
```

## Rubric graders used

- **C:** the answer states the gold answer's key fact and contradicts nothing in it. Other wording
  and extra correct detail are fine.
- **W:** the key fact is missing; the answer is vague but on topic; any part contradicts the gold
  answer; a list misses a key item or adds a wrong one.
- **Dates:** must name the same day, week, month or year as the gold answer, at its precision.
  "The Tuesday before 20 July 2023" is 18 July 2023. Off by a day or more is W.
- **?:** the gold answer itself looks wrong or ambiguous.

Licence: CC BY-NC 4.0, like the rest of `data/` (the questions and gold answers are LoCoMo's).
