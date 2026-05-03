## Context

Actaware turns unstructured documents into decision-grade data. Pulling specific facts out of messy real-world text is one of the core problems we solve every day. This task is a tiny slice of that.

---

## The Task

In the attached `snippets.txt` you'll find **7 short excerpts** from corporate sustainability reports. Each snippet *may or may not* contain the company's **annual CO2 emissions, expressed in metric tons**.

Your job:

1. Write a program that processes all 7 snippets and produces a JSON object mapping each snippet ID to an **array** of extracted entries. Each entry contains the CO2 value (in metric tons) and the reporting year. Return an empty array `[]` if no value is present.
2. Submit your code, the resulting JSON, and a **short writeup (max half a page)**.

### Expected output format

```json
{
  "snippet_1": [
    { "value": 9999, "year": 2021 }
  ],
  "snippet_2": [],
  "snippet_3": [
    { "value": 3500, "year": 2022 }
  ],
  ...
}
```

(The numbers above are illustrative, not the answer.)

---

## The Writeup (this matters as much as the code)

Keep it under **half a page**. Be concise - we value clarity over volume. Cover:

- **Approach** - what did you do, and why this approach over others you considered?
- **Assumptions** - what did you assume about the input? Where could those assumptions break?
- **Edge cases** - which tricky cases did you notice in the snippets? Did you handle them all? Which ones did you skip and why?
- **Scaling** - how would your approach change if you had to process **100,000 documents** instead of 6? What breaks first?
- **Time spent** - roughly how long did this take you, and what would you do with another hour?

---

## Rules

- Use **any library, framework, or tool** you want. Seriously - any.
- Aim for **30-60 minutes** total. Don't over-engineer it.
- We care more about your reasoning than about perfect code.
- If you get stuck on something, write down what you tried and move on.

---

## How to Submit

Send a ZIP file (or a link to a public repo) containing:

1. Your code
2. The output JSON
3. Your writeup (PDF or markdown)

Email it to **job@actaware.com** with the subject line: `[Task] Your Name`
