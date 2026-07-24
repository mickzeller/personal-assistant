# Training Data: From Raw Writing to JSONL

How `src/training_data_parser.py` turns messy real-world writing samples into
clean fine-tuning data. This is the least glamorous and most important part of
the pipeline: data quality matters a ton more than quantity.

## Overview

Three kinds of source material go in, one JSONL file comes out:

```
  knowledge_base/training_data/
  ├── Perplexity exports (.md)   ──┐    "help me write this" conversations
  ├── Real messages (.docx)      ──┼──► writing_style.jsonl
  ├── Conversation logs (.md)    ──┤    Slack/SMS threads
  └── Plain markdown (.md)       ──┘    reviews, notes, direct writing
```

Every example ends up in the chat format MLX (and most fine-tuning tools) expect:

```json
{"messages": [
  {"role": "system",    "content": "You are a writing assistant that matches Mick's style..."},
  {"role": "user",      "content": "Context: casual text to a friend\n\nTask: ..."},
  {"role": "assistant", "content": "<my actual writing, or a version I approved>"}
]}
```

The model learns from (instruction → response) pairs: SYSTEM says who it's
writing as, USER gives the task and audience, ASSISTANT is the ideal output.
Each example also gets a classified context ("work message to a colleague",
"church or ministry context", etc.) baked into the user prompt so the model
learns to tone-shift by audience.

## Perplexity exports: mining approved drafts

These are multi-turn "help me write this" conversations, and they're the
trickiest to parse. Each conversation alternates user headings and AI responses
split by `---` dividers, and the parsing strategy is:

1. Capture the user's **initial request** as the prompt
2. Track the **last suggested version** as the candidate output
3. If the user explicitly confirms ("This!", "perfect", "send it"), lock in
   that version as approved
4. Multi-topic conversations get split into separate examples

Two edge cases worth remembering:

- **"This!" vs "this is to a..."** The first is a confirmation of the previous
  suggestion, the second starts a new topic. The parser checks first-line
  prefixes to tell them apart.
- **User drafts can be the gold.** Sometimes I paste my own draft and the AI
  just approves it. In that case the *draft* is the training example, not
  anything the AI wrote, which is even better data.

Extracting the suggested text is its own adventure since Perplexity formats
suggestions half a dozen ways. The extractor tries formats in order of
reliability: blockquotes, code blocks, bold quotes, bulleted options, numbered
options, standalone quoted lines. When a response lists multiple options it
grabs the LAST one, since Perplexity tends to put the best/most casual option
last.

## Raw writing (.docx) and conversation logs

These are pure gold: authentic examples of my voice with zero AI involvement.
The catch is there's no prompt attached, so the parser reverse-engineers a
plausible one from the content ("Write a casual message to my dev team", "Write
a warm, brief response in a church context"), classified by keyword matching.

Conversation logs (Slack/SMS exports) need extra cleanup:

- Speaker detection across several header formats ("mick 2:38 PM", "You
  sent...", bare "m"/"k" lines)
- Stripping noise: emoji reaction lines, thread markers, date headers,
  "(edited)" markers
- Back-to-back messages from me (continuation timestamps) get merged into one
  example, because that rapid-fire multi-bubble thing is how real texting works

Only my messages become training examples; the other side of the conversation
is just context that gets dropped.

## The quality filter

Before writing the JSONL, every example passes a filter:

| Rule                                     | Why                                                              |
|------------------------------------------|------------------------------------------------------------------|
| ≥ 10 chars                               | Shorter isn't a real message                                     |
| ≤ 4000 chars                             | Longer is an essay, not a message, and skews generation length   |
| Doesn't start with "how's this...", etc. | Means the parser grabbed the question side instead of the answer |

Cutting a bad example is always the right call. A handful of garbage examples
does more damage than a handful of good ones does good.

## The flywheel

`src/conversation_logger.py` quietly saves every chat exchange to
`data/conversations/` (local only, one JSONL per session). When it's time for
the next fine-tuning round, the best of those exchanges can be cherry-picked
into the training data, so the assistant improves from actually being used.

## Further reading

- [MLX-LM data format docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#data)
- `docs/FINE_TUNING.md` for what happens to this data next
