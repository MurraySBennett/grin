# grin — agent entry point

GRIN manuscript and GRT identifiability analysis

## Resume here

If you are a new session with no prior context, read in this order and stop
when you can state the next action:

1. `WORK.md` — the live work state: where things left off (`## Now`)
   and the open tasks by stream. **This alone is usually enough.**
2. `CHAT_STATE.md` — background: the fuller record this repo kept before
   WORK.md, still useful, no longer the resume-here target.
3. `git log --oneline -10` and `git status --short` — what actually happened
   last, and what is dirty right now.
4. Only if the live state names them: the specific files it names.

Do **not** read the whole repo or the full `docs/` tree to "get up to speed."
If the live state is not enough to act, that is a defect in the live state —
say so and fix it first.

There is exactly one live state file for this repo: `WORK.md`.
Do not create a second one, a dated copy, or a `-v2`. Overwrite it; git
history is the archive. `CHAT_STATE.md` stays as background.

## Work tracking — WORK.md

Maintain it as you work; never ask permission to update it.

- **Session start:** read `## Now` and the streams relevant to your task.
  If `## Inbox` has items, file them into the right stream first.
- **As you work:** when a task surfaces, add it under its stream with a
  horizon (`now`/`next`/`later`). When something is finished, mark it
  `[x] YYYY-MM-DD`. Cross off, don't delete.
- **Session end:** rewrite `## Now` (3-6 lines: where this ended up, what's
  next, what's blocked), check task states are current, and include WORK.md
  in your final commit.

Streams are a fixed vocabulary: `writing, lit-review, experiments, analysis,
collab, comms, admin, build, infra`. `build` is product work, `infra` is the
plumbing under it. Don't invent headings - declare extras in front-matter, or
use the closest and say so in the task text.


