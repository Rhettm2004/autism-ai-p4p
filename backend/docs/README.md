# Project notes

Working notes for COMPSYS/SOFTENG 700 Project 18, kept so that the final report
can be written from recorded evidence rather than from memory, and so that the
research compendium has something to point at.

Everything here is version controlled with the code that produced it. A snapshot
is exported to `final report/notes/` for writing with
`python scripts/sync_report_notes.py`.

## How this is organised

| Directory | What goes in it | How it is written |
|---|---|---|
| `evidence/` | One append-only log per phase. Numbered findings, each with its numbers, where they came from, and the commit that produced them. | Append a new finding at the bottom. Never rewrite an existing one; if it turns out to be wrong, add a correcting finding and mark the old one superseded. |
| `report/` | One file per section of the final report. Outline, argument, and which findings support which claim. | Edited freely. Cites findings by id rather than repeating numbers. |
| `figures/` | Generated figures and their manifest. | `python scripts/make_report_figures.py`. Never edited by hand. |
| `references.md` | Every source we cite, with what it is used for and whether it is already cited. | Edited as sources are used. |
| `decisions.md` | Design decisions with their date, rationale, and the evidence behind them. | Appended when a decision is made. |
| `contributions.md` | Who did what, for the Statement of Contribution the report requires. | Appended as work happens. |
| `PHASE3_ROUTER_BRIEF.md` | Generated progression table for the router. | `python scripts/evaluate_router.py --brief`. Never edited by hand. |

## Finding ids

Every finding has an id of the form `F-P3-004`: phase, then a number in the order
it was recorded. Report sections cite these ids, so a claim in the report can
always be traced to a finding, and a finding to the data and commit behind it.

Ids are permanent. A superseded finding stays where it is with a note pointing at
whatever replaced it, because the report describes the progression and needs the
earlier state to still be readable.

## The rule about numbers

No number is typed into a note by hand if a file already holds it. A finding
either quotes a generated artefact (`data/router_eval/experiments.csv`,
`docs/PHASE3_ROUTER_BRIEF.md`, `data/history/*.csv`) and says which one, or it
records the command that produced it. Numbers that appear only in prose cannot be
checked later, and by October there will be too many of them to remember.

## Working loop

1. Make a change on a branch or in the working tree.
2. Run the relevant evaluation, recording it under a new label.
3. Rebuild the figures that depend on it.
4. Append a finding to the phase's evidence log, with the numbers, the artefact
   they came from, and what the change means.
5. Commit code, data, figures and notes together, so the note and the numbers it
   describes share a commit.

`.claude/skills/research-log/SKILL.md` encodes this loop.
