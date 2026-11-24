---
name: pr-review
description: Review a pull request or a set of local changes (git diff) for correctness, tests, style and risks, then report findings. Use when the user asks for a PR review, code review, or to review a diff, branch or recent changes.
---

# PR Review

You only get a few tool calls per request, so be economical (aim for 4 or fewer).

## Steps

1. Find the change with `run_shell`: start with `git diff --stat`, then `git diff`.
   - If the diff is empty, try `git diff <default_branch>...HEAD` (the default branch is in the Workspace section) and `git log --oneline -5`.
   - If there is still nothing to review, say so in `<final>` and stop.
2. Only call `read_file` on a changed file when the diff alone does not give enough context.
3. Load the checklist once: `<tool>{"name":"use_skill","args":{"skill":"pr-review","file":"checklist.md"}}</tool>`, then apply it to the diff.
4. Finish with one `<final>` answer in exactly this format:

```
Summary: <one sentence on what the change does>
Blocking issues:
- <file:line> <problem and why it matters>   (or "none")
Suggestions:
- <file:line> <optional improvement>          (or "none")
Tests: <what is covered, what is missing>
Verdict: APPROVE or REQUEST CHANGES
```

## Rules

- This is a read-only task: do not modify any files.
- Cite `file:line` for every finding. Never invent code that is not in the diff.
- Separate blocking problems (bugs, security, data loss, missing tests for new logic) from style nits.
