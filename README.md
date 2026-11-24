&nbsp;
# Mini-Coding-Agent

This folder contains a small standalone coding agent:

- code: `mini_coding_agent.py`
- CLI: `mini-coding-agent`

It is a minimal local agent loop with:

- workspace snapshot collection
- stable prompt plus turn state
- structured tools
- approval handling for risky tools
- transcript and memory persistence
- bounded delegation
- agent skills (`skills/<name>/SKILL.md`)

The model backend is currently based on Ollama.

<a href="https://magazine.sebastianraschka.com/p/components-of-a-coding-agent">
  <img src="https://substack-post-media.s3.amazonaws.com/public/images/49b97718-57f4-4977-99c8-8ad5c4d32af3_1548x862.png" width="500px">
</a>

<br>

**[The detailed tutorial: Components of a Coding Agent](https://magazine.sebastianraschka.com/p/components-of-a-coding-agent)**


&nbsp;
## Core Components

<a href="https://magazine.sebastianraschka.com/p/components-of-a-coding-agent">
  <img alt="Six core components of a coding agent" src="https://sebastianraschka.com/images/github/mini-coding-agent/six-components.webp" width="500px">
</a>

This coding harness is organized around six practical building blocks, plus a seventh (agent skills) added in Project 2:

1. **Live repo context**  
   The agent collects stable workspace facts upfront, such as repo layout, instructions, and git state.
2. **Prompt shape and cache reuse**  
   A stable prompt prefix, which is separate from the changing request, transcript, and memory so repeated model calls can reuse the static parts efficiently.
3. **Structured tools, validation, and permissions**  
   The model works through named tools with checked inputs, workspace path validation, and approval gates instead of free-form arbitrary actions.
4. **Context reduction and output management**  
   Long outputs are clipped, repeated reads are deduplicated, and older transcript entries are compressed to keep prompt size under control.
5. **Transcripts, memory, and resumption**  
   The runtime keeps both a full durable transcript and a smaller working memory so sessions can be resumed while preserving important state via working memory.
6. **Delegation and bounded subagents**  
   Scoped subtasks can be delegated to helper agents that inherit enough context to help (but operate within limits).
7. **Agent skills**  
   Reusable, folder-based instructions (`SKILL.md`) that the agent discovers, and loads only when a request or a step of the task needs them. See [Agent Skills](#agent-skills).

&nbsp;
## Requirements

You need:

- Python 3.10+
- Ollama installed
- an Ollama model pulled locally

Optional:

- `uv` for environment management and the `mini-coding-agent` CLI entry point

This project has no Python runtime dependency beyond the standard library, so you can run it directly with `python mini_coding_agent.py` if you do not want to use `uv`.

&nbsp;
## Install Ollama

Install Ollama on your machine so the `ollama` command is available in your shell.

Official installation link: [ollama.com/download](https://ollama.com/download)

Then verify:

```bash
ollama --help
```

Start the server:

```bash
ollama serve
```

In another terminal, pull a model. Example:

```bash
ollama pull qwen3.5:4b
```

Qwen 3.5 model library:

- [ollama.com/library/qwen3.5](https://ollama.com/library/qwen3.5)

The default in this project is `qwen3.5:4b`. If you have sufficient memory, it is worth trying a larger model such as `qwen3.5:9b` or another larger Qwen 3.5 variant. The agent just sends prompts to Ollama's `/api/generate` endpoint.

&nbsp;
## Project Setup

Clone the repo or your fork and change into it:

```bash
git clone https://github.com/rasbt/mini-coding-agent.git
cd mini-coding-agent
```

If you forked it first, use your fork URL instead:

```bash
git clone https://github.com/<your-github-user>/mini-coding-agent.git
cd mini-coding-agent
```



&nbsp;
## Basic Usage

Start the agent:

```bash
cd mini-coding-agent
uv run mini-coding-agent
```

Without `uv`, run the script directly:

```bash
cd mini-coding-agent
python mini_coding_agent.py
```

By default it uses:

- model: `qwen3.5:4b`
- approval: `ask`

For a concrete usage example, see [EXAMPLE.md](EXAMPLE.md).

&nbsp;
## Approval Modes

Risky tools such as shell commands and file writes are gated by approval.

- `--approval ask`
  prompts before risky actions (default and recommended)
- `--approval auto`
  allows risky actions automatically, including arbitrary command execution and file writes by the model; use only with trusted prompts and trusted repositories
- `--approval never`
  denies risky actions

Example:

```bash
uv run mini-coding-agent --approval auto
```



&nbsp;
## Resume Sessions

The agent saves sessions under the target workspace root in:

```text
.mini-coding-agent/sessions/
```

Resume the latest session:

```bash
uv run mini-coding-agent --resume latest
```


Resume a specific session:

```bash
uv run mini-coding-agent --resume 20260401-144025-2dd0aa
```


&nbsp;
## Interactive Commands

Inside the REPL, slash commands are handled directly by the agent instead of
being sent to the model as a normal task.

- `/help`
  shows the list of available interactive commands
- `/memory`
  prints the distilled session memory, including the current task, tracked files, and notes
- `/session`
  prints the path to the current saved session JSON file
- `/reset`
  clears the current session history and distilled memory but keeps you in the REPL
- `/exit`
  exits the interactive session
- `/quit`
  exits the interactive session; alias for `/exit`

&nbsp;
## Agent Skills

A skill is a folder with a `SKILL.md` file. Optional extra files in the folder are loaded on demand.

```text
skills/
  pr-review/
    SKILL.md        <- frontmatter + instructions
    checklist.md    <- optional bundled file
```

`SKILL.md` starts with a small header, followed by the instructions the agent should follow:

```markdown
---
name: pr-review
description: Review a pull request or local changes. Use when the user asks for a PR review or code review.
---

# PR Review
1. Run `git diff --stat` ...
```

The `description` matters most: it is what the agent uses to decide whether the skill fits a request.

**Where skills are discovered** (the first skill with a given name wins):

1. the folder given with `--skills-dir` (scanned directly, and also for nested `skills/`, `skill/` and `.claude/skills/` folders)
2. `skills/`, `skill/` or `.claude/skills/` in the workspace (`--cwd`, then the repo root)
3. `skills/` next to `mini_coding_agent.py`, so skills still work when `--cwd` points at another repo

`SKILL.md` and `skill.md` are both accepted.

**How a skill is triggered**

| Trigger | Example | What happens |
| --- | --- | --- |
| The request matches a skill | `Please review my pull request` | The agent activates the skill before the first model call. It matches on the skill name (`pr-review`, `pr review`) or on enough overlap with name + description words (see below). |
| You ask for it explicitly | `/skill pr-review check my last commit` or `/pr-review check my last commit` | The skill is activated, then the rest of the line is sent as the request. |
| The model needs it mid-task | `<tool>{"name":"use_skill","args":{"skill":"pr-review"}}</tool>` | The `use_skill` tool activates the skill; `use_skill` with `file` reads a bundled file (e.g. `checklist.md`). |

`/skills` lists the discovered skills; `*` marks the active ones. `/skill-logs` shows the most recent skill events (see below). `/agents` lists the named sub-agents and `/agent <name> [task]` runs one directly (see Named sub-agents). Slash commands also work in one-shot mode: `python mini_coding_agent.py /skill build-q-learning Build the lab`.

**How a skill is executed.** Skills load in three levels so the prompt stays small:

1. *Catalog*: only `name: description` of every skill sits in the stable prompt prefix.
2. *Activation*: the full `SKILL.md` body is added to an `Active skills` section of the prompt and the model follows it using the normal tools. This section is rebuilt on every step and is separate from the transcript, so context reduction never clips the instructions (bodies are capped at 6000 characters).
3. *Resources*: bundled files are read only when asked for via `use_skill`.

Active skills are saved with the session, survive `--resume`, are capped at 3 (the oldest is dropped), and are cleared by `/reset`.

**Skill log (audit trail).** Every similarity check and every skill execution is appended as one JSON object to the session file (`session["skill_logs"]`) and to `.mini-coding-agent/skill_logs.jsonl`, whether or not a skill matched:

```json
{
  "timestamp": "2026-11-23T06:28:38+00:00",
  "session_id": "20261122-232827-9b8e96",
  "trigger": "auto",
  "prompt": "Build the Q-learning vacuum robot for Lab 4 ...",
  "similarity_check": {
    "method": "weighted_token_overlap_and_cosine",
    "threshold": 3,
    "prompt_tokens": ["build", "q", "learning", "..."],
    "candidates": [{"skill": "build-q-learning", "score": 100, "cosine_similarity": 1.0,
                    "exact_mention": true, "matched_tokens": ["..."], "matched": true, "already_active": false}],
    "selected": "build-q-learning"
  },
  "executed_skill": {"name": "build-q-learning", "path": ".../SKILL.md", "bundled_file": null}
}
```

`trigger` is one of `auto` (prompt matched), `model_tool` / `model_tool_bundled` (the model called `use_skill`) or `slash_command` (`/skill ...`). `executed_skill` is `null` when no skill matched. Note that `.mini-coding-agent/` is git-ignored, so copy `skill_logs.jsonl` somewhere tracked if you need to submit it.

**Named sub-agents.** A skill can ship sub-agent definitions as `.claude/agents/<name>.md` (found next to the workspace or next to any discovered `.claude/skills/` folder). `delegate` with the `agent` argument starts an isolated child agent that gets only that file's instructions plus the task, with no parent transcript and no parent skills, and it runs under the parent's approval policy, so it can write files:

```text
<tool>{"name":"delegate","args":{"agent":"q-table-setup","task":"Implement your component."}}</tool>
```

A named sub-agent is a workflow stage, so its step budget is your `--max-steps` (never fewer than 8). The model's own `max_steps` argument is ignored for named sub-agents, because small models copy the example value of 3 and starve the stage. Plain `delegate` without `agent` is unchanged: a bounded, read-only investigator.

**Finishing a multi-stage skill.** If an active skill's instructions name several sub-agents (for example `q-table-setup`, then `reward-logic`, ...), the agent tracks which ones have finished in the current request:

- A sub-agent counts as *finished* only if it returned a final answer. If its instructions say it writes code (create, write, implement, ...), it must also have created or changed a file in the workspace. A sub-agent that reports "rewards.py created" but wrote nothing is marked failed and its result says so, including its last tool calls and the last reply the parser rejected.
- If the model gives a final answer while sub-agents are still unfinished, the agent sends it back (up to 3 times) with the exact next `delegate` call. If it still stops, the answer ends with `[Not completed: these sub-agents did not finish: ...]` so an unfinished run can never look like a success.
- This only applies to the top-level agent, and only to sub-agents that the skill's `SKILL.md` mentions by name.
- If the model sends two invalid or empty tool calls in a row while sub-agents are unfinished, the error message also spells out the exact next `delegate` call, so it cannot burn its whole step budget on `use_skill {}` / `run_shell {}`.
- A skill's bundled scripts (for example `check_contract.py`) live in the skill folder, not in the workspace. The `Active skills` section shows the script's full path, because `run_shell` starts in the workspace and a relative path would not be found.

If one stage keeps failing, run just that stage: `/agent reward-logic` (add your own task text after the name to override the default).

**Safety.** A skill only adds instructions. Anything it tells the model to do still goes through the normal tools, so `run_shell`, `write_file` and `patch_file` stay behind the approval policy, and `use_skill` can only read files inside the skill's own folder.

**Limitations.**

- Prompt matching is plain word overlap, not understanding. It needs name hits (worth 2 each) and description hits (worth 1 each) to add up to at least 3, so `review README` does not trigger `pr-review` but `write a request handler and review it` can. Use `/skill` or let the model call `use_skill` when the heuristic misses or over-triggers.
- Small local models may ignore `use_skill`; the prompt-triggered path does not depend on the model.
- Small models often mangle the tool format. The parser accepts JSON calls, `<tool name="..." ...></tool>`, attribute-only calls such as `<tool name="delegate" agent="x" task="y"/>`, and the `[tool:name] {json}` notation that the transcript shows the model (models copy it back verbatim), but it never acts on a tool body that was cut off mid-way. XML child elements are accepted for every tool argument (for example `<tool name="use_skill"><skill>x</skill></tool>`). When a reply is rejected, or a tool call fails, the session file keeps the model's raw text (`"raw"` on the retry notice or on the failed tool entry) so you can see why; the model itself never sees it.
- `HTTP 400 ... prompt is longer than the context length` means the model's context window is too small for that stage (its instructions plus the history of its own steps). Use a larger `--num-ctx`, for example `--num-ctx 32768`.
- Long multi-step orchestration (several sub-agents in one run) is hard for a 4B model. Running one stage per invocation is far more reliable than one long prompt, and `--ollama-timeout 900` avoids spurious "timed out" errors while a sub-agent works.
- Frontmatter supports simple `key: value` lines (and wrapped continuation lines) only, not full YAML.

&nbsp;
## Main CLI Flags

```bash
uv run mini-coding-agent --help
```

Without `uv`:

```bash
python mini_coding_agent.py --help
```

CLI flags are passed before the agent starts. Use them to choose the workspace,
model connection, resume behavior, approval mode, and generation limits.

Important flags:

- `--cwd`
  sets the workspace directory the agent should inspect and modify; default: `.`
- `--model`
  selects the Ollama model name, such as `qwen3.5:4b`; default: `qwen3.5:4b`
- `--host`
  points the agent at the Ollama server URL (usually not needed); default: `http://127.0.0.1:11434`
- `--ollama-timeout`
  controls how long the client waits for an Ollama response (usually not needed); default: `300` seconds
- `--resume`
  resumes a saved session by id or uses `latest`; default: start a new session
- `--approval`
  controls how risky tools are handled: `ask`, `auto`, or `never`; default: `ask`
- `--skills-dir`
  adds a folder of skills (`<name>/SKILL.md`) that is searched first; default: auto-discover (see [Agent Skills](#agent-skills))
- `--num-ctx`
  context window requested from Ollama, in tokens; default: `16384` (`0` keeps Ollama's own, much smaller default). Raise it if a stage fails with `the prompt is longer than the context length`
- `--max-steps`
  limits how many model and tool turns are allowed for one user request; default: `6`
- `--max-new-tokens`
  caps the model output length for each step; default: `512`
- `--temperature`
  controls sampling randomness; default: `0.2`
- `--top-p`
  controls nucleus sampling for generation; default: `0.9`

&nbsp;
## Example

See [EXAMPLE.md](EXAMPLE.md)

&nbsp;
## Notes & Tips

- The agent expects the model to emit either `<tool>...</tool>` or `<final>...</final>`.
- Different Ollama models will follow those instructions with different reliability.
- If the model does not follow the format well, use a stronger instruction-following model.
- The agent is intentionally small and optimized for readability, not robustness.
