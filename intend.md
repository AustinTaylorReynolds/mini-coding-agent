# Intent: Component 7 — Agent Skills for Mini-Coding-Agent

## Problem Statement
`mini-coding-agent` follows the six-component coding-agent architecture covered in class:
1. Live Repo Context (`WorkspaceContext`)
2. Prompt Shape and Cache Reuse (`build_prefix`, `memory_text`, `prompt`)
3. Structured Tools, Validation, and Permissions (`build_tools`, `run_tool`, `validate_tool`, `approve`)
4. Context Reduction and Output Management (`clip`, `history_text`)
5. Transcripts, Memory, and Resumption (`SessionStore`, `record`, `note_tool`)
6. Delegation and Bounded Subagents (`tool_delegate`)

However, it lacks the **Agent Skills** component. Without folder-based skills, reusable task workflows (such as pull-request reviews or the Lab 5 Q-learning build and verification pipeline) cannot be automatically discovered, matched to prompts, executed during an agent loop, or audited.

## Objectives
1. **Folder Skill Discovery:** Automatically discover folder-based skills containing a `SKILL.md` / `skill.md` file (e.g., `skills/pr-review/SKILL.md`, `skill/<name>/skill.md`, `.claude/skills/<name>/SKILL.md`, or an explicit `--skills-dir`).
2. **Prompt & Mid-Execution Triggering:** Execute a skill's instructions when:
   - a user prompt matches the skill's name or description (automatic similarity trigger),
   - the model invokes `use_skill` while solving a task (on-demand mid-execution trigger, including reading bundled helper files), or
   - a user explicitly activates it via `/skill <name>` or `/<name>`.
3. **Structured Observability & Logging:** Log every evaluated prompt, the full similarity check across candidate skills, and the executed skill (or `null` when no skill matches) to both session state and a persistent JSONL audit log.
4. **End-to-End Validation (Lab 5 Skill on Lab 4):** Enable `mini-coding-agent` to use the Lab 5 agent skills (`build-q-learning` and `verify-lab`) and their `.claude/agents/` sub-agent definitions to solve and verify Lab 4 without modifying Lab 4 or Lab 5 source guidelines.

## Constraints & Non-Goals
- Keep `mini_coding_agent.py` pure Python standard library at runtime (no external vector DB or heavy ML dependencies), backed by Ollama.
- Enforce strict path confinement so bundled skill files can never escape their skill folder.
- Preserve full backward compatibility with existing tools, session files, and unit tests.

## Success Criteria
- `pytest -v` passes all existing tests plus new tests for skill discovery, similarity evaluation, logging, and named sub-agent delegation.
- `.mini-coding-agent/skill_logs.jsonl` records `prompt`, `similarity_check`, and `executed_skill` for every evaluation and execution.
- Running `mini-coding-agent` against Lab 4 with `--skills-dir` pointed at Lab 5's skills discovers, logs, and executes the Lab 5 skills.