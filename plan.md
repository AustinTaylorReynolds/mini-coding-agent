# Implementation Plan: Agent Skills Component & Lab 4 Execution

## Stage 1 — Core Skill Discovery & Prompt Integration
- Implement `normalize_skill_name`, `parse_frontmatter`, `skill_tokens`, `Skill`, and `SkillRegistry` in `mini_coding_agent.py`.
- Scan `skills/`, `skill/`, `.claude/skills/`, and explicit `--skills-dir` locations.
- Register the safe `use_skill` tool in `MiniAgent.build_tools()` and expose the skill catalog in `MiniAgent.build_prefix()`.
- Render active skill instructions in `MiniAgent.active_skills_text()` so context reduction never clips them.

## Stage 2 — Similarity Diagnostics & Persistent Logging
- Add `Skill.similarity_details()` (weighted token overlap + cosine similarity) and `SkillRegistry.evaluate()` to return full ranked candidate reports.
- Add `MiniAgent.record_skill_event()` persisting every event (`prompt`, `similarity_check`, `executed_skill`, `trigger`) to `session["skill_logs"]` and `.mini-coding-agent/skill_logs.jsonl`.
- Hook logging into:
  - `auto_activate_skills()` (both matches and non-matches),
  - `tool_use_skill()` (mid-execution skill activation and bundled file reads), and
  - `run_skill_command()` (manual CLI `/skill` activations).
- Add the `/skill-logs` interactive CLI command and update `/help`.

## Stage 3 — Named Sub-Agent Support for Lab 5 Skills
- Add `MiniAgent.find_subagent_file()` to locate `.claude/agents/<name>.md` near the workspace or discovered skill folders.
- Extend `delegate` with an optional `agent` parameter so isolated sub-agents defined by Lab 5's `build-q-learning` skill can write their assigned modules under the parent's approval policy.

## Stage 4 — Verification & Testing
- Expand `tests/test_mini_coding_agent.py` to test:
  - `.claude/skills` discovery,
  - JSONL + session logging on auto-match, non-match, model-triggered `use_skill`, and `/skill-logs`,
  - named sub-agent delegation from `.claude/agents/`.
- Run `pytest -v` from `mini-coding-agent/` and verify all tests pass.

## Stage 5 — Solve Lab 4 Using Lab 5 Skills
1. Run `mini_coding_agent.py` against the Lab 4 directory with `--skills-dir` pointed at Lab 5's `.claude/skills`:
   ```powershell
   python .\mini_coding_agent.py `
     --cwd ..\Lab4 `
     --skills-dir ..\CS440_Group16_Lab-main\.claude\skills `
     --model qwen3.5:4b `
     --approval auto `
     --max-steps 25 `
     --max-new-tokens 2048 `
     "Build and implement the Q-learning vacuum robot for Lab 4 using the build-q-learning skill"
   ```
2. Run verification using the `verify-lab` skill:
   ```powershell
   python .\mini_coding_agent.py `
     --cwd ..\Lab4 `
     --skills-dir ..\CS440_Group16_Lab-main\.claude\skills `
     --model qwen3.5:4b `
     --approval auto `
     --max-steps 10 `
     "Verify Lab 4 and check whether the Q-learning results match value iteration"
   ```
3. Inspect `..\Lab4\.mini-coding-agent\skill_logs.jsonl` to confirm the logged prompt, similarity check scores, and executed skills for submission.