# Specification: Component 7 — Agent Skills

## 1. Skill Folder Convention & Discovery (`Skill`, `SkillRegistry`)
- A skill is a directory containing `SKILL.md` (matched case-insensitively) plus optional bundled files (e.g., `checklist.md`, `check_contract.py`).
- Frontmatter (`---` fenced `name` and `description`) is parsed using stdlib-only `parse_frontmatter()`; if omitted, the normalized folder name and first non-heading prose line are used as fallbacks.
- `SkillRegistry.discover(bases, skills_dir)` scans in precedence order (first match wins):
  1. Explicit `--skills-dir` (scanned both directly and for nested `skills/`, `skill/`, `.claude/skills/` directories)
  2. Workspace `cwd` (`skills/`, `skill/`, `.claude/skills/`)
  3. Repository root (`skills/`, `skill/`, `.claude/skills/`)
  4. Agent home directory (`AGENT_HOME`)

## 2. Three-Level Context Loading
1. **Catalog (Prefix):** Every discovered skill's `name: description` is rendered under `Skills:` in `MiniAgent.build_prefix()`, keeping the stable prefix small and cache-friendly.
2. **Activation (`Active skills`):** When activated, the full `SKILL.md` body (clipped to `MAX_SKILL_CHARS = 6000`) is injected via `active_skills_text()` outside the compressed transcript so `history_text()` compression never truncates active instructions. Up to `MAX_ACTIVE_SKILLS = 3` skills remain active (LRU eviction), persisted in `session["active_skills"]` and inherited by read-only delegated sub-agents.
3. **Bundled Resources:** Accessible on demand via `use_skill(skill, file)` with resolved-path containment checks rejecting `..` or symlink traversal outside the skill folder.

## 3. Similarity Check & Triggering
- **Tokenization (`skill_tokens`):** Lowercase alphanumeric tokens (`len >= 2`), stopword removal (`SKILL_STOPWORDS`), and lightweight suffix stemming (`-ing`, `-ed`, `-s`).
- **Scoring (`Skill.similarity_details`, `SkillRegistry.evaluate`):**
  - Direct skill-name mention (`Skill.mentioned_in`): `score = 100`, `cosine_similarity = 1.0`.
  - Otherwise: weighted overlap (`2 * |name_tokens ∩ P| + 1 * |description_tokens ∩ P|`) and normalized cosine overlap `|K ∩ P| / sqrt(|K| * |P|)`.
  - Match threshold: `score >= MIN_PROMPT_MATCH_SCORE` (`3`).
- **Trigger Paths:**
  - `auto` (`auto_activate_skills`): Runs on every `MiniAgent.ask(user_message)` before the first model call.
  - `model_tool` / `model_tool_bundled` (`tool_use_skill`): Invoked mid-execution by the model via `<tool>{"name":"use_skill","args":{...}}</tool>`.
  - `slash_command` (`run_skill_command`): Manual CLI activation via `/skill <name> [request]` or `/<name> [request]`.

## 4. Logging Contract (`MiniAgent.record_skill_event`)
Every similarity evaluation and skill execution appends a JSON event to:
- `session["skill_logs"]` (persisted in `.mini-coding-agent/sessions/<id>.json`), and
- `.mini-coding-agent/skill_logs.jsonl` (newline-delimited JSON).

### Event Schema
```json
{
  "timestamp": "UTC ISO-8601 string",
  "session_id": "string",
  "trigger": "auto | model_tool | model_tool_bundled | slash_command",
  "prompt": "raw user prompt string",
  "similarity_check": {
    "method": "weighted_token_overlap_and_cosine",
    "threshold": 3,
    "prompt_tokens": ["..."],
    "candidates": [
      {
        "skill": "string",
        "score": 100,
        "cosine_similarity": 1.0,
        "exact_mention": true,
        "matched_tokens": ["..."],
        "threshold": 3,
        "matched": true,
        "already_active": false
      }
    ],
    "selected": "skill-name or null"
  },
  "executed_skill": {
    "name": "skill-name",
    "path": "relative/or/abs/path/SKILL.md",
    "bundled_file": "optional relative filename or null"
  }
}
```
When no skill matches an `auto` check, `executed_skill` is `null` while `prompt` and `similarity_check` are still recorded. Users can inspect recent entries interactively with `/skill-logs`.

## 5. Sub-Agent Compatibility (`tool_delegate`, `find_subagent_file`)
- `delegate(task, max_steps=3)` without `agent` remains a bounded read-only sub-agent inheriting parent skills.
- `delegate(task, agent="<name>", max_steps=8)` locates `<name>.md` in `.claude/agents/` near the workspace or discovered skills, injects its instructions into an isolated child agent (no parent transcript leakage), and applies the parent's `approval_policy` so multi-agent skills (such as Lab 5's `build-q-learning`) can execute end to end.