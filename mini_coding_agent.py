import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
HELP_TEXT = "/help, /memory, /session, /reset, /skills, /skill, /skill-logs, /agents, /agent, /exit"
WELCOME_ART = (
    "/\\     /\\\\",
    "{  `---'  }",
    "{  O   O  }",
    "~~>  V  <~~",
    "\\\\  \\|/  /",
    "`-----'__",
)
HELP_LINES = [
    "Commands:",
    "/help        Show this help message.",
    "/memory      Show the agent's distilled working memory.",
    "/session     Show the path to the saved session file.",
    "/reset       Clear the current session history and memory.",
    "/skills      List the agent skills that were discovered.",
    "/skill       Activate a skill: /skill <name> [request] (or /<name> [request]).",
    "/skill-logs  Show logged prompts, similarity checks, and executed skills.",
    "/agents      List the named sub-agents (.claude/agents) that were discovered.",
    "/agent       Run one named sub-agent directly: /agent <name> [task].",
    "/exit        Exit the agent.",
]
HELP_DETAILS = "\n".join(HELP_LINES)
MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000
IGNORED_PATH_NAMES = {".git", ".mini-coding-agent", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}
AGENT_HOME = Path(__file__).resolve().parent
SKILL_DIR_NAMES = ("skills", "skill", ".claude/skills")
SKILL_FILE_NAME = "skill.md"  # matched case-insensitively, so SKILL.md and skill.md both work
MAX_SKILL_CHARS = 6000
MAX_ACTIVE_SKILLS = 3
NAMED_SUBAGENT_STEPS = 8  # minimum budget for delegate(agent=...); a larger --max-steps wins
DEFAULT_NUM_CTX = 16384  # Ollama's own default is too small for a sub-agent's instructions plus its history
MAX_COMPLETION_NUDGES = 3  # times a premature <final> is sent back while sub-agents are unfinished
XML_ARG_KEYS = (
    "content", "old_text", "new_text", "command", "task", "pattern", "path",
    "skill", "agent", "file", "timeout", "max_steps", "start", "end",
)  # every argument a tool takes, so <tool name="use_skill"><skill>x</skill></tool> is not read as args={}
INVALID_ARG_HINT_AFTER = 2  # consecutive invalid-argument errors before the exact next call is spelled out
DEFAULT_SUBAGENT_TASK = "Implement your component as described in your instructions."
MIN_PROMPT_MATCH_SCORE = 3
SKILL_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "without", "my", "our",
    "your", "me", "you", "i", "it", "its", "is", "are", "be", "this", "that", "these", "those",
    "please", "can", "could", "would", "should", "will", "when", "whenever", "use", "used", "using",
    "user", "users", "ask", "asks", "asked", "need", "needs", "want", "wants", "help", "any", "all",
    "so", "as", "at", "by", "from", "into", "about", "then", "than", "if", "do", "does", "did",
    "what", "how", "also", "new", "get", "make",
})

##############################
#### Agent Components ########
##############################
# 1) Live Repo Context -> WorkspaceContext
# 2) Prompt Shape And Cache Reuse -> build_prefix, memory_text, prompt
# 3) Structured Tools, Validation, And Permissions -> build_tools, run_tool, validate_tool, approve, parse, path, tool_*
# 4) Context Reduction And Output Management -> clip, history_text
# 5) Transcripts, Memory, And Resumption -> SessionStore, record, note_tool, ask, reset
# 6) Delegation And Bounded Subagents -> tool_delegate, find_subagent_file
# 7) Agent Skills -> SkillRegistry, Skill, use_skill, activate_skill, auto_activate_skills, record_skill_event, active_skills_text


def now():
    return datetime.now(timezone.utc).isoformat()


# Supporting helper for component 4 (context reduction and output management).
def clip(text, limit=MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def middle(text, limit):
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


##############################
#### 1) Live Repo Context ####
##############################
class WorkspaceContext:
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @classmethod
    def build(cls, cwd):
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:  # noqa: BLE001
                return fallback

        repo_root = Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        docs = {}
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main").removeprefix("origin/"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self):
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return "\n".join([
            "Workspace:",
            f"- cwd: {self.cwd}",
            f"- repo_root: {self.repo_root}",
            f"- branch: {self.branch}",
            f"- default_branch: {self.default_branch}",
            "- status:",
            self.status,
            "- recent_commits:",
            commits,
            "- project_docs:",
            docs,
        ])


##############################
#### 5) Session Memory #######
##############################
class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id):
        return self.root / f"{session_id}.json"

    def save(self, session):
        path = self.path(session["id"])
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return path

    def load(self, session_id):
        return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None


##############################
#### 7) Agent Skills #########
##############################
# A skill is a folder holding a SKILL.md file (plus optional bundled files):
#
#   skills/pr-review/SKILL.md      <- frontmatter (name, description) + instructions
#   skills/pr-review/checklist.md  <- optional resources, loaded only when needed
#
# Skills are loaded in three levels so the prompt stays small:
#   1. catalog:    name + description of every skill, always in the stable prefix
#   2. activation: the full SKILL.md body, added to the prompt only while the skill is active
#   3. resources:  bundled files, read on demand through use_skill(skill, file)
def normalize_skill_name(name):
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def parse_frontmatter(text):
    """Split an optional `---` fenced `key: value` header from the markdown body (stdlib only)."""
    lines = str(text).lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, "\n".join(lines).strip()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, "\n".join(lines).strip()

    meta, key = {}, None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t" and key:  # continuation of a wrapped value
            meta[key] = f"{meta[key]} {line.strip()}".strip()
            continue
        if ":" not in line:
            key = None
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        meta[key] = "" if value in {">", "|", ">-", "|-"} else value
    for key, value in meta.items():
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            meta[key] = value[1:-1]
    return meta, "\n".join(lines[end + 1:]).strip()


def skill_tokens(text):
    tokens = set()
    for word in re.findall(r"[a-z0-9]+", str(text).lower()):
        if len(word) < 2 or word in SKILL_STOPWORDS:
            continue
        for suffix in ("ing", "ed", "s"):  # tiny stemmer: reviewing/reviews -> review
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        tokens.add(word)
    return tokens


class Skill:
    def __init__(self, name, description, body, path):
        self.name = name
        self.description = description
        self.body = body
        self.path = Path(path)  # the SKILL.md file
        self.folder = self.path.parent
        self.name_tokens = skill_tokens(name)
        self.keywords = self.name_tokens | skill_tokens(description)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        try:
            skill_file = next(
                item for item in sorted(folder.iterdir())
                if item.is_file() and item.name.lower() == SKILL_FILE_NAME
            )
            meta, body = parse_frontmatter(skill_file.read_text(encoding="utf-8", errors="replace"))
        except (StopIteration, OSError):
            return None
        name = normalize_skill_name(meta.get("name", "")) or normalize_skill_name(folder.name)
        if not name or not body:
            return None
        description = meta.get("description", "").strip()
        if not description:  # fall back to the first prose line of the body
            first = next((line.strip() for line in body.splitlines() if line.strip() and not line.startswith("#")), "")
            description = first or name
        return cls(name, clip(description, 400), body, skill_file)

    def mentioned_in(self, text):
        """True when the request names the skill itself, e.g. 'pr-review', 'pr review', 'PR_Review'."""
        spaced = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
        phrase = re.sub(r"[^a-z0-9]+", " ", self.name)
        return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", spaced) is not None

    def similarity_details(self, text):
        """Compute weighted token overlap and normalized cosine similarity against a prompt."""
        prompt_tokens = skill_tokens(text)
        matched = sorted(self.keywords & prompt_tokens)
        exact_mention = self.mentioned_in(text)
        raw_score = 100 if exact_mention else sum(2 if token in self.name_tokens else 1 for token in matched)
        if exact_mention:
            cosine = 1.0
        elif prompt_tokens and self.keywords:
            cosine = round(len(matched) / math.sqrt(len(prompt_tokens) * len(self.keywords)), 4)
        else:
            cosine = 0.0
        return {
            "skill": self.name,
            "score": raw_score,
            "cosine_similarity": cosine,
            "exact_mention": exact_mention,
            "matched_tokens": matched,
            "threshold": MIN_PROMPT_MATCH_SCORE,
            "matched": raw_score >= MIN_PROMPT_MATCH_SCORE,
        }

    def score(self, text):
        return self.similarity_details(text)["score"]

    def bundled_files(self, limit=20):
        files = []
        for item in sorted(self.folder.rglob("*")):
            relative = item.relative_to(self.folder)
            if not item.is_file() or item == self.path:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative.parts):
                continue
            files.append(relative.as_posix())
        return files[:limit]

    def read_bundled(self, relative):
        folder = self.folder.resolve()
        target = (folder / str(relative)).resolve()
        if target != folder and folder not in target.parents:
            raise ValueError(f"file escapes skill folder: {relative}")
        if not target.is_file():
            raise ValueError(f"no such file in skill '{self.name}': {relative} (bundled: {', '.join(self.bundled_files()) or 'none'})")
        return f"# {self.name}/{relative}\n" + target.read_text(encoding="utf-8", errors="replace")

    def display_folder(self, root):
        try:
            return self.folder.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            return str(self.folder.resolve())

    def render(self, root):
        lines = [f"[skill: {self.name}] folder: {self.display_folder(root)}", clip(self.body, MAX_SKILL_CHARS)]
        files = self.bundled_files()
        if files:
            lines.append(f"Bundled files (read with use_skill file=...): {', '.join(files)}")
            scripts = [name for name in files if name.endswith(".py")]
            if scripts:
                # run_shell starts in the workspace, so a bare relative path to a skill script fails
                lines.append(f'Run a bundled script with its full path, e.g. python "{self.folder.resolve() / scripts[0]}"')
        return "\n".join(lines)


class SkillRegistry:
    def __init__(self, skills=()):
        self.skills = {}
        for skill in skills:
            self.skills.setdefault(skill.name, skill)  # first one wins

    def __len__(self):
        return len(self.skills)

    def __iter__(self):
        return iter(self.skills.values())

    def names(self):
        return sorted(self.skills)

    @classmethod
    def discover(cls, bases=(), skills_dir=None):
        """Scan <base>/(skills|skill|.claude/skills)/*/SKILL.md for every base (earlier bases override later ones).

        An explicit skills_dir is scanned first (both directly and its nested skill folders).
        """
        dirs = []
        if skills_dir:
            explicit = Path(skills_dir)
            dirs.append(explicit)
            dirs.extend(explicit / name for name in SKILL_DIR_NAMES)
        dirs += [Path(base) / name for base in bases for name in SKILL_DIR_NAMES]
        found, seen = [], set()
        for directory in dirs:
            try:
                directory = directory.resolve()
                if directory in seen or not directory.is_dir():
                    continue
                seen.add(directory)
                folders = sorted(directory.iterdir(), key=lambda item: item.name.lower())
            except OSError:
                continue
            for folder in folders:
                if folder.is_dir() and not folder.name.startswith(".") and folder.name not in IGNORED_PATH_NAMES:
                    skill = Skill.load(folder)
                    if skill:
                        found.append(skill)
        return cls(found)

    def find(self, name):
        return self.skills.get(normalize_skill_name(name))

    def evaluate(self, request, exclude=()):
        """Score every discovered skill against `request` and return (best_skill, similarity_report)."""
        excluded = set(exclude)
        candidates = []
        for skill in self:
            details = skill.similarity_details(request)
            details["already_active"] = skill.name in excluded
            candidates.append(details)
        candidates.sort(key=lambda item: (-item["score"], -item["cosine_similarity"], item["skill"]))

        eligible = [item for item in candidates if not item["already_active"] and item["matched"]]
        best = self.skills[eligible[0]["skill"]] if eligible else None
        report = {
            "method": "weighted_token_overlap_and_cosine",
            "threshold": MIN_PROMPT_MATCH_SCORE,
            "prompt_tokens": sorted(skill_tokens(request)),
            "candidates": candidates,
            "selected": best.name if best else None,
        }
        return best, report

    def match(self, request, exclude=()):
        """Return the best skill for a request, or None. Deterministic, so it works with any model."""
        best, _ = self.evaluate(request, exclude=exclude)
        return best

    def catalog_text(self):
        return "\n".join(f"- {skill.name}: {skill.description}" for skill in self)


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)


class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout, num_ctx=None):
        self.num_ctx = num_ctx  # context window in tokens; None keeps Ollama's (small) default
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def complete(self, prompt, max_new_tokens):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        if self.num_ctx:
            payload["options"]["num_ctx"] = int(self.num_ctx)
        request = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"Ollama request failed with HTTP {exc.code}: {body}"
            if "context length" in body:
                message += (
                    f"\nThe prompt no longer fits the model's context window (num_ctx={self.num_ctx or 'Ollama default'}). "
                    "Run again with a larger window, for example --num-ctx 32768."
                )
            raise RuntimeError(message) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")


class MiniAgent:
    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        skills=None,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        # Skills: workspace folders first, then the folder this agent was installed in.
        self.skills = skills if skills is not None else SkillRegistry.discover([Path(workspace.cwd), self.root, AGENT_HOME])
        self.skill_log = None  # optional callback(str) so the CLI can show skill activations live
        self.subagent_status = {}  # sub-agent name -> "done" | "failed", for the current request only
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.session = session or {
            "id": datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": {"task": "", "files": [], "notes": []},
            "active_skills": [],
            "skill_logs": [],
        }
        self.session.setdefault("active_skills", [])  # sessions saved before skills existed
        self.session.setdefault("skill_logs", [])
        self.skill_log_path = self.session_store.root.parent / "skill_logs.jsonl"
        self.last_user_prompt = ""
        self.tools = self.build_tools()
        self.prefix = self.build_prefix()
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    @staticmethod
    def remember(bucket, item, limit):
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    ###############################################
    #### 3) Structured Tools And Permissions ######
    ###############################################
    def build_tools(self):
        tools = {
            "list_files": {
                "schema": {"path": "str='.'"},
                "risky": False,
                "description": "List files in the workspace.",
                "run": self.tool_list_files,
            },
            "read_file": {
                "schema": {"path": "str", "start": "int=1", "end": "int=200"},
                "risky": False,
                "description": "Read a UTF-8 file by line range.",
                "run": self.tool_read_file,
            },
            "search": {
                "schema": {"pattern": "str", "path": "str='.'"},
                "risky": False,
                "description": "Search the workspace with rg or a simple fallback.",
                "run": self.tool_search,
            },
            "run_shell": {
                "schema": {"command": "str", "timeout": "int=20"},
                "risky": True,
                "description": "Run a shell command in the repo root.",
                "run": self.tool_run_shell,
            },
            "write_file": {
                "schema": {"path": "str", "content": "str"},
                "risky": True,
                "description": "Write a text file.",
                "run": self.tool_write_file,
            },
            "patch_file": {
                "schema": {"path": "str", "old_text": "str", "new_text": "str"},
                "risky": True,
                "description": "Replace one exact text block in a file.",
                "run": self.tool_patch_file,
            },
        }
        if self.skills:
            tools["use_skill"] = {
                "schema": {"skill": "str", "file": "str=''"},
                "risky": False,
                "description": "Load an agent skill's instructions when the task matches it, or read one of its bundled files.",
                "run": self.tool_use_skill,
            }
        if self.depth < self.max_depth:
            tools["delegate"] = {
                "schema": {"task": "str", "max_steps": "int=3", "agent": "str=''"},
                "risky": False,
                "description": "Ask a bounded read-only child agent to investigate, or dispatch a named sub-agent from .claude/agents/.",
                "run": self.tool_delegate,
            }
        return tools

    ############################################
    #### 2) Prompt Shape And Cache Reuse #######
    ############################################
    def build_prefix(self):
        tool_lines = []
        for name, tool in self.tools.items():
            fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
            risk = "approval required" if tool["risky"] else "safe"
            tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
        tool_text = "\n".join(tool_lines)
        example_lines = [
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "<final>Done.</final>",
        ]
        examples = "\n".join(example_lines)
        rule_lines = [
            "- Use tools instead of guessing about the workspace.",
            "- Return exactly one <tool>...</tool> or one <final>...</final>.",
            "- Tool calls must look like:",
            '  <tool>{"name":"tool_name","args":{...}}</tool>',
            "- For write_file and patch_file with multi-line text, prefer XML style:",
            '  <tool name="write_file" path="file.py"><content>...</content></tool>',
            "- Final answers must look like:",
            "  <final>your answer</final>",
            "- Never invent tool results.",
            "- Keep answers concise and concrete.",
            "- If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
            "- Before writing tests for existing code, read the implementation first.",
            "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
            "- New files should be complete and runnable, including obvious imports.",
            "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.",
            "- Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, use_skill, or delegate with args={}.",
        ]
        sections = ["You are Mini-Coding-Agent, a small local coding agent running through Ollama."]
        if self.skills:
            rule_lines += [
                "- If a skill listed under Skills fits the task, call use_skill with its name before doing the work, then follow its instructions.",
                "- Instructions of active skills appear under Active skills in the prompt; follow them step by step.",
            ]
            example_skill = self.skills.names()[0]
            examples += "\n" + f'<tool>{{"name":"use_skill","args":{{"skill":"{example_skill}"}}}}</tool>'
        sections += ["Rules:\n" + "\n".join(rule_lines), "Tools:\n" + tool_text]
        if self.skills:
            sections.append("Skills:\n" + self.skills.catalog_text())
        sections += ["Valid response examples:\n" + examples, self.workspace.text()]
        return "\n\n".join(sections)

    def memory_text(self):
        memory = self.session["memory"]
        notes = "\n".join(f"- {note}" for note in memory["notes"]) or "- none"
        return "\n".join([
            "Memory:",
            f"- task: {memory['task'] or '-'}",
            f"- files: {', '.join(memory['files']) or '-'}",
            "- notes:",
            notes,
        ])

    #####################################################
    #### 4) Context Reduction And Output Management #####
    #####################################################
    def history_text(self):
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] in ("write_file", "patch_file"):
                path = str(item["args"].get("path", ""))
                seen_reads.discard(path)
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    ########################################################
    #### 2) Prompt Shape And Cache Reuse (Continued) #######
    ########################################################
    def prompt(self, user_message):
        parts = [self.prefix, self.memory_text()]
        active = self.active_skills_text()
        if active:
            parts.append(active)
        parts += [
            "Transcript:\n" + self.history_text(),
            "Current user request:\n" + user_message,
        ]
        return "\n\n".join(parts)

    ###############################################
    #### 5) Session Memory (Continued) ###########
    ###############################################
    def record(self, item):
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def note_tool(self, name, args, result):
        memory = self.session["memory"]
        path = args.get("path")
        if name in {"read_file", "write_file", "patch_file"} and path:
            self.remember(memory["files"], str(path), 8)
        note = f"{name}: {clip(str(result).replace(chr(10), ' '), 220)}"
        self.remember(memory["notes"], note, 5)

    def ask(self, user_message):
        self.last_user_prompt = user_message
        self.subagent_status = {}
        memory = self.session["memory"]
        if not memory["task"]:
            memory["task"] = clip(user_message.strip(), 300)
        self.record({"role": "user", "content": user_message, "created_at": now()})
        self.auto_activate_skills(user_message)

        tool_steps = 0
        attempts = 0
        nudges = 0
        invalid_streak = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            raw = self.model_client.complete(self.prompt(user_message), self.max_new_tokens)
            kind, payload = self.parse(raw)

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                result = self.run_tool(name, args)
                invalid_streak = invalid_streak + 1 if result.startswith("error: invalid arguments") else 0
                pending = self.pending_subagents()
                if invalid_streak >= INVALID_ARG_HINT_AFTER and pending:
                    result += (
                        f"\nYou keep sending empty or invalid arguments. Sub-agents still unfinished: "
                        f"{', '.join(pending)}. {self.next_call_hint(pending)}"
                    )
                item = {"role": "tool", "name": name, "args": args, "content": result, "created_at": now()}
                if result.startswith("error:"):
                    item["raw"] = clip(str(raw), 600)  # debugging only: history_text() never shows it to the model
                self.record(item)
                self.note_tool(name, args, result)
                continue

            if kind == "retry":
                # "raw" is only for debugging the session file; history_text() never puts it in the prompt.
                self.record({"role": "assistant", "content": payload, "raw": clip(str(raw), 600), "created_at": now()})
                continue

            final = (payload or raw).strip()
            pending = self.pending_subagents()
            if pending and nudges < MAX_COMPLETION_NUDGES:
                # The model announced it was done while the active skill still has unfinished
                # sub-agents (small models do this a lot). Send it back with the exact next call.
                nudges += 1
                self.record(
                    {
                        "role": "assistant",
                        "content": self.completion_notice(pending),
                        "raw": clip(final, 300),
                        "created_at": now(),
                    }
                )
                continue
            if pending:
                final += f"\n\n[Not completed: these sub-agents did not finish: {', '.join(pending)}.]"
            self.record({"role": "assistant", "content": final, "created_at": now()})
            self.remember(memory["notes"], clip(final, 220), 5)
            return final

        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
        else:
            final = "Stopped after reaching the step limit without a final answer."
        self.record({"role": "assistant", "content": final, "created_at": now()})
        return final

    #############################################################
    #### 3) Structured Tools, Validation, And Permissions #######
    #############################################################
    def run_tool(self, name, args):
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            self.validate_tool(name, args)
        except Exception as exc:  # noqa: BLE001
            example = self.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            return message
        if self.repeated_tool_call(name, args):
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tool["risky"] and not self.approve(name, args):
            return f"error: approval denied for {name}"
        try:
            return clip(tool["run"](args))
        except Exception as exc:  # noqa: BLE001
            return f"error: tool {name} failed: {exc}"

    def repeated_tool_call(self, name, args):
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item["name"] == name and item["args"] == args for item in recent)

    def tool_example(self, name):
        skill_name = self.skills.names()[0] if self.skills else "pr-review"
        examples = {
            "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
            "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
            "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
            "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
            "use_skill": f'<tool>{{"name":"use_skill","args":{{"skill":"{skill_name}"}}}}</tool>',
        }
        return examples.get(name, "")

    def validate_tool(self, name, args):
        args = args or {}

        if name == "list_files":
            path = self.path(args.get("path", "."))
            if not path.is_dir():
                raise ValueError("path is not a directory")
            return

        if name == "read_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            start = int(args.get("start", 1))
            end = int(args.get("end", 200))
            if start < 1 or end < start:
                raise ValueError("invalid line range")
            return

        if name == "search":
            pattern = str(args.get("pattern", "")).strip()
            if not pattern:
                raise ValueError("pattern must not be empty")
            self.path(args.get("path", "."))
            return

        if name == "run_shell":
            command = str(args.get("command", "")).strip()
            if not command:
                raise ValueError("command must not be empty")
            timeout = int(args.get("timeout", 20))
            if timeout < 1 or timeout > 120:
                raise ValueError("timeout must be in [1, 120]")
            return

        if name == "write_file":
            path = self.path(args["path"])
            if path.exists() and path.is_dir():
                raise ValueError("path is a directory")
            if "content" not in args:
                raise ValueError("missing content")
            return

        if name == "patch_file":
            path = self.path(args["path"])
            if not path.is_file():
                raise ValueError("path is not a file")
            old_text = str(args.get("old_text", ""))
            if not old_text:
                raise ValueError("old_text must not be empty")
            if "new_text" not in args:
                raise ValueError("missing new_text")
            text = path.read_text(encoding="utf-8")
            count = text.count(old_text)
            if count != 1:
                raise ValueError(f"old_text must occur exactly once, found {count}")
            return

        if name == "use_skill":
            skill_name = str(args.get("skill", "")).strip()
            if not skill_name:
                raise ValueError("skill must not be empty")
            if self.skills.find(skill_name) is None:
                raise ValueError(f"unknown skill '{skill_name}'; available: {', '.join(self.skills.names())}")
            return

        if name == "delegate":
            if self.depth >= self.max_depth:
                raise ValueError("delegate depth exceeded")
            task = str(args.get("task", "")).strip()
            if not task:
                raise ValueError("task must not be empty")
            agent_name = str(args.get("agent", "")).strip()
            if agent_name and self.find_subagent_file(agent_name) is None:
                raise ValueError(f"unknown sub-agent '{agent_name}'")
            return

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        raw = str(raw)
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = MiniAgent.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:  # noqa: BLE001
                return "retry", MiniAgent.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", MiniAgent.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", MiniAgent.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", MiniAgent.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = MiniAgent.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", MiniAgent.retry_notice()
        bracket = MiniAgent.parse_bracket_tool(raw)
        if bracket is not None and ("<final>" not in raw or raw.find("[tool:") < raw.find("<final>")):
            return bracket
        if "<final>" in raw:
            final = MiniAgent.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", MiniAgent.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", MiniAgent.retry_notice("model returned an empty response")

    @staticmethod
    def parse_bracket_tool(raw):
        """Accept '[tool:NAME] {json args}', the notation the transcript shows the model.

        Small models copy it back verbatim. Without this the reply was taken as a final answer
        and the run ended after one step. Returns None when the reply has no bracket call.
        """
        match = re.search(r"\[tool:(?P<name>[A-Za-z0-9_-]+)\]\s*(?=\{)", raw)
        if not match:
            return None
        try:
            # strict=False lets literal newlines sit inside JSON strings, which models do when writing code
            args, _ = json.JSONDecoder(strict=False).raw_decode(raw, match.end())
        except ValueError:
            return "retry", MiniAgent.retry_notice("model returned malformed bracket tool JSON")
        if not isinstance(args, dict):
            return "retry", MiniAgent.retry_notice("tool arguments must be a JSON object")
        return "tool", {"name": match.group("name"), "args": args}

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        # Small models often write attribute-only calls: <tool name="delegate" agent="x" task="y"/>
        # (self-closing) or forget </tool>. Accept those, but never act on a body cut off mid-way,
        # because that would write half a file and report success.
        opening = re.search(r"""<tool(?P<attrs>(?:[^>"']|"[^"]*"|'[^']*')*)>""", raw)
        if not opening:
            return None
        attrs_text = opening.group("attrs")
        rest = raw[opening.end():]
        if attrs_text.rstrip().endswith("/"):
            body = ""
        elif "</tool>" in rest:
            body = rest[: rest.index("</tool>")]
        elif not rest.strip():
            body = ""
        else:
            return None
        attrs = MiniAgent.parse_attrs(attrs_text)
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        args = dict(attrs)
        for key in XML_ARG_KEYS:
            if f"<{key}>" in body:
                args[key] = MiniAgent.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        self.session["history"] = []
        self.session["memory"] = {"task": "", "files": [], "notes": []}
        self.session["active_skills"] = []
        self.session["skill_logs"] = []
        self.last_user_prompt = ""
        self.session_store.save(self.session)

    def path_is_within_root(self, resolved):
        probe = resolved
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        for candidate in (probe, *probe.parents):
            try:
                if candidate.samefile(self.root):
                    return True
            except OSError:
                continue
        return False

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not self.path_is_within_root(resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved

    def tool_list_files(self, args):
        path = self.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        entries = [
            item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            if item.name not in IGNORED_PATH_NAMES
        ]
        lines = []
        for entry in entries[:200]:
            kind = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{kind} {entry.relative_to(self.root)}")
        return "\n".join(lines) or "(empty)"

    def tool_read_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
        return f"# {path.relative_to(self.root)}\n{body}"

    def tool_search(self, args):
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        path = self.path(args.get("path", "."))

        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
                check=False,
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"

        matches = []
        files = [path] if path.is_file() else [
            item for item in path.rglob("*")
            if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(self.root).parts)
        ]
        for file_path in files:
            for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path.relative_to(self.root)}:{number}:{line}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def tool_run_shell(self, args):
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        result = subprocess.run(
            command,
            check=False,
            cwd=self.root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return "\n".join(
            [
                f"exit_code: {result.returncode}",
                "stdout:",
                result.stdout.strip() or "(empty)",
                "stderr:",
                result.stderr.strip() or "(empty)",
            ]
        )

    def tool_write_file(self, args):
        path = self.path(args["path"])
        content = str(args["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(self.root)} ({len(content)} chars)"

    def tool_patch_file(self, args):
        path = self.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
        return f"patched {path.relative_to(self.root)}"

    ###################################################
    #### 7) Agent Skills ##############################
    ###################################################
    def log_skill(self, message):
        if self.skill_log:
            self.skill_log(message)

    def record_skill_event(self, prompt, similarity_check, executed_skill=None, trigger="auto", bundled_file=None):
        """Log prompt, similarity check, and executed skill to session state and skill_logs.jsonl."""
        skill_entry = None
        if executed_skill is not None:
            skill_entry = {
                "name": executed_skill.name,
                "path": f"{executed_skill.display_folder(self.root)}/{executed_skill.path.name}",
                "bundled_file": bundled_file or None,
            }
        event = {
            "timestamp": now(),
            "session_id": self.session["id"],
            "trigger": trigger,
            "prompt": prompt,
            "similarity_check": similarity_check,
            "executed_skill": skill_entry,
        }
        self.session.setdefault("skill_logs", []).append(event)
        self.session_path = self.session_store.save(self.session)
        self.skill_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.skill_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        return event

    def activate_skill(self, name):
        """Mark a skill active so its full instructions appear in every following prompt."""
        skill = self.skills.find(name)
        if skill is None:
            return None
        self.remember(self.session["active_skills"], skill.name, MAX_ACTIVE_SKILLS)  # most recent wins
        self.session_path = self.session_store.save(self.session)
        return skill

    def auto_activate_skills(self, user_message):
        """Trigger A: evaluate prompt similarity against all skills and activate the best match."""
        if not self.skills:
            return None
        skill, report = self.skills.evaluate(user_message, exclude=self.session["active_skills"])
        self.record_skill_event(
            prompt=user_message,
            similarity_check=report,
            executed_skill=skill,
            trigger="auto",
        )
        if skill is None:
            return None
        self.activate_skill(skill.name)
        self.log_skill(f"activated '{skill.name}' (matched the request)")
        self.record(
            {
                "role": "tool",
                "name": "use_skill",
                "args": {"skill": skill.name},
                "content": f"auto-activated skill '{skill.name}' because the request matches its description.",
                "created_at": now(),
            }
        )
        return skill

    def tool_use_skill(self, args):
        """Trigger B: the model asks for a skill (or one of its bundled files) while working."""
        skill = self.skills.find(args["skill"])
        if skill is None:
            raise ValueError(f"unknown skill '{args['skill']}'; available: {', '.join(self.skills.names())}")
        prompt = self.last_user_prompt or self.session["memory"].get("task", "")
        _, report = self.skills.evaluate(prompt)
        bundled = str(args.get("file", "")).strip()
        if bundled:
            content = skill.read_bundled(bundled)
            self.record_skill_event(
                prompt=prompt,
                similarity_check=report,
                executed_skill=skill,
                trigger="model_tool_bundled",
                bundled_file=bundled,
            )
            return content
        already = skill.name in self.session["active_skills"]
        self.activate_skill(skill.name)
        self.record_skill_event(
            prompt=prompt,
            similarity_check=report,
            executed_skill=skill,
            trigger="model_tool",
        )
        if not already:
            self.log_skill(f"activated '{skill.name}' (requested by the model)")
        if already:
            return (
                f"skill '{skill.name}' is already active. Do not call use_skill for it again: "
                "carry out the next step of its instructions under 'Active skills' with your other tools."
            )
        return (
            f"skill '{skill.name}' is now active. "
            "Its full instructions are shown under 'Active skills' in the prompt; follow them for this task."
        )

    def active_skills_text(self):
        # Rendered into the prompt on every step instead of the transcript, so context
        # reduction (clip / history_text) can never truncate the instructions.
        blocks = []
        for name in self.session["active_skills"]:
            skill = self.skills.skills.get(name)
            if skill is not None:
                blocks.append(skill.render(self.root))
        if not blocks:
            return ""
        return "Active skills (follow these instructions for the current task):\n" + "\n\n".join(blocks)

    def skills_listing(self):
        if not self.skills:
            return "No skills found. Add one as skills/<name>/SKILL.md (frontmatter with name and description)."
        active = set(self.session["active_skills"])
        lines = [f"Skills ({len(self.skills)} found):"]
        for skill in self.skills:
            flag = "*" if skill.name in active else " "
            lines.append(f"{flag} {skill.name}: {middle(skill.description, 90)}  [{skill.display_folder(self.root)}]")
        lines.append("(* = active; use /skill <name> [request] to activate one)")
        return "\n".join(lines)

    def skill_logs_text(self, limit=10):
        logs = self.session.get("skill_logs", [])
        if not logs:
            return f"No skill events logged yet. (log file: {self.skill_log_path})"
        lines = [f"Skill log ({len(logs)} events, file: {self.skill_log_path}):"]
        for entry in logs[-limit:]:
            executed = entry["executed_skill"]["name"] if entry["executed_skill"] else "none"
            candidates = entry["similarity_check"].get("candidates", [])
            top = (
                f"{candidates[0]['skill']}={candidates[0]['score']} (cos={candidates[0]['cosine_similarity']})"
                if candidates
                else "none"
            )
            lines.append(
                f"- [{entry['trigger']}] prompt={middle(entry['prompt'], 48)!r} | top={top} | executed={executed}"
            )
        return "\n".join(lines)

    ###################################################
    #### 6) Delegation And Bounded Subagents ##########
    ###################################################
    def list_subagents(self):
        """Map normalized sub-agent name -> <name>.md found in .claude/agents/ near the workspace or any skill."""
        candidate_dirs = [
            Path(self.workspace.cwd) / ".claude" / "agents",
            self.root / ".claude" / "agents",
        ]
        for skill in self.skills:
            # If skills live in <repo>/.claude/skills/<name>, look in <repo>/.claude/agents
            candidate_dirs.append(skill.folder.parent.parent / "agents")
            candidate_dirs.append(skill.folder.parent.parent / ".claude" / "agents")
        agents, seen = {}, set()
        for directory in candidate_dirs:
            try:
                directory = directory.resolve()
                if directory in seen or not directory.is_dir():
                    continue
                seen.add(directory)
                for item in sorted(directory.iterdir()):
                    if item.is_file() and item.suffix.lower() == ".md":
                        agents.setdefault(normalize_skill_name(item.stem), item)  # first location wins
            except OSError:
                continue
        return agents

    def find_subagent_file(self, agent_name):
        target = normalize_skill_name(str(agent_name).removesuffix(".md"))
        return self.list_subagents().get(target) if target else None

    def subagents_listing(self):
        agents = self.list_subagents()
        if not agents:
            return "No sub-agents found. Add one as .claude/agents/<name>.md."
        lines = [f"Sub-agents ({len(agents)} found):"]
        lines += [f"  {name}  [{path}]" for name, path in agents.items()]
        lines.append("(run one directly with /agent <name> [task])")
        return "\n".join(lines)

    def skill_subagents(self, skill):
        """The sub-agents a skill is built around: agent files whose name its instructions mention."""
        spaced = re.sub(r"[^a-z0-9]+", " ", skill.body.lower())
        found = []
        for key in self.list_subagents():
            phrase = re.sub(r"[^a-z0-9]+", " ", key)
            match = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", spaced)
            if match:
                found.append((match.start(), key))
        return [key for _, key in sorted(found)]  # in the order the skill mentions them

    def focus_skill(self):
        """The most recently activated skill: the workflow this request is meant to finish."""
        for name in reversed(self.session["active_skills"]):
            skill = self.skills.skills.get(name)
            if skill is not None:
                return skill
        return None

    def pending_subagents(self):
        if self.depth > 0 or "delegate" not in self.tools:
            return []
        skill = self.focus_skill()
        if skill is None:
            return []
        return [name for name in self.skill_subagents(skill) if self.subagent_status.get(name) != "done"]

    @staticmethod
    def next_call_hint(pending):
        call = json.dumps({"name": "delegate", "args": {"agent": pending[0], "task": DEFAULT_SUBAGENT_TASK}})
        return f"Next, call:\n<tool>{call}</tool>"

    def completion_notice(self, pending):
        skill = self.focus_skill()
        return (
            f"Runtime notice: that final answer is too early. Skill '{skill.name}' still has sub-agents "
            f"that have not finished: {', '.join(pending)}. Do not report completion yet. "
            f"{self.next_call_hint(pending)}"
        )

    def subagent_budget(self):
        # Named sub-agents are workflow stages, so the user's --max-steps decides their budget.
        # (The model's own max_steps is ignored: it copies the example value and starves the stage.)
        return max(self.max_steps, NAMED_SUBAGENT_STEPS)

    def workspace_snapshot(self, limit=3000):
        snapshot = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [name for name in dirnames if name not in IGNORED_PATH_NAMES]
            for filename in filenames:
                path = Path(dirpath) / filename
                try:
                    stat = path.stat()
                except OSError:
                    continue
                snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
                if len(snapshot) >= limit:
                    return snapshot
        return snapshot

    @staticmethod
    def child_trace(child, limit=3):
        history = child.session["history"]
        events = [item for item in history if item["role"] == "tool"][-limit:]
        lines = [
            f"- {item['name']} {clip(json.dumps(item['args'], sort_keys=True), 120)} -> "
            f"{clip(str(item['content']).replace(chr(10), ' '), 160)}"
            for item in events
        ] or ["- (no tool calls were made)"]
        rejected = [item for item in history if "raw" in item]
        if rejected:
            lines.append(f"- last rejected model reply: {clip(rejected[-1]['raw'].replace(chr(10), ' '), 160)}")
        return "\n".join(lines)

    def tool_delegate(self, args):
        if self.depth >= self.max_depth:
            raise ValueError("delegate depth exceeded")
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        agent_name = str(args.get("agent", "")).strip()
        if agent_name:
            agent_file = self.find_subagent_file(agent_name)
            if agent_file is None:
                raise ValueError(f"unknown sub-agent '{agent_name}'")
            _, body = parse_frontmatter(agent_file.read_text(encoding="utf-8", errors="replace"))
            child = MiniAgent(
                model_client=self.model_client,
                workspace=self.workspace,
                session_store=self.session_store,
                approval_policy=self.approval_policy,
                max_steps=self.subagent_budget(),
                max_new_tokens=self.max_new_tokens,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                read_only=self.read_only,
                skills=SkillRegistry(),  # isolated context: no parent skill or transcript leakage
            )
            child.prefix = f"Sub-agent instructions ({agent_file.stem}):\n{body}\n\n" + child.prefix
            child.session["memory"]["task"] = task
            before = self.workspace_snapshot()
            answer = child.ask(task)
            changed = self.workspace_snapshot() != before
            # A sub-agent whose instructions say it writes code has not finished if nothing was written,
            # however confidently it reports success (small models do this). Checking agents may change nothing.
            writes_code = re.search(r"\b(create|write|implement|generate|save|produce|build)\b", body, re.IGNORECASE) is not None
            stopped = answer.startswith("Stopped after")
            unwritten = writes_code and not changed
            self.subagent_status[normalize_skill_name(agent_file.stem)] = "failed" if stopped or unwritten else "done"
            text = f"delegate_result ({agent_file.stem}):\n{answer}"
            if stopped:
                text += "\nThis sub-agent did NOT finish. Its last steps were:\n" + self.child_trace(child)
            elif unwritten:
                text += (
                    "\nThis sub-agent did NOT finish: it is meant to write code, but no file in the workspace "
                    "was created or changed. Its last steps were:\n" + self.child_trace(child)
                )
            elif not changed:
                text += "\nWarning: no file in the workspace was created or changed by this sub-agent."
            return text

        child = MiniAgent(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            approval_policy="never",
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
            skills=self.skills,
        )
        child.session["active_skills"] = list(self.session["active_skills"])
        child.session["memory"]["task"] = task
        child.session["memory"]["notes"] = [clip(self.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)


def build_welcome(agent, model, host):
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center("MINI CODING AGENT"),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])


def build_agent(args):
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(Path(workspace.repo_root) / ".mini-coding-agent" / "sessions")
    model = OllamaModelClient(
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.ollama_timeout,
        num_ctx=getattr(args, "num_ctx", None),
    )
    skills_dir = getattr(args, "skills_dir", None)
    if skills_dir:
        skills_dir = Path(skills_dir)
        skills_dir = skills_dir if skills_dir.is_absolute() else Path(workspace.cwd) / skills_dir
    skills = SkillRegistry.discover([Path(workspace.cwd), Path(workspace.repo_root), AGENT_HOME], skills_dir=skills_dir)
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return MiniAgent.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            skills=skills,
        )
    return MiniAgent(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        skills=skills,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for Ollama models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama model name.")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama server URL.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument(
        "--approval",
        choices=("ask", "auto", "never"),
        default="ask",
        help="Approval policy for risky tools; auto grants the model arbitrary command execution and file writes.",
    )
    parser.add_argument(
        "--skills-dir",
        default=None,
        help="Extra folder of skills (each <name>/SKILL.md). Searched before ./skills and the agent's own skills folder.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help="Context window (tokens) requested from Ollama; 0 keeps Ollama's default. Raise it if a stage fails with 'prompt is longer than the context length'.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    return parser


def run_skill_command(agent, user_input):
    """Handle /skills, /skill-logs, /agents, /agent <name> [task], /skill <name> [request], /<skill-name> [request].

    Returns (handled, request). When a request is given it is then sent to the model
    with the skill already active.
    """
    parts = user_input.split(None, 2)
    command = parts[0]
    if command == "/skills":
        print(agent.skills_listing())
        return True, None
    if command == "/skill-logs":
        print(agent.skill_logs_text())
        return True, None
    if command == "/agents":
        print(agent.subagents_listing())
        return True, None
    if command == "/agent":
        if len(parts) < 2:
            print("usage: /agent <name> [task]")
            return True, None
        name, task = parts[1], (parts[2] if len(parts) > 2 else DEFAULT_SUBAGENT_TASK)
        print(f"running sub-agent: {name}")
        try:
            print(agent.tool_delegate({"agent": name, "task": task}))
        except ValueError as exc:
            print(f"{exc}. available: {', '.join(agent.list_subagents()) or 'none'}")
        except RuntimeError as exc:  # model backend failure: show the message, not a traceback
            print(str(exc), file=sys.stderr)
        return True, None
    if command == "/skill":
        if len(parts) < 2:
            print("usage: /skill <name> [request]")
            return True, None
        name, request = parts[1], (parts[2] if len(parts) > 2 else "")
    elif command.startswith("/") and agent.skills.find(command[1:]) is not None:
        name, request = command[1:], user_input[len(command):].strip()
    else:
        return False, None
    skill = agent.activate_skill(name)
    if skill is None:
        print(f"unknown skill '{name}'. available: {', '.join(agent.skills.names()) or 'none'}")
        return True, None
    _, report = agent.skills.evaluate(request or user_input)
    agent.record_skill_event(
        prompt=request or user_input,
        similarity_check=report,
        executed_skill=skill,
        trigger="slash_command",
    )
    print(f"skill activated: {skill.name}")
    return True, request


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)
    agent.skill_log = lambda message: print(f"[skill] {message}")

    print(build_welcome(agent, model=args.model, host=args.host))

    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if prompt.startswith("/"):
            handled, request = run_skill_command(agent, prompt)
            if handled:
                if not request:
                    return 0
                prompt = request
        if prompt:
            print()
            try:
                print(agent.ask(prompt))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    while True:
        try:
            user_input = input("\nmini-coding-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue
        if user_input.startswith("/"):
            handled, request = run_skill_command(agent, user_input)
            if handled:
                if not request:
                    continue
                user_input = request

        print()
        try:
            print(agent.ask(user_input))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
