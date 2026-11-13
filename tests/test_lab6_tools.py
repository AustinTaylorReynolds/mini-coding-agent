import os
import subprocess
from unittest.mock import patch

from mini_coding_agent import FakeModelClient, MiniAgent, SessionStore, WorkspaceContext

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def git(path, *args):
    return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, check=True).stdout.strip()


def build_repo_agent(tmp_path, outputs=()):
    git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    git(tmp_path, "add", "README.md")
    with patch.dict(os.environ, GIT_ENV):
        git(tmp_path, "commit", "-q", "-m", "init")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    return MiniAgent(
        model_client=FakeModelClient(list(outputs)),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )


def fake_run_cmd(origin_url, login, calls, branch="fix/pow-overflow"):
    """Stands in for MiniAgent.run_cmd so no real network or gh call happens."""

    def fake(self, argv, timeout=60):
        calls.append(argv)
        if argv[:3] == ["git", "remote", "get-url"]:
            return origin_url
        if argv[:2] == ["gh", "api"]:
            return login
        if argv[:3] == ["git", "branch", "--show-current"]:
            return branch
        return "ok"

    return fake


def test_new_tools_are_listed_in_the_prompt(tmp_path):
    agent = build_repo_agent(tmp_path)

    for name in ("fork_repo", "create_branch", "commit_changes", "submit_pr"):
        assert f"- {name}(" in agent.prefix
    assert "Never push to or open a pull request against the upstream project" in agent.prefix


def test_contributing_md_is_loaded_into_the_prompt(tmp_path):
    (tmp_path / "CONTRIBUTING.md").write_text("Please fork and open a pull request.\n", encoding="utf-8")

    workspace = WorkspaceContext.build(tmp_path)

    assert "CONTRIBUTING.md" in workspace.project_docs
    assert "fork and open a pull request" in workspace.text()


def test_create_branch_switches_to_new_branch(tmp_path):
    agent = build_repo_agent(tmp_path)

    result = agent.run_tool("create_branch", {"name": "fix/pow-overflow"})

    assert not result.startswith("error"), result
    assert git(tmp_path, "branch", "--show-current") == "fix/pow-overflow"


def test_create_branch_rejects_unsafe_names(tmp_path):
    agent = build_repo_agent(tmp_path)

    for bad in ("--force", "has space", "a..b", ""):
        assert agent.run_tool("create_branch", {"name": bad}).startswith("error: invalid arguments")


def test_commit_changes_stages_only_listed_files(tmp_path):
    agent = build_repo_agent(tmp_path)
    (tmp_path / "Pow.java").write_text("class Pow {}\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("scratch\n", encoding="utf-8")

    with patch.dict(os.environ, GIT_ENV):
        result = agent.run_tool("commit_changes", {"message": "fix: pow", "paths": ["Pow.java"]})

    assert not result.startswith("error"), result
    assert "Pow.java" in git(tmp_path, "show", "--name-only", "--format=", "HEAD")
    assert "notes.txt" in git(tmp_path, "status", "--short")


def test_commit_changes_refuses_agent_internals(tmp_path):
    agent = build_repo_agent(tmp_path)

    result = agent.run_tool(
        "commit_changes",
        {"message": "oops", "paths": [".mini-coding-agent/sessions/x.json"]},
    )

    assert result.startswith("error: invalid arguments")
    assert "refusing to commit" in result


def test_path_list_accepts_lists_and_comma_separated_strings():
    assert MiniAgent.path_list(["a.java", "b.java"]) == ["a.java", "b.java"]
    assert MiniAgent.path_list("a.java, b.java") == ["a.java", "b.java"]


def test_repo_slug_handles_https_and_ssh_urls():
    assert MiniAgent.repo_slug("https://github.com/me/Java.git") == "me/Java"
    assert MiniAgent.repo_slug("git@github.com:me/Java.git") == "me/Java"
    assert MiniAgent.repo_slug("https://github.com/me/Java") == "me/Java"


def test_submit_pr_refuses_to_touch_upstream(tmp_path):
    agent = build_repo_agent(tmp_path)
    calls = []

    with patch.object(MiniAgent, "run_cmd", new=fake_run_cmd("https://github.com/TheAlgorithms/Java.git", "student", calls)):
        result = agent.run_tool("submit_pr", {"title": "Fix", "body": "why"})

    assert "never contribute upstream" in result
    assert not any(argv[:2] in (["git", "push"], ["gh", "pr"]) for argv in calls)


def test_submit_pr_pushes_and_opens_pr_on_own_fork(tmp_path):
    agent = build_repo_agent(tmp_path)
    calls = []

    with patch.object(MiniAgent, "run_cmd", new=fake_run_cmd("git@github.com:student/Java.git", "student", calls)):
        result = agent.run_tool("submit_pr", {"title": "Fix overflow", "body": "why"})

    assert "student/Java" in result
    assert ["git", "push", "-u", "origin", "fix/pow-overflow"] in calls
    pr = next(argv for argv in calls if argv[:3] == ["gh", "pr", "create"])
    assert pr[pr.index("--repo") + 1] == "student/Java"
    assert pr[pr.index("--base") + 1] == agent.workspace.default_branch


def test_submit_pr_refuses_to_run_from_the_base_branch(tmp_path):
    agent = build_repo_agent(tmp_path)
    calls = []
    base = agent.workspace.default_branch

    with patch.object(MiniAgent, "run_cmd", new=fake_run_cmd("git@github.com:student/Java.git", "student", calls, branch=base)):
        result = agent.run_tool("submit_pr", {"title": "Fix", "body": "why"})

    assert "use create_branch first" in result
    assert not any(argv[:2] == ["git", "push"] for argv in calls)


def test_fork_repo_skips_when_origin_is_already_your_fork(tmp_path):
    agent = build_repo_agent(tmp_path)
    calls = []

    with patch.object(MiniAgent, "run_cmd", new=fake_run_cmd("https://github.com/student/Java.git", "student", calls)):
        result = agent.run_tool("fork_repo", {})

    assert "already a fork" in result
    assert not any(argv[:3] == ["gh", "repo", "fork"] for argv in calls)


def test_fork_repo_forks_when_origin_is_the_upstream_project(tmp_path):
    agent = build_repo_agent(tmp_path)
    calls = []

    with patch.object(MiniAgent, "run_cmd", new=fake_run_cmd("https://github.com/TheAlgorithms/Java.git", "student", calls)):
        result = agent.run_tool("fork_repo", {})

    assert not result.startswith("error"), result
    assert ["gh", "repo", "fork", "--remote"] in calls


def test_model_can_drive_a_new_tool_end_to_end(tmp_path):
    agent = build_repo_agent(
        tmp_path,
        outputs=[
            '<tool>{"name":"create_branch","args":{"name":"fix/demo"}}</tool>',
            "<final>Created the branch.</final>",
        ],
    )

    answer = agent.ask("Start a branch for the fix")

    assert answer == "Created the branch."
    assert git(tmp_path, "branch", "--show-current") == "fix/demo"
