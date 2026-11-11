"""The public read-only request API: ``agent.lsp.request`` / ``LSPService.request_sync``.

A plugin sends one read-only LSP request for a file and gets the server's raw result.  ``None`` means
the server has no answer; :class:`agent.lsp.LSPUnavailable` means no server can answer.  The mock
server's ``query`` mode echoes each request with its copy of the document, so the tests read what
the server received and the text it answered on.
"""
from __future__ import annotations

import asyncio
import sys
import time

import pytest

import agent.lsp
from agent.lsp import LSPRequestError, LSPUnavailable
from agent.lsp.client import LSPClient, file_uri
from agent.lsp.manager import READ_ONLY_METHODS, LSPService
from agent.lsp.protocol import ERROR_CONTENT_MODIFIED
from agent.lsp.servers import SERVERS, ServerDef, SpawnSpec
from tests.agent.lsp.test_service import MOCK_SERVER, _install_mock_server

_HOVER = {"position": {"line": 0, "character": 0}}


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    """``install(server_id=..., **service_kwargs)`` → ``(repo, svc, spawn_count)``: a trusted git
    workspace (the cwd) whose ``server_id`` server is the ``query`` mock."""
    created = []

    def install(server_id="pyright", **kwargs):
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
        (repo / ".git").mkdir(exist_ok=True)
        (repo / "pyproject.toml").write_text("", encoding="utf-8")
        monkeypatch.chdir(str(repo))
        gen = _install_mock_server(monkeypatch, "query", server_id)
        spawn_count = next(gen)
        kwargs.setdefault("enabled", True)
        kwargs.setdefault("wait_timeout", 5.0)
        svc = LSPService(wait_mode="document", install_strategy="manual", **kwargs)
        created.append((svc, gen))
        return repo, svc, spawn_count

    yield install
    for svc, gen in created:
        svc.shutdown()
        try:
            next(gen)
        except StopIteration:
            pass


def _source(repo, name="a.py", text="x = 1\n"):
    path = repo / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_hover_gets_the_raw_result_with_the_document_filled_in(workspace):
    repo, svc, _ = workspace()
    path = _source(repo)
    params = dict(_HOVER)

    result = svc.request_sync(path, "textDocument/hover", params)

    assert result["method"] == "textDocument/hover"
    assert result["params"] == {"position": {"line": 0, "character": 0}, "textDocument": {"uri": file_uri(path)}}
    assert result["text"] == "x = 1\n", "the file must be synced before the request"
    assert params == _HOVER, "the caller's params dict must not change"


def test_a_textdocument_the_caller_gives_is_sent_unchanged(workspace):
    repo, svc, _ = workspace()
    path = _source(repo)
    other = _source(repo, "b.py", "y = 2\n")
    params = {"textDocument": {"uri": file_uri(other)}, **_HOVER}

    result = svc.request_sync(path, "textDocument/definition", params)

    assert result["params"]["textDocument"] == {"uri": file_uri(other)}


def test_workspace_symbol_opens_the_file_but_adds_no_textdocument(workspace):
    """typescript-language-server answers ``workspace/symbol`` with "No Project." until a file is open."""
    repo, svc, _ = workspace()
    path = _source(repo)

    result = svc.request_sync(path, "workspace/symbol", {"query": "x"})

    assert result["params"] == {"query": "x"}
    assert result["open"] == [file_uri(path)]


def test_the_server_sees_the_current_disk_content(workspace):
    repo, svc, _ = workspace()
    path = _source(repo)
    assert svc.request_sync(path, "textDocument/hover", _HOVER)["text"] == "x = 1\n"

    _source(repo, text="x = 2\n")

    assert svc.request_sync(path, "textDocument/hover", _HOVER)["text"] == "x = 2\n"


def test_no_resync_while_an_edit_check_diagnoses_its_snapshot(workspace):
    """An edit check syncs the edit's own snapshots and waits for the verdict on them; a re-sync of the
    disk content in between would answer the check's baseline wait with the post-edit diagnostics."""
    from agent.lsp.edit_check import _Inflight

    repo, svc, _ = workspace()
    path = _source(repo)
    svc.request_sync(path, "textDocument/hover", _HOVER)
    _source(repo, text="x = 2\n")

    async def _hold_check():
        release = asyncio.Event()
        task = asyncio.ensure_future(release.wait())
        svc._edit_checks.tail[path] = _Inflight(task, "x = 1\n", ())
        return release

    release = svc._loop.run(_hold_check(), timeout=2.0)
    try:
        assert svc.request_sync(path, "textDocument/hover", _HOVER)["text"] == "x = 1\n"
    finally:
        svc._loop._loop.call_soon_threadsafe(release.set)


def test_a_null_result_is_none(workspace):
    repo, svc, _ = workspace()

    assert svc.request_sync(_source(repo), "textDocument/hover", {**_HOVER, "mock": "null"}) is None


def test_the_module_function_uses_the_active_service(workspace, monkeypatch):
    repo, svc, _ = workspace()
    monkeypatch.setattr(agent.lsp, "get_service", lambda: svc)

    result = agent.lsp.request(_source(repo), "textDocument/documentSymbol")

    assert result["params"] == {"textDocument": {"uri": file_uri(str(repo / "a.py"))}}


@pytest.mark.parametrize("method", ["textDocument/rename", "textDocument/codeAction", "workspace/executeCommand",
                                    "textDocument/didOpen", "shutdown"])
def test_a_method_outside_the_allowlist_raises_before_any_server_starts(workspace, monkeypatch, method):
    repo, svc, spawn_count = workspace()
    path = _source(repo)
    monkeypatch.setattr(agent.lsp, "get_service", lambda: None)

    with pytest.raises(ValueError, match="read-only"):
        svc.request_sync(path, method, {})
    with pytest.raises(ValueError, match="read-only"):
        agent.lsp.request(path, method, {})
    assert spawn_count["value"] == 0


@pytest.mark.parametrize("timeout", [0, -1.0, float("nan")])
def test_a_timeout_that_is_not_positive_is_rejected(workspace, timeout):
    repo, svc, spawn_count = workspace()

    with pytest.raises(ValueError, match="timeout"):
        svc.request_sync(_source(repo), "textDocument/hover", _HOVER, timeout=timeout)
    assert spawn_count["value"] == 0


def test_the_allowlist_is_read_only_methods():
    assert READ_ONLY_METHODS == {
        "textDocument/hover", "textDocument/definition", "textDocument/declaration",
        "textDocument/typeDefinition", "textDocument/implementation", "textDocument/references",
        "textDocument/documentSymbol", "textDocument/documentHighlight", "workspace/symbol",
    }


def test_disabled_lsp_is_unavailable(workspace, monkeypatch):
    repo, svc, spawn_count = workspace(enabled=False)
    path = _source(repo)
    monkeypatch.setattr(agent.lsp, "get_service", lambda: None)

    with pytest.raises(LSPUnavailable, match="disabled") as svc_err:
        svc.request_sync(path, "textDocument/hover", _HOVER)
    with pytest.raises(LSPUnavailable, match="disabled") as module_err:
        agent.lsp.request(path, "textDocument/hover", _HOVER)
    assert svc_err.value.reason and module_err.value.reason
    assert spawn_count["value"] == 0


def test_a_file_no_server_handles_is_unavailable(workspace):
    repo, svc, _ = workspace()

    with pytest.raises(LSPUnavailable, match="no language server"):
        svc.request_sync(_source(repo, "notes.unknownext"), "textDocument/hover", _HOVER)


def test_a_server_disabled_in_config_is_unavailable(workspace):
    repo, svc, spawn_count = workspace(disabled_servers=["pyright"])

    with pytest.raises(LSPUnavailable, match="disabled"):
        svc.request_sync(_source(repo), "textDocument/hover", _HOVER)
    assert spawn_count["value"] == 0


def test_a_file_outside_a_git_workspace_is_unavailable(workspace, tmp_path):
    _repo, svc, spawn_count = workspace()
    loose = tmp_path / "loose"
    loose.mkdir()

    with pytest.raises(LSPUnavailable, match="git"):
        svc.request_sync(_source(loose), "textDocument/hover", _HOVER)
    assert spawn_count["value"] == 0


def test_an_untrusted_workspace_is_unavailable(workspace, tmp_path):
    """rust-analyzer runs project build scripts, so it starts only in a trusted workspace."""
    _repo, svc, spawn_count = workspace(server_id="rust-analyzer")
    other = tmp_path / "other"
    (other / ".git").mkdir(parents=True)
    (other / "Cargo.toml").write_text("", encoding="utf-8")

    with pytest.raises(LSPUnavailable, match="trusted"):
        svc.request_sync(_source(other, "lib.rs", "fn main() {}\n"), "textDocument/hover", _HOVER)
    assert spawn_count["value"] == 0


def test_an_error_response_propagates_and_keeps_the_server(workspace):
    repo, svc, _ = workspace()
    path = _source(repo)

    with pytest.raises(LSPRequestError) as err:
        svc.request_sync(path, "textDocument/hover", {**_HOVER, "mock": "error"})

    assert err.value.code == -32602
    assert svc.get_status()["broken"] == []
    assert svc.request_sync(path, "textDocument/hover", _HOVER)["text"] == "x = 1\n"


def test_a_server_that_dies_mid_request_is_marked_broken(workspace):
    repo, svc, _ = workspace()
    path = _source(repo)

    with pytest.raises(LSPUnavailable):
        svc.request_sync(path, "textDocument/hover", {**_HOVER, "mock": "exit"})

    assert svc.get_status()["broken"], "a protocol failure must mark the server broken"
    with pytest.raises(LSPUnavailable, match="broken"):
        svc.request_sync(path, "textDocument/hover", _HOVER)


def test_no_answer_within_the_timeout_is_unavailable_but_not_broken(workspace):
    repo, svc, _ = workspace()
    path = _source(repo)

    with pytest.raises(LSPUnavailable, match="did not answer"):
        svc.request_sync(path, "textDocument/hover", {**_HOVER, "mock": "hang"}, timeout=0.5)

    assert svc.get_status()["broken"] == [], "slow is not dead"
    assert svc.request_sync(path, "textDocument/hover", _HOVER)["text"] == "x = 1\n"


def test_the_default_timeout_is_lsp_wait_timeout(workspace):
    repo, svc, _ = workspace(wait_timeout=0.5)
    path = _source(repo)
    svc.request_sync(path, "textDocument/hover", _HOVER)  # spawn outside the timed call

    started = time.monotonic()
    with pytest.raises(LSPUnavailable, match="did not answer"):
        svc.request_sync(path, "textDocument/hover", {**_HOVER, "mock": "hang"})

    assert time.monotonic() - started < 3.0


def test_a_server_a_plugin_uses_survives_the_idle_reaper(workspace):
    repo, svc, _ = workspace(idle_timeout=60.0)
    path = _source(repo)
    svc.request_sync(path, "textDocument/hover", _HOVER)
    key = next(iter(svc._clients))
    svc._last_used[key] -= 120.0

    svc.request_sync(path, "textDocument/hover", _HOVER)
    svc._loop.run(svc._reap_idle_once(), timeout=5.0)

    assert key in svc._clients


def test_an_excluded_workspace_is_unavailable(workspace, tmp_path):
    repo, svc, spawn_count = workspace(exclude_roots=[str(tmp_path / "repo")])

    with pytest.raises(LSPUnavailable, match="exclude_roots"):
        svc.request_sync(_source(repo), "textDocument/hover", _HOVER)
    assert spawn_count["value"] == 0


def test_params_that_are_not_a_dict_are_rejected(workspace):
    repo, svc, spawn_count = workspace()

    with pytest.raises(TypeError, match="params"):
        svc.request_sync(_source(repo), "workspace/symbol", ["x"])
    assert spawn_count["value"] == 0


def test_a_slow_start_with_a_short_timeout_does_not_break_the_server(workspace):
    """The server initializes in 3.5s; the agent's edit check tolerates that, so a plugin call with a
    0.5s timeout must too, or it turns the agent's own diagnostics off for the process lifetime."""
    repo, svc, _ = workspace(wait_timeout=0.5)
    index = next(i for i, s in enumerate(SERVERS) if s.server_id == "pyright")
    original = SERVERS[index]
    code = (f"import runpy, sys, time; time.sleep(3.5); sys.argv = [{MOCK_SERVER!r}]; "
            f"runpy.run_path({MOCK_SERVER!r}, run_name='__main__')")
    # Replaces the fixture's mock entry; the fixture's teardown restores the real server.
    SERVERS[index] = ServerDef(
        server_id="pyright", extensions=original.extensions, resolve_root=lambda fp, ws: ws,
        build_spawn=lambda root, ctx: SpawnSpec(command=[sys.executable, "-c", code], workspace_root=root,
                                                cwd=root, env={"MOCK_LSP_SCRIPT": "query"},
                                                initialization_options={}),
        seed_first_push=False, description="slow-starting mock pyright")
    path = _source(repo)

    assert svc.request_sync(path, "textDocument/hover", _HOVER, timeout=0.5)["text"] == "x = 1\n"
    assert svc.get_status()["broken"] == []
    assert svc.enabled_for(path)


def test_content_modified_retries_stay_inside_the_timeout(workspace, monkeypatch):
    """A busy server answering ContentModified: the retries must end within the request's own timeout
    (slow, not broken), never run into the outer limit that marks the server broken."""
    repo, svc, _ = workspace(wait_timeout=1.0)
    path = _source(repo)
    svc.request_sync(path, "textDocument/hover", _HOVER)
    real = LSPClient._send_request

    async def busy(self, method, params):
        if method == "textDocument/hover":
            # health: allow HX013 -- the delay IS the behaviour under test (a busy server answering late)
            await asyncio.sleep(0.9)
            raise LSPRequestError(ERROR_CONTENT_MODIFIED, "content modified")
        return await real(self, method, params)

    monkeypatch.setattr(LSPClient, "_send_request", busy)
    with pytest.raises(LSPUnavailable, match="did not answer"):
        svc.request_sync(path, "textDocument/hover", _HOVER)
    assert svc.get_status()["broken"] == []


def test_lsp_request_error_is_public():
    from agent.lsp.protocol import LSPRequestError as defined

    assert LSPRequestError is defined
    assert "LSPRequestError" in agent.lsp.__all__
