import functools
import http.server
import threading

import pytest

import imglab as L


class FakePopen:
    """Stand-in for subprocess.Popen that replays scripted pip results."""
    script = []
    commands = []

    def __init__(self, cmd, **kw):
        FakePopen.commands.append(cmd)
        self.rc, text = FakePopen.script.pop(0)
        self.stdout = iter(text.splitlines(True))

    def wait(self):
        return self.rc


@pytest.fixture
def fake_pip(monkeypatch):
    FakePopen.script, FakePopen.commands = [], []
    monkeypatch.setattr(L.subprocess, "Popen", FakePopen)
    return FakePopen


@pytest.mark.parametrize("verb", ["install", "uninstall"])
def test_pip_retries_on_externally_managed(fake_pip, verb):
    fake_pip.script = [(1, "error: externally-managed-environment\n"), (0, "ok\n")]
    logs = []
    assert L.run_pip([verb, "pkg"], logs.append) is True
    assert "--break-system-packages" not in fake_pip.commands[0]
    assert "--break-system-packages" in fake_pip.commands[1]
    assert any("externally managed" in line for line in logs)


def test_pip_failure_is_reported(fake_pip):
    fake_pip.script = [(1, "some other error\n")]
    assert L.run_pip(["install", "pkg"], lambda t: None) is False
    assert len(fake_pip.commands) == 1


@pytest.fixture
def model_server(tmp_path, monkeypatch):
    served = tmp_path / "served"
    served.mkdir()
    (served / "EDSR_x4.pb").write_bytes(b"x" * 200_000)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    models = tmp_path / "models"
    monkeypatch.setattr(L, "MODELS_DIR", str(models))
    monkeypatch.setitem(L.MODEL_URLS, "EDSR_x4.pb", f"{base}/EDSR_x4.pb")
    monkeypatch.setitem(L.MODEL_URLS, "EDSR_x2.pb", f"{base}/missing_x2.pb")
    monkeypatch.setitem(L.MODEL_URLS, "EDSR_x3.pb", f"{base}/missing_x3.pb")
    yield models
    srv.shutdown()


def test_model_download(model_server):
    assert L.COMPONENTS["model_edsr_x4"]["check"]() == "missing"
    ok, msg = L.install_component("model_edsr_x4", lambda t: None)
    assert ok, msg
    assert (model_server / "EDSR_x4.pb").stat().st_size == 200_000
    assert L.COMPONENTS["model_edsr_x4"]["check"]() == "ok"
    assert not list(model_server.glob("*.part"))


def test_model_download_failure_explains_what_to_do(model_server):
    ok, msg = L.install_component("model_edsr_x2", lambda t: None)
    assert not ok
    assert "missing_x2.pb" in msg and str(model_server) in msg
    assert not list(model_server.glob("*.part"))


def test_frozen_build_never_runs_pip(monkeypatch):
    monkeypatch.setattr(L, "FROZEN", True)
    monkeypatch.setattr(L, "run_pip", lambda *a, **k: pytest.fail("pip must not run in a frozen build"))
    ok, msg = L.install_component("rapidocr", lambda t: None)
    assert not ok and "standalone" in msg


def test_component_checks_return_known_states():
    for key, c in L.COMPONENTS.items():
        assert c["check"]() in ("ok", "missing", "restart"), key
