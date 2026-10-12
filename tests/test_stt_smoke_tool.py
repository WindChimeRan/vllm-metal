# SPDX-License-Identifier: Apache-2.0

import io
import json
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tools import stt_sampling_serve_smoke as smoke


@pytest.fixture
def healthy_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = (
                json.dumps({"data": [{"id": "fixture-model"}]}).encode()
                if self.path == "/v1/models"
                else b""
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_main_rejects_occupied_port(healthy_server, monkeypatch, capsys):
    monkeypatch.setattr(
        smoke.sys,
        "argv",
        [
            "stt-smoke",
            "--audio",
            "unused.wav",
            "--port",
            str(healthy_server.server_port),
        ],
    )

    def unexpected_start(*args, **kwargs):
        pytest.fail(
            "The smoke script must reject an occupied port before starting vLLM"
        )

    monkeypatch.setattr(smoke.subprocess, "Popen", unexpected_start)
    assert smoke.main() == 1
    assert "cannot bind localhost port" in capsys.readouterr().err


@pytest.mark.parametrize("exit_codes", [(7,), (None, 7)])
def test_health_response_cannot_hide_server_exit(healthy_server, exit_codes):
    codes = iter(exit_codes)
    serve = SimpleNamespace(poll=lambda: next(codes))
    base_url = f"http://127.0.0.1:{healthy_server.server_port}"
    assert not smoke._wait_for_health(base_url, 2, serve, "fixture-model")


def test_health_wait_accepts_live_server(healthy_server):
    serve = SimpleNamespace(poll=lambda: None)
    base_url = f"http://127.0.0.1:{healthy_server.server_port}"
    assert smoke._wait_for_health(base_url, 2, serve, "fixture-model")


def test_old_health_with_live_child_is_not_our_server(healthy_server):
    serve = SimpleNamespace(poll=lambda: None)
    base_url = f"http://127.0.0.1:{healthy_server.server_port}"
    assert not smoke._wait_for_health(base_url, 0.05, serve, "this-run")


def test_health_request_uses_remaining_timeout(monkeypatch):
    seen = []

    def unavailable(*args, timeout):
        seen.append(timeout)
        raise urllib.error.URLError("unavailable")

    monkeypatch.setattr(smoke.urllib.request, "urlopen", unavailable)
    serve = SimpleNamespace(poll=lambda: None)
    assert not smoke._wait_for_health("http://127.0.0.1:1", 0.05, serve, "this-run")
    assert seen and all(0 < timeout <= 0.05 for timeout in seen)


@pytest.mark.parametrize("data", [None, 7, "invalid"])
def test_malformed_model_list_does_not_crash(healthy_server, monkeypatch, data):
    original_urlopen = smoke.urllib.request.urlopen

    def malformed_models(url, *, timeout):
        if url.endswith("/v1/models"):
            return io.BytesIO(json.dumps({"data": data}).encode())
        return original_urlopen(url, timeout=timeout)

    monkeypatch.setattr(smoke.urllib.request, "urlopen", malformed_models)
    serve = SimpleNamespace(poll=lambda: None)
    base_url = f"http://127.0.0.1:{healthy_server.server_port}"
    assert not smoke._wait_for_health(base_url, 0.05, serve, "this-run")


def test_transcriptions_use_the_served_model(monkeypatch):
    requested_models = []

    def transcribe(_base_url, _audio, **form):
        requested_models.append(form["model"])
        return (200, "same transcript")

    monkeypatch.setattr(smoke, "_transcribe", transcribe)
    assert smoke._run_checks(
        "http://localhost", "speech.wav", "silence.wav", "this-run"
    )
    assert requested_models == ["this-run"] * 9


def test_unrelated_startup_failure_does_not_retry(monkeypatch, capsys):
    starts = []

    def failed_vllm(command):
        starts.append(command)
        return SimpleNamespace(poll=lambda: 7, wait=lambda timeout: 7)

    monkeypatch.setattr(smoke.subprocess, "Popen", failed_vllm)
    monkeypatch.setattr(sys, "argv", ["stt-smoke", "--audio", "unused.wav"])
    assert smoke.main() == 1
    assert len(starts) == 1
    assert "server exited during startup (exit code 7)" in capsys.readouterr().err


def test_auto_port_retries_claimed_port(monkeypatch, capsys):
    ports = iter([8100, 8200])
    starts = []
    checked_urls = []

    def probe_port(requested_port):
        if requested_port == 0:
            return next(ports)
        assert requested_port == 8100
        raise OSError("port claimed")

    class Child:
        def __init__(self, exit_code):
            self.exit_code = exit_code

        def poll(self):
            return self.exit_code

        def terminate(self):
            self.exit_code = 0

        def wait(self, timeout):
            return self.exit_code

    def fake_vllm(command):
        port = int(command[command.index("--port") + 1])
        starts.append(port)
        return Child(7 if port == 8100 else None)

    def record_checks(base_url, *_):
        checked_urls.append(base_url)
        return True

    monkeypatch.setattr(smoke, "_probe_port", probe_port)
    monkeypatch.setattr(smoke.subprocess, "Popen", fake_vllm)
    monkeypatch.setattr(
        smoke,
        "_wait_for_health",
        lambda url, timeout, child, model: child.poll() is None,
    )
    monkeypatch.setattr(smoke, "_run_checks", record_checks)
    monkeypatch.setattr(sys, "argv", ["stt-smoke", "--audio", "unused.wav"])
    assert smoke.main() == 0
    assert starts == [8100, 8200]
    assert checked_urls == ["http://127.0.0.1:8200"]
    assert "retrying on a new port" in capsys.readouterr().out
