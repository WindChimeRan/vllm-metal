# SPDX-License-Identifier: Apache-2.0

import io
import json
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tools import gguf_serve_smoke as smoke


@pytest.fixture
def old_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = (
                json.dumps({"data": [{"id": "old-model"}]}).encode()
                if self.path == "/v1/models"
                else b""
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            self.server.completion_requests += 1
            body = json.dumps({"choices": [{"text": " Paris"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.completion_requests = 0
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


def test_explicit_occupied_port_refuses_to_start(old_server, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf-smoke",
            "unused.gguf",
            "--tokenizer",
            "unused",
            "--port",
            str(old_server.server_port),
        ],
    )

    def unexpected_start(*args, **kwargs):
        pytest.fail("The GGUF smoke must reject an occupied port before spawning")

    monkeypatch.setattr(smoke.subprocess, "Popen", unexpected_start)
    assert smoke.main() == 1
    assert "cannot bind localhost port" in capsys.readouterr().err


@pytest.mark.parametrize("exit_codes", [(7,), (None, 7)])
def test_old_health_cannot_hide_child_exit(old_server, exit_codes):
    codes = iter(exit_codes)
    child = SimpleNamespace(poll=lambda: next(codes))
    base_url = f"http://127.0.0.1:{old_server.server_port}"
    assert not smoke._wait_for_health(base_url, 2, child, "this-run")


def test_old_server_with_live_child_cannot_claim_health(old_server):
    child = SimpleNamespace(poll=lambda: None)
    base_url = f"http://127.0.0.1:{old_server.server_port}"
    assert not smoke._wait_for_health(base_url, 0.05, child, "this-run")


def test_health_request_uses_remaining_timeout(monkeypatch):
    seen = []

    def unavailable(*args, timeout):
        seen.append(timeout)
        raise urllib.error.URLError("unavailable")

    monkeypatch.setattr(smoke.urllib.request, "urlopen", unavailable)
    child = SimpleNamespace(poll=lambda: None)
    assert not smoke._wait_for_health("http://127.0.0.1:1", 0.05, child, "this-run")
    assert seen and all(0 < timeout <= 0.05 for timeout in seen)


@pytest.mark.parametrize("model_data", [[{"id": "this-run"}], None, 7, "invalid"])
def test_model_check_closes_health_and_handles_malformed_data(monkeypatch, model_data):
    health = io.BytesIO()
    health.status = 200
    requested = []

    def urlopen(url, timeout):
        requested.append(url)
        assert timeout > 0
        if url.endswith("/health"):
            return health
        assert health.closed
        return io.BytesIO(json.dumps({"data": model_data}).encode())

    monkeypatch.setattr(smoke.urllib.request, "urlopen", urlopen)
    polls = iter([None, None, None if isinstance(model_data, list) else 7])
    child = SimpleNamespace(poll=lambda: next(polls))
    assert smoke._wait_for_health("http://127.0.0.1:1", 1, child, "this-run") == (
        isinstance(model_data, list)
    )
    assert requested == ["http://127.0.0.1:1/health", "http://127.0.0.1:1/v1/models"]


def test_child_startup_failure_returns_nonzero(monkeypatch, capsys):
    starts = []

    def failed_vllm(command):
        starts.append(command)
        return SimpleNamespace(poll=lambda: 7, wait=lambda timeout: 7)

    monkeypatch.setattr(smoke.subprocess, "Popen", failed_vllm)
    monkeypatch.setattr(
        sys, "argv", ["gguf-smoke", "unused.gguf", "--tokenizer", "unused"]
    )
    assert smoke.main() == 1
    assert len(starts) == 1
    assert "server exited during startup (exit code 7)" in capsys.readouterr().err


def test_port_claimed_after_probe_cannot_report_success(
    old_server, monkeypatch, capsys
):
    polls = iter([None, 7, 7, 7])
    child = SimpleNamespace(poll=lambda: next(polls), wait=lambda timeout: 7)
    monkeypatch.setattr(smoke, "_probe_port", lambda port: old_server.server_port)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda command: child)
    monkeypatch.setattr(
        sys, "argv", ["gguf-smoke", "unused.gguf", "--tokenizer", "unused"]
    )

    assert smoke.main() == 1
    assert old_server.completion_requests == 0
    assert "server exited during startup (exit code 7)" in capsys.readouterr().err


def test_completion_uses_unique_model_name(monkeypatch):
    def urlopen(request, timeout):
        assert timeout == 120
        assert json.loads(request.data)["model"] == "this-run"
        return io.BytesIO(b'{"choices": [{"text": " Paris"}]}')

    monkeypatch.setattr(smoke.urllib.request, "urlopen", urlopen)
    assert smoke._greedy_completion("http://127.0.0.1:1", "this-run", "prompt", 4) == (
        " Paris"
    )
