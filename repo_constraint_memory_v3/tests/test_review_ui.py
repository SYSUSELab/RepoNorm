from __future__ import annotations

from contextlib import contextmanager
from http.client import HTTPConnection
from threading import Thread
from urllib.parse import urlencode, urlsplit

import re

import pytest

from repo_constraint_memory_v3.review import service, ui
from test_consumer_cli import _mock_source_run


@pytest.fixture(scope="module")
def ui_source(tmp_path_factory):
    return _mock_source_run(tmp_path_factory.mktemp("review-ui-source"), incomplete=True)


@contextmanager
def running_server(directory, repository):
    with ui.create_review_server(directory, repository_root=repository) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def request(server, method, path=None, *, fields=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    request_headers = dict(headers or {})
    body = None
    if fields is not None:
        body = urlencode(fields)
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        request_headers.setdefault("Origin", server.origin)
    connection.request(method, path or "/?" + urlsplit(server.url).query,
                       body=body, headers=request_headers)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read().decode()
    connection.close()
    return result


def test_review_page_save_resume_and_publish_without_model(ui_source, tmp_path, monkeypatch):
    source, repository, _ = ui_source
    directory = tmp_path / "review"
    session = service.create_review(source, repository_root=repository, output_directory=directory)
    claim_id = next(iter(session.claim_digests))

    def forbidden(*args, **kwargs):
        pytest.fail("human review must not call a model or execute a subprocess")

    # The non-git fixture's snapshot check needs no subprocess either.
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with running_server(directory, repository) as server:
        status, headers, body = request(server, "GET")
        assert status == 200 and "Unprocessed" in body and "partial" in body
        assert 'lang="en"' in body
        assert not re.search(r"[\u3400-\u9fff]", body)
        assert "give it to a coding agent" in body
        assert claim_id in body and "Supporting evidence, counterevidence, and existing assessment" in body
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        assert headers["Referrer-Policy"] == "strict-origin"
        status, headers, _ = request(server, "POST", "/decision", fields={
            "token": server.token, "claim_id": claim_id, "decision": "accept", "rationale": "The repository follows this convention.",
        })
        assert status == 303 and headers["Location"] == server.url
        status, _, body = request(server, "GET")
        assert status == 200 and "The repository follows this convention." in body
        assert service.load_review(directory).progress()["accepted"] == 1
        status, _, _ = request(server, "POST", "/publish", fields={"token": server.token})
        assert status == 303
        publication = server.publication
        assert (publication / "consumer/START.md").is_file()
        assert (publication / f"consumer/claims/{claim_id}.md").is_file()
        assert request(server, "GET")[2].find(str(publication / "consumer")) >= 0
    with running_server(directory, repository) as resumed:
        assert 'value="accept" required checked' in request(resumed, "GET")[2]
        status, _, _ = request(resumed, "POST", "/decision", fields={
            "token": resumed.token, "claim_id": claim_id, "decision": "reject", "rationale": "Reconsidered after reading the repository.",
        })
        assert status == 303
        assert service.load_review(directory).progress()["rejected"] == 1
        assert (publication / f"consumer/claims/{claim_id}.md").is_file()


def test_loopback_form_security_and_no_file_server(ui_source, tmp_path):
    source, repository, _ = ui_source
    directory = tmp_path / "review"
    service.create_review(source, repository_root=repository, output_directory=directory)
    with running_server(directory, repository) as server:
        assert server.server_address[0] == "127.0.0.1"
        assert request(server, "GET", "/")[0] == 403
        assert request(server, "GET", "/?token=wrong")[0] == 403
        assert request(server, "GET", headers={"Host": "attacker.example"})[0] == 403
        assert request(server, "GET", "/session.json")[0] == 404
        assert request(server, "GET", "/../../.env")[0] == 404
        for headers in [{"Origin": "https://attacker.example"}, {"Origin": "null"}]:
            assert request(server, "POST", "/publish", fields={"token": server.token}, headers=headers)[0] == 403
        assert request(server, "POST", "/publish", fields={"token": "wrong"})[0] == 403
        assert request(server, "POST", "/publish", fields={"token": server.token, "output_directory": "/tmp/injected"})[0] == 400
        assert request(server, "POST", "/publish", fields={"token": [server.token, server.token]})[0] == 403
        assert request(server, "POST", "/publish", fields={"token": server.token}, headers={"Content-Length": str(ui.MAX_REQUEST_BYTES + 1)})[0] == 400
        assert not (directory / "publications").exists()


def test_repository_and_reason_text_are_inert(ui_source, tmp_path):
    source, repository, _ = ui_source
    directory = tmp_path / "review"
    service.create_review(source, repository_root=repository, output_directory=directory)
    queue = service.review_queue(directory, repository_root=repository)
    payload = '</textarea><script src="https://attacker.example/x.js">alert(1)</script><img src=x onerror=alert(1)>'
    queue["claims"][0]["semantics"]["rule"] = payload
    queue["claims"][0]["detail_markdown"] = payload
    queue["claims"][0]["review_explanations"] = [payload]
    queue["claims"][0]["decision"] = {"decision": "accept", "reviewer_id": payload,
                                             "rationale": payload, "decided_at": "now", "version": 1}
    rendered = ui.render_page(queue, token="test-token").decode()
    assert '<script' not in rendered and '<img' not in rendered
    assert rendered.count('&lt;script') == 5
    assert 'action="/decision"' in rendered


def test_review_stop_and_invalid_decision(ui_source, tmp_path):
    source, repository, _ = ui_source
    directory = tmp_path / "review"
    session = service.create_review(source, repository_root=repository, output_directory=directory)
    with running_server(directory, repository) as server:
        status, _, _ = request(server, "POST", "/decision", fields={
            "token": server.token, "claim_id": next(iter(session.claim_digests)), "decision": "defer", "rationale": "",
        })
        assert status == 400
        assert service.load_review(directory).progress()["unprocessed"] == len(session.claim_digests)
        status, _, body = request(server, "POST", "/stop", fields={"token": server.token})
        assert status == 200 and "Review server stopped" in body
