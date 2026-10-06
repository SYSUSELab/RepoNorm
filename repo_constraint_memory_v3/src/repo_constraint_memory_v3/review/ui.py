"""On-demand, provider-free developer review on the loopback interface."""
from __future__ import annotations

from collections.abc import Callable
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from threading import Thread
from urllib.parse import parse_qs, urlencode, urlsplit
import webbrowser

from . import service

# A full 32,768-character rationale may expand twelvefold in URL encoding.
MAX_REQUEST_BYTES = 500_000


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _hidden(name: str, value: str) -> str:
    return f'<input type="hidden" name="{name}" value="{_text(value)}">'


def _document(body: str) -> bytes:
    return ('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Norm4Code Repository Norm Review</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1040px;margin:32px auto;padding:0 20px;color:#17212c;background:#f5f7fa;overflow-wrap:anywhere}
article,header,section{background:white;border:1px solid #dce1e8;border-radius:10px;padding:20px;margin:20px 0}
h1{font-size:26px}h2{font-size:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;background:#f5f7fa;padding:12px}
dl{display:grid;grid-template-columns:minmax(140px,180px) minmax(0,1fr);gap:6px}dt{font-weight:bold}dd{margin:0;overflow-wrap:anywhere}
textarea{box-sizing:border-box;width:100%;min-height:90px;font:inherit}button{font:inherit;padding:7px 16px;margin:8px 10px 0 0;cursor:pointer}
label{margin-right:22px}summary{cursor:pointer}code{overflow-wrap:anywhere}.muted{color:#4a5869}.status{font-weight:bold}
</style><body>''' + body + '</body></html>').encode('utf-8')


def render_page(queue: dict[str, object], *, token: str, publication: Path | None = None) -> bytes:
    """Render supplied repository text as inert text, including evidence Markdown."""
    session = queue["session"]
    progress = queue["progress"]
    completeness = queue["completeness"]
    controls = _hidden("token", token)
    body = [f'''<header><h1>Norm4Code Repository Norm Review</h1>
<p>Repository: {_text(session['repository'])}<br>Snapshot: <code>{_text(session['snapshot_digest'])}</code>
<br>Review session: <code>{_text(session['session_id'])}</code></p>
<p>Total: {progress['planned']} · Accepted: {progress['accepted']} · Rejected: {progress['rejected']} · Unprocessed: {progress['unprocessed']}</p>
<p>Source extraction completeness: <strong>{_text(completeness.get('package_status', 'unknown'))}</strong>.
Completing review does not repair source extraction gaps.</p>
<details><summary>Source completeness record</summary><pre>{_text(json.dumps(completeness, ensure_ascii=False, indent=2))}</pre></details>
<p>Act as a repository maintainer or developer: would you endorse each rule as written and give it to a coding agent working in this repository? Save decisions to resume later.
Publication includes original accepted claims and this session's saved acceptances; rejected and unprocessed claims are excluded. Rules cannot be edited.</p>
<form method="post" action="/publish">{controls}<button>Publish new consumer package</button></form>
<form method="post" action="/stop">{controls}<button>Stop review server</button></form></header>''']
    if publication is not None:
        body.append(f'<section role="status">Published. Give the coding agent this directory: <code>{_text(publication / "consumer")}</code></section>')
    if not queue["claims"]:
        body.append('<section>This session has no review candidates. You can still publish a separate consumer package containing the original accepted claims.</section>')
    for claim in queue["claims"]:
        semantics = claim["semantics"]
        decision = claim.get("decision")
        selected = decision["decision"] if decision else None
        state = {"accept": "Accepted", "reject": "Rejected", None: "Unprocessed"}[selected]
        metadata = [
            ("Normalized action", semantics["normalized_action"]),
            ("Polarity", semantics["polarity"]),
            ("Applies to", "\n".join(semantics["applies_to"])),
            ("Condition", semantics.get("condition") or "none"),
            ("Exception", semantics.get("exception") or "none"),
            ("Normative strength", claim["normative_strength"]),
            ("Execution risk", claim["execution_risk"]),
            ("Original delivery policy", claim["delivery_policy"]),
        ]
        fields = ''.join(f'<dt>{label}</dt><dd>{_text(value)}</dd>' for label, value in metadata)
        reasons = ''.join(f'<li>{_text(reason)}</li>' for reason in claim["review_explanations"])
        radios = ''.join(
            f'<label><input type="radio" name="decision" value="{value}" required'
            + (' checked' if selected == value else '') + f'>{label}</label>'
            for value, label in [("accept", "Accept"), ("reject", "Reject")]
        )
        provenance = '' if decision is None else f'<p class="muted">Saved: {_text(decision["reviewer_id"])} · {_text(decision["decided_at"])} · Version {decision["version"]}</p>'
        body.append(f'''<article id="{_text(claim['claim_id'])}">
<h2>{_text(semantics['rule'])}</h2><p><code>{_text(claim['claim_id'])}</code> · <span class="status">{state}</span></p>
<dl>{fields}</dl><details><summary>Review reasons from the source extraction</summary><ul>{reasons}</ul></details>
<details><summary>Supporting evidence, counterevidence, and existing assessment</summary><pre>{_text(claim['detail_markdown'])}</pre></details>
<form method="post" action="/decision">{controls}{_hidden('claim_id', claim['claim_id'])}
<p>{radios}</p><label>Rationale (optional)<textarea name="rationale" maxlength="32768">{_text(decision['rationale'] if decision else '')}</textarea></label>
<button>Save decision</button></form>{provenance}</article>''')
    return _document(''.join(body))


class ReviewHTTPServer(HTTPServer):
    """Single local operator; service-level locks protect persistent decisions."""

    def __init__(self, directory: Path, repository: Path, *, port: int, reviewer_id: str):
        self.review_directory = directory
        self.repository_root = repository
        self.reviewer_id = reviewer_id
        self.token = secrets.token_urlsafe(32)
        self.publication: Path | None = None
        super().__init__(("127.0.0.1", port), _ReviewHandler)
        self.authority = f"127.0.0.1:{self.server_port}"
        self.origin = "http://" + self.authority
        self.url = self.origin + "/?" + urlencode({"token": self.token})

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address


class _ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        # Request URLs contain the local access token.
        return

    def _respond(self, status: int, body: bytes, *, location: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # no-referrer makes browser form POSTs send Origin: null. Keep the
        # origin for same-origin checking without exposing the URL's token.
        self.send_header("Referrer-Policy", "strict-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
        if location is not None:
            self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._respond(status, _document(f'<h1>Review request could not be completed</h1><p>{_text(message)}</p>'))

    def _host_allowed(self) -> bool:
        if self.headers.get_all("Host") != [self.server.authority]:
            self._error(403, "Only the displayed loopback address is allowed.")
            return False
        return True

    def _authorized(self, fields: dict[str, list[str]]) -> bool:
        values = fields.get("token", [])
        if len(values) != 1 or not secrets.compare_digest(values[0].encode(), self.server.token.encode()):
            self._error(403, "Invalid review access token.")
            return False
        return True

    def do_GET(self) -> None:
        if not self._host_allowed():
            return
        path = urlsplit(self.path)
        if path.path != "/":
            self._error(404, "Unknown review page.")
            return
        try:
            fields = parse_qs(path.query, max_num_fields=4)
            if not self._authorized(fields):
                return
            queue = service.review_queue(self.server.review_directory, repository_root=self.server.repository_root)
            self._respond(200, render_page(queue, token=self.server.token, publication=self.server.publication))
        except (ValueError, OSError) as error:
            self._error(400, str(error))

    def do_POST(self) -> None:
        if not self._host_allowed():
            return
        if self.headers.get_all("Origin") != [self.server.origin]:
            self._error(403, "A same-origin review form is required.")
            return
        if self.headers.get_content_type() != "application/x-www-form-urlencoded" or self.headers.get("Transfer-Encoding") is not None:
            self._error(400, "Expected a bounded form request.")
            return
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                raise ValueError("A single content length is required.")
            size = int(lengths[0])
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("Review form exceeds request size limit.")
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("Incomplete review form.")
            fields = parse_qs(raw.decode("utf-8"), keep_blank_values=True, strict_parsing=True, max_num_fields=8)
            if not self._authorized(fields):
                return
            if any(len(values) != 1 for values in fields.values()):
                raise ValueError("Duplicate review form fields.")
            if self.path == "/decision":
                if set(fields) != {"token", "claim_id", "decision", "rationale"}:
                    raise ValueError("Unexpected decision form fields.")
                if fields["decision"][0] not in {"accept", "reject"}:
                    raise ValueError("Choose accept or reject.")
                service.save_decision(self.server.review_directory, repository_root=self.server.repository_root,
                                      claim_id=fields["claim_id"][0], decision=fields["decision"][0],
                                      reviewer_id=self.server.reviewer_id, rationale=fields["rationale"][0])
            elif self.path in {"/publish", "/stop"}:
                if set(fields) != {"token"}:
                    raise ValueError("Unexpected review form fields.")
                if self.path == "/publish":
                    plan = service.publish_review(self.server.review_directory, repository_root=self.server.repository_root)
                    self.server.publication = plan.publication_directory
                else:
                    self._respond(200, _document('<h1>Review server stopped</h1><p>Saved decisions remain in this review session. Resume using the same directory.</p>'))
                    Thread(target=self.server.shutdown, daemon=True).start()
                    return
            else:
                self._error(404, "Unknown review action.")
                return
            self._respond(303, b"", location=self.server.url)
        except (ValueError, OSError) as error:
            self._error(400, str(error))


def create_review_server(review_directory: Path, *, repository_root: Path,
                         port: int = 0, reviewer_id: str = "local-developer") -> ReviewHTTPServer:
    if not 0 <= port <= 65535:
        raise ValueError("review port must be between 0 and 65535")
    if not reviewer_id.strip() or len(reviewer_id) > 1024:
        raise ValueError("reviewer identity must contain 1 to 1024 characters")
    service.load_review(review_directory, repository_root=repository_root)
    return ReviewHTTPServer(review_directory, repository_root, port=port, reviewer_id=reviewer_id)


def serve_review(review_directory: Path, *, repository_root: Path, port: int = 0,
                 reviewer_id: str = "local-developer", open_browser: bool = True,
                 on_ready: Callable[[str], object] = print) -> None:
    """Serve until Ctrl-C or the operator selects the stop button."""
    with create_review_server(review_directory, repository_root=repository_root,
                              port=port, reviewer_id=reviewer_id) as server:
        on_ready(server.url)
        if open_browser:
            webbrowser.open(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
