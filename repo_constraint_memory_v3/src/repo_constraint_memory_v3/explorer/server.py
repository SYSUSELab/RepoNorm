"""Serve one immutable catalog and the bundled explorer assets on loopback."""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from .package import load_package


def create_server(path: Path, port: int) -> ThreadingHTTPServer:
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    catalog = json.dumps(load_package(path), ensure_ascii=False).encode("utf-8")
    routes = {"/package.json": ("application/json", catalog)}
    for filename, mime in [("index.html", "text/html"), ("explorer.js", "text/javascript"),
                           ("explorer.css", "text/css")]:
        route = "/" if filename == "index.html" else "/" + filename
        routes[route] = (mime, files(__package__).joinpath(filename).read_bytes())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            allowed_hosts = {f"{host}:{self.server.server_port}" for host in ("localhost", "127.0.0.1")}
            if self.server.server_port == 80:
                allowed_hosts.update({"localhost", "127.0.0.1"})
            if self.headers.get("Host") not in allowed_hosts:
                self.send_error(403)
                return
            resource = routes.get(self.path.partition("?")[0])
            if resource is None:
                self.send_error(404)
                return
            mime, body = resource
            self.send_response(200)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'self'; style-src 'self'; "
                             "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(path: Path, port: int, open_browser: bool) -> None:
    with create_server(path, port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Knowledge explorer: {url}\nPress Ctrl+C to stop.", flush=True)
        if open_browser:
            webbrowser.open(url)
        server.serve_forever()
