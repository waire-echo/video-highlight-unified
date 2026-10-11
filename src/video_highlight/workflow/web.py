"""Loopback WebUI with a shared controller and a session-only mutation token."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
from urllib.parse import parse_qs, urlsplit

from .control import Controller

ASSETS = Path(__file__).with_name("ui")


def server(controller, port=8189):
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, data, content_type="application/json; charset=utf-8"):
            body = json.dumps(data, ensure_ascii=False).encode() if isinstance(data, (dict, list)) else data
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlsplit("http://" + self.headers.get("Host", "")).hostname not in {"localhost", "127.0.0.1", "::1"}:
                return self.send(403, {"error": "Use a loopback host through SSH forwarding"})
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            try:
                if parts.path == "/":
                    body = (ASSETS / "index.html").read_text().replace("SESSION_TOKEN", token).encode()
                    return self.send(200, body, "text/html; charset=utf-8")
                if parts.path in {"/app.js", "/style.css"}:
                    file = ASSETS / parts.path[1:]
                    return self.send(200, file.read_bytes(), "text/javascript; charset=utf-8" if file.suffix == ".js" else "text/css; charset=utf-8")
                if parts.path == "/api/bootstrap":
                    return self.send(200, controller.bootstrap())
                if parts.path == "/api/resources":
                    return self.send(200, controller.resources())
                if parts.path == "/api/runs":
                    return self.send(200, controller.runs())
                if parts.path == "/api/run":
                    return self.send(200, controller.detail(query["id"][0]))
                if parts.path == "/api/job":
                    return self.send(200, controller.job(query["id"][0]))
                if parts.path == "/api/artifact":
                    file = controller.artifact(query["id"][0], query["name"][0])
                    with file.open("rb") as stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(file.stat().st_size))
                        self.send_header("Content-Disposition", 'attachment; filename="' + file.name + '"')
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        shutil.copyfileobj(stream, self.wfile, 128 << 10)
                    return
                self.send(404, {"error": "Not found"})
            except (OSError, ValueError, KeyError) as error:
                self.send(400, {"error": str(error)})

        def do_POST(self):
            if urlsplit("http://" + self.headers.get("Host", "")).hostname not in {"localhost", "127.0.0.1", "::1"}:
                return self.send(403, {"error": "Use a loopback host through SSH forwarding"})
            if self.headers.get("X-Workflow-Token") != token:
                return self.send(403, {"error": "Invalid UI session"})
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                return self.send(403, {"error": "Cross-origin actions are disabled"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= (1 << 20):
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")
                action = urlsplit(self.path).path
                if action == "/api/plan":
                    result = controller.plan(data)
                elif action == "/api/start":
                    result = controller.start(data["plan_id"], data.get("until", "complete"))
                elif action == "/api/resume":
                    result = controller.resume(data["run_id"], data.get("until", "complete"))
                elif action == "/api/stop":
                    result = controller.stop(data["job_id"])
                else:
                    return self.send(404, {"error": "Not found"})
                self.send(200, result)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.send(400, {"error": str(error)})
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(local, port=8189, runs_root=None):
    controller = Controller(local, runs_root)
    with server(controller, port) as http:
        print(f"WebUI listening on http://127.0.0.1:{http.server_port} (SSH forwarding supported)", flush=True)
        try:
            http.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
