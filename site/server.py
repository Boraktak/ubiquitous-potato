#!/usr/bin/env python3
"""Cerita — static file server + tiny JSON story store (stdlib only).

Serves the site files and a single resource:
    GET /api/stories   -> JSON array of all stories
    PUT /api/stories   -> replace the whole list (body = JSON array)

Stories are persisted to stories.json next to this file so they survive
restarts and can be shared across devices behind the same host.
"""

import json
import os
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT, "stories.json")
PORT = int(os.environ.get("PORT", "8080"))

_lock = threading.Lock()

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


def load():
    if os.path.isfile(DATA_FILE):
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (OSError, ValueError):
            pass
    return []


def save(data):
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DATA_FILE)


STORIES = load()


class Handler(BaseHTTPRequestHandler):
    server_version = "Cerita/1.0"
    protocol_version = "HTTP/1.1"

    # --- helpers ---------------------------------------------------------
    def _send(self, code, body=b"", ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self):
        self._send(204)

    # --- API -------------------------------------------------------------
    def do_GET(self):
        if self.path.split("?")[0] == "/api/stories":
            with _lock:
                body = json.dumps(STORIES, ensure_ascii=False).encode("utf-8")
            self._send(200, body)
            return
        self._serve_static()

    def do_PUT(self):
        global STORIES
        if self.path.split("?")[0] != "/api/stories":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, list):
                raise ValueError("body must be a JSON array")
            with _lock:
                STORIES = data
                save(data)
            self._json(200, {"ok": True, "count": len(data)})
        except (ValueError, OSError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})

    # --- static ----------------------------------------------------------
    def _serve_static(self):
        path = self.path.split("?")[0].split("#")[0]
        if path in ("", "/"):
            path = "/index.html"

        file_path = os.path.normpath(os.path.join(ROOT, path.lstrip("/")))
        if not file_path.startswith(ROOT) or not os.path.isfile(file_path):
            self._send(404, b"404 not found", "text/plain; charset=utf-8")
            return

        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, "rb") as f:
            body = f.read()
        self._send(200, body, MIME.get(ext, "application/octet-stream"))

    def log_message(self, fmt, *args):
        # quiet output
        pass


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Cerita server listening on http://0.0.0.0:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
