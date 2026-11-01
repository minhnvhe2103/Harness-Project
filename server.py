import json
import os
import queue
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import DEFAULT_MODEL
from supervisor import DEFAULT_WORKERS, SupervisorOrchestrator, WorkerProfile

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

# Session State
SESSION_LOCK = threading.Lock()
supervisor_model_name = DEFAULT_MODEL
worker_profiles: list[WorkerProfile] = [WorkerProfile(**vars(w)) for w in DEFAULT_WORKERS]


class AgentHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()

        elif parsed.path == "/api/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            data = {
                "models": [
                    "gemini-3.1-flash-lite",
                    "gemini-3.8-flash",
                    "claude-3-7-sonnet-20250219",
                ],
                "default": supervisor_model_name,
            }
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif parsed.path == "/api/team":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            with SESSION_LOCK:
                workers_data = [vars(w) for w in worker_profiles]
                response_data = {
                    "supervisor": {
                        "name": "Supervisor",
                        "avatar": "👑",
                        "role_title": "Quản lý & Điều phối tập trung (Saga Orchestrator)",
                        "model": supervisor_model_name,
                    },
                    "workers": workers_data,
                }
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        elif parsed.path == "/api/files":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()

            files_info = []
            try:
                for entry in sorted(os.scandir(BASE_DIR), key=lambda e: (not e.is_dir(), e.name.lower())):
                    if entry.name.startswith(".git") and entry.name != ".gitignore":
                        continue
                    is_dir = entry.is_dir()
                    size_str = "-" if is_dir else f"{entry.stat().st_size:,} B"
                    files_info.append({
                        "name": entry.name,
                        "is_dir": is_dir,
                        "size": size_str,
                    })
            except Exception as e:
                files_info = [{"name": f"Error: {e}", "is_dir": False, "size": "-"}]

            self.wfile.write(json.dumps({"files": files_info}).encode("utf-8"))
            return

        return super().do_GET()

    def do_POST(self):
        global supervisor_model_name
        parsed = urlparse(self.path)

        if parsed.path == "/api/reset":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            return

        elif parsed.path == "/api/team/toggle":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                target_id = payload.get("id")
                enabled = payload.get("enabled", True)
                model = payload.get("model")

                with SESSION_LOCK:
                    if target_id == "supervisor":
                        if model:
                            supervisor_model_name = model
                    else:
                        for w in worker_profiles:
                            if w.id == target_id:
                                w.enabled = enabled
                                if model:
                                    w.model = model
                                break

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "updated"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode("utf-8"))
            return

        elif parsed.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                payload = json.loads(post_data.decode("utf-8"))
                prompt = payload.get("prompt", "").strip()
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Invalid JSON: {e}".encode("utf-8"))
                return

            if not prompt:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Prompt cannot be empty")
                return

            # Setup SSE Headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            event_queue = queue.Queue()

            def forward_event(ev_type: str, data: dict):
                event_queue.put((ev_type, data))

            def worker():
                try:
                    with SESSION_LOCK:
                        active_workers = [WorkerProfile(**vars(w)) for w in worker_profiles]
                        sup_model = supervisor_model_name

                    orchestrator = SupervisorOrchestrator(
                        supervisor_model=sup_model,
                        workers_config=active_workers,
                        on_event=forward_event,
                    )
                    orchestrator.run(prompt)
                except Exception as ex:
                    event_queue.put(("error", {"message": str(ex)}))
                finally:
                    event_queue.put(("__DONE__", {}))

            t = threading.Thread(target=worker, daemon=True)
            t.start()

            # Stream events to client
            while True:
                try:
                    ev_type, data = event_queue.get(timeout=60)
                except queue.Empty:
                    break

                if ev_type == "__DONE__":
                    break

                msg = f"event: {ev_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                try:
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break

            return

        self.send_response(404)
        self.end_headers()


def start_server(host: str = "127.0.0.1", port: int = 8000):
    server = ThreadingHTTPServer((host, port), AgentHTTPHandler)
    print("=" * 60)
    print(f"🚀 Supervisor Multi-Agent Web UI đang chạy tại: http://localhost:{port}")
    print(f"Supervisor Model: {supervisor_model_name}")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port=port)
