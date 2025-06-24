import http.server
import json
import socketserver
import webbrowser
import socket
import os
import time
import threading


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=script_dir, **kwargs)

    def do_GET(self):
        if self.path == "/graph_data":
            graph_data = self.server.graph_data
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(graph_data).encode())
        else:
            super().do_GET()


def open_browser(port):
    webbrowser.open(f"http://localhost:{port}")


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("localhost", port))
            return True
        except OSError:
            return False


def find_available_port(initial_port):
    port = initial_port
    while not is_port_available(port):
        print(f"Port {port} not available. Trying next one...")
        port += 1
    return port


def start_server_on_port(graph_data, port):
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving HTTP on localhost port {port}...")
        httpd.graph_data = graph_data
        httpd.serve_forever()


def start_server(graph_data):
    """
    The server will only stay open for a fixed amount of time, enough to load the visualization.
    Afterwards, the simulator execution will continue.
    To achieve so, the server is run on a different thread with a timeout.
    """
    print("Launching topology visualization...")
    print("=========================================")
    port = find_available_port(initial_port=8000)
    open_browser(port)
    server_thread = threading.Thread(
        target=start_server_on_port, args=(graph_data, port)
    )
    server_thread.daemon = (
        True  # This allows the program to exit even if the thread is running
    )
    server_thread.start()
    time.sleep(1)
    server_thread.join(timeout=1)
    print("Stopping server and continuing program execution...")
    print(
        "[Refreshing the visualization tab will not work since the server is stopped]"
    )
    print("=========================================")
