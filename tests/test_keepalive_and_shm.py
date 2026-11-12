"""Unit tests for Colab keepalive daemon, RFC 6455 WS framing, and /dev/shm Cgroup RAM guard."""

import ast
from collections import namedtuple
import io
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

import colab_bridge
from colab_cli.state import SessionState

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PY_PATH = os.path.join(REPO_ROOT, "app.py")


def _load_app_helpers():
    with open(APP_PY_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=APP_PY_PATH)
    wanted = {
        "_build_ws_masked_text_frame",
        "_build_ws_masked_pong_frame",
        "_recv_ws_exact",
        "_recv_ws_frame",
        "_drain_local_ws_frames",
        "_safe_shm_available_bytes",
    }
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {
        "os": os,
        "socket": socket,
        "shutil": mock.MagicMock(),
        "Optional": object,
        "Tuple": tuple,
        "LOADED_RUNTIMES": {},
        "_get_system_ram_gb": lambda: (0.0, 0.0),
    }
    exec(compile(mod, APP_PY_PATH, "exec"), ns)
    return ns


class WebSocketFramingTest(unittest.TestCase):
    def test_fragmented_payload_with_0x08_and_0x09_nibbles_not_misread_as_close(self):
        ns = _load_app_helpers()
        s_client, s_server = socket.socketpair()
        self.addCleanup(s_client.close)
        self.addCleanup(s_server.close)

        # Construct a 260-byte unmasked server text frame (opcode=0x1, len=126 extended)
        # whose payload contains 'h' (0x68 & 0x0F == 0x08) and 'i' (0x69 & 0x0F == 0x09) at chunk boundaries.
        payload = b"A" * 50 + b"h" + b"B" * 80 + b"i" + b"C" * 128
        header = bytes([0x81, 126]) + len(payload).to_bytes(2, "big")
        wire = header + payload

        def _trickle():
            s_server.sendall(wire[:54])  # First chunk ends right before 'h' (0x68)
            time.sleep(0.02)
            s_server.sendall(wire[54:135])  # Second chunk starts with 'h' (low nibble 0x08!)
            time.sleep(0.02)
            s_server.sendall(wire[135:])  # Third chunk starts with 'i' (low nibble 0x09!)

        t = threading.Thread(target=_trickle, daemon=True)
        t.start()
        self.assertTrue(ns["_drain_local_ws_frames"](s_client, max_frames=8, poll_timeout=0.5))
        t.join(timeout=1.0)

    def test_ping_echoes_masked_pong_with_payload_and_close_returns_false(self):
        ns = _load_app_helpers()
        s_client, s_server = socket.socketpair()
        self.addCleanup(s_client.close)
        self.addCleanup(s_server.close)

        ping_payload = b"colab-ping-42"
        ping_frame = bytes([0x89, len(ping_payload)]) + ping_payload
        s_server.sendall(ping_frame)

        self.assertTrue(ns["_drain_local_ws_frames"](s_client, max_frames=4, poll_timeout=0.3))

        pong = ns["_recv_ws_frame"](s_server, poll_timeout=0.5)
        self.assertIsNotNone(pong)
        opcode, echoed = pong
        self.assertEqual(opcode, 0x0A)
        self.assertEqual(echoed, ping_payload)

        # Now send an actual RFC 6455 Close frame (0x88)
        s_server.sendall(bytes([0x88, 0x02, 0x03, 0xE8]))
        self.assertFalse(ns["_drain_local_ws_frames"](s_client, max_frames=4, poll_timeout=0.3))


class SafeShmCgroupGuardTest(unittest.TestCase):
    def test_blocks_shm_staging_when_cgroup_ram_is_pinned_by_hot_standby(self):
        ns = _load_app_helpers()
        DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
        ns["shutil"].disk_usage.return_value = DiskUsage(88 * 1024**3, 0, 88 * 1024**3)
        # Simulate G4 (176GB total) with MiniMax-H3 Tier-1 pinning ~146GB, leaving only 30GB MemAvailable
        ns["_get_system_ram_gb"] = lambda: (176.0, 30.0)
        ns["LOADED_RUNTIMES"] = {"minimax_h3": {"mode_label": "MiniMax-H3 · 双主干内存热备"}}

        with mock.patch("os.path.exists", return_value=True):
            self.assertEqual(ns["_safe_shm_available_bytes"](reserve_ram_gb=34.0), 0)

    def test_allows_shm_staging_when_system_ram_is_plentiful(self):
        ns = _load_app_helpers()
        DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
        ns["shutil"].disk_usage.return_value = DiskUsage(88 * 1024**3, 8 * 1024**3, 80 * 1024**3)
        ns["_get_system_ram_gb"] = lambda: (176.0, 150.0)
        ns["LOADED_RUNTIMES"] = {}

        with mock.patch("os.path.exists", return_value=True):
            avail = ns["_safe_shm_available_bytes"](reserve_ram_gb=34.0)
            self.assertEqual(avail, 80 * 1024**3)


class ColabBridgeKeepaliveTest(unittest.TestCase):
    def test_quiet_colab_cli_keeps_stdout_clean(self):
        fake_out = io.StringIO()
        fake_err = io.StringIO()
        with mock.patch.object(sys, "stdout", fake_out), mock.patch.object(sys, "stderr", fake_err):
            with colab_bridge._quiet_colab_cli():
                print("[colab] Pruned 1 stale local session(s).")
            print('{"ok": true}')
        self.assertEqual(fake_out.getvalue().strip(), '{"ok": true}')
        self.assertIn("Pruned 1 stale", fake_err.getvalue())

    def test_resolve_remote_kernel_never_deletes_other_kernels_and_detects_401(self):
        sess = SessionState(
            name="easycolab",
            token="tok_abc",
            url="https://8080-gpu-test.prod.r.colab.dev",
            endpoint="m-s-test",
            kernel_id="kid-main",
            session_id="sid-fixed",
        )
        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.text = '[{"id":"kid-other"},{"id":"kid-main"}]'
        fake_resp.json.return_value = [{"id": "kid-other"}, {"id": "kid-main"}]

        with mock.patch("colab_bridge.requests.get", return_value=fake_resp) as m_get, \
             mock.patch("colab_bridge.requests.delete") as m_del, \
             mock.patch.object(colab_bridge.state.store, "add"):
            kid = colab_bridge._resolve_or_start_remote_kernel(sess)
            self.assertEqual(kid, "kid-main")
            m_del.assert_not_called()
            _, kwargs = m_get.call_args
            self.assertEqual(kwargs["params"]["colab-runtime-proxy-token"], "tok_abc")
            self.assertEqual(kwargs["params"]["authuser"], "0")

        expired_resp = mock.MagicMock(status_code=401, text="")
        with mock.patch("colab_bridge.requests.get", return_value=expired_resp):
            with self.assertRaises(colab_bridge._ProxyAuthExpiredError):
                colab_bridge._resolve_or_start_remote_kernel(sess)

    def test_open_colab_proxy_ws_includes_token_query_and_headers(self):
        sess = SessionState(
            name="easycolab",
            token="proxy_tok_123",
            url="https://8080-gpu-test.prod.r.colab.dev/",
            endpoint="m-s-test",
            kernel_id="kid-1",
            session_id="sid-1",
        )
        with mock.patch("colab_bridge.websocket.create_connection") as m_ws:
            colab_bridge._open_colab_proxy_ws(sess, "kid-1", timeout_s=9.0)
            args, kwargs = m_ws.call_args
            ws_url = args[0]
            self.assertTrue(ws_url.startswith("wss://8080-gpu-test.prod.r.colab.dev/api/kernels/kid-1/channels?"))
            self.assertIn("colab-runtime-proxy-token=proxy_tok_123", ws_url)
            self.assertIn("token=proxy_tok_123", ws_url)
            self.assertIn("authuser=0", ws_url)
            self.assertIn("session_id=sid-1", ws_url)
            self.assertIn("X-Colab-Runtime-Proxy-Token: proxy_tok_123", kwargs["header"])

    def test_keepalive_daemon_exits_promptly_when_parent_stdin_closes(self):
        code = (
            "import time, unittest.mock as m, colab_bridge;"
            "m.patch('colab_bridge._has_valid_token', return_value=True).start();"
            "m.patch('colab_bridge._sync_and_adopt_sessions', return_value=({}, [])).start();"
            "info = type('I', (), {'paid_compute_units_balance': 100.0, 'consumption_rate_hourly': 0.0, 'assignments_count': 0})();"
            "m.patch.object(colab_bridge.state, '_client', type('C', (), {'get_consumption_user_info': lambda self: info})());"
            "colab_bridge.cmd_keepalive_daemon()"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        first_line = proc.stdout.readline().strip()
        self.assertTrue(first_line.startswith("KEEPALIVE_JSON:"), msg=first_line)
        # Close parent stdin pipe -> daemon must exit immediately without hanging for 6s+
        t0 = time.time()
        proc.stdin.close()
        rc = proc.wait(timeout=2.5)
        proc.stdout.close()
        proc.stderr.close()
        self.assertEqual(rc, 0)
        self.assertLess(time.time() - t0, 2.0)


if __name__ == "__main__":
    unittest.main()
