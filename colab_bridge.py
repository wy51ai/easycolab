#!/usr/bin/env python3
"""
Local Headless Google Colab Bridge for EasyColab macOS App.
Powered by google-colab-cli (inspired by FireTable/comfy-colab-kit).
Supports:
  - auth_status : Check OAuth login state, real CU balance, hourly burn rate, and active VMs
  - auth_url    : Generate Google OAuth2 authorization URL for 1-click browser login
  - auth_code   : Exchange Google authorization code and save ~/.config/colab-cli/token.json
  - wake        : Provision/reuse Colab GPU VM (G4/H100/A100/L4/T4), deploy app.py, and return gradio.live URL
  - keepalive   : Refresh Colab runtime proxy & kernel activity while active
  - stop_all    : Unassign and terminate all active Colab VMs on the account (verify 0 CU/hr burn)
"""
import os
import re
import sys
import json
import time
import uuid
import signal
import argparse
import tempfile
import threading
import contextlib
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from importlib import resources

import requests
import websocket
from websocket import ABNF
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


def _ensure_utf8_stdio():
    """确保 Windows / macOS 子进程管道输出始终采用 UTF-8 编码，避免 emoji 在 GBK 代码页下抛出 UnicodeEncodeError"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_ensure_utf8_stdio()


def _sync_macos_system_proxies_to_env():
    """
    macOS / Linux 原生 GUI 应用启动子进程时默认可能不包含终端的 HTTP_PROXY / HTTPS_PROXY 环境变量，
    而 websocket-client (Jupyter 内核 WebSocket 通道) 仅从 os.environ 读取代理配置。
    此处自动读取 macOS / Windows / Linux 系统级代理 (scutil --proxy / Internet Settings / gsettings / urllib.request.getproxies) 并注入环境变量。
    """
    try:
        proxies = urllib.request.getproxies()
        for k in ("http", "https"):
            if k in proxies and proxies[k]:
                os.environ.setdefault(f"{k}_proxy", proxies[k])
                os.environ.setdefault(f"{k.upper()}_PROXY", proxies[k])
        if sys.platform.startswith("linux") and not (os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")):
            import subprocess
            mode = subprocess.run(
                ["gsettings", "get", "org.gnome.system.proxy", "mode"],
                capture_output=True, text=True, timeout=2
            ).stdout.strip().strip("'")
            if mode == "manual":
                for k in ("http", "https"):
                    host = subprocess.run(
                        ["gsettings", "get", f"org.gnome.system.proxy.{k}", "host"],
                        capture_output=True, text=True, timeout=2
                    ).stdout.strip().strip("'")
                    port_raw = subprocess.run(
                        ["gsettings", "get", f"org.gnome.system.proxy.{k}", "port"],
                        capture_output=True, text=True, timeout=2
                    ).stdout.strip()
                    port = port_raw.split()[-1].strip("'") if port_raw else ""
                    if host and port and port.isdigit() and int(port) > 0:
                        proxy_url = f"http://{host}:{port}"
                        os.environ.setdefault(f"{k}_proxy", proxy_url)
                        os.environ.setdefault(f"{k.upper()}_PROXY", proxy_url)
    except Exception:
        pass


_sync_macos_system_proxies_to_env()

from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH, REMOTE_REDIRECT_URI
from colab_cli.common import state
from colab_cli.client import Accelerator, Variant, Shape, resolve_assign_shape, PostAssignmentResponse, ColabRequestError
from colab_cli.state import SessionState
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient


def _load_client_config() -> dict:
    cfg_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    res = resources.files("colab_cli").joinpath("oauth_config.json")
    return json.loads(res.read_text(encoding="utf-8"))


def _has_valid_token() -> bool:
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return False
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.valid:
            return True
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            return True
    except Exception:
        return False
    return False


PKCE_VERIFIER_PATH = os.path.expanduser("~/.config/colab-cli/pkce_verifier.txt")
ACCOUNT_PROFILE_CACHE_PATH = os.path.expanduser("~/.config/colab-cli/account_profile.json")


def _clear_local_session_store():
    """切换或退出 Google 账号时，清理本地缓存的旧账号 Colab 会话记录"""
    try:
        existing = state.store.list()
        if isinstance(existing, dict):
            for name in list(existing.keys()):
                state.store.remove(name)
    except Exception:
        pass
    state._client = None
    state._sessions = None


def _get_account_email() -> str:
    """获取当前绑定 Google 账号邮箱（按 refresh_token 指纹本地缓存，避免重复请求）"""
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return ""
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        tok_fp = (creds.refresh_token or creds.token or "")[-20:]
        if tok_fp and os.path.exists(ACCOUNT_PROFILE_CACHE_PATH):
            try:
                with open(ACCOUNT_PROFILE_CACHE_PATH, "r", encoding="utf-8") as cf:
                    cached = json.load(cf)
                if cached.get("fp") == tok_fp and cached.get("email"):
                    return str(cached["email"])
            except Exception:
                pass
        if creds and creds.token:
            r = requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=5,
            )
            if r.status_code == 200:
                email = str(r.json().get("email") or "").strip()
                if email and tok_fp:
                    os.makedirs(os.path.dirname(ACCOUNT_PROFILE_CACHE_PATH), exist_ok=True)
                    with open(ACCOUNT_PROFILE_CACHE_PATH, "w", encoding="utf-8") as cf:
                        json.dump({"fp": tok_fp, "email": email}, cf)
                return email
    except Exception:
        pass
    return ""


def cmd_auth_url():
    client_config = _load_client_config()
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    # 显式传入 select_account consent，强制弹出 Google 多账号选择器以支持自由切换账号
    auth_url, _ = flow.authorization_url(prompt="select_account consent", token_usage="remote")
    os.makedirs(os.path.dirname(PKCE_VERIFIER_PATH), exist_ok=True)
    if getattr(flow, "code_verifier", None):
        with open(PKCE_VERIFIER_PATH, "w", encoding="utf-8") as f:
            f.write(flow.code_verifier)
    print(json.dumps({"ok": True, "auth_url": auth_url}))


def cmd_auth_code(code: str):
    try:
        client_config = _load_client_config()
        flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
        flow.redirect_uri = REMOTE_REDIRECT_URI
        if os.path.exists(PKCE_VERIFIER_PATH):
            with open(PKCE_VERIFIER_PATH, "r", encoding="utf-8") as f:
                flow.code_verifier = f.read().strip()
        flow.fetch_token(code=code.strip())
        creds = flow.credentials
        _clear_local_session_store()
        for p in (ACCOUNT_PROFILE_CACHE_PATH, PKCE_VERIFIER_PATH):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
        with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        return cmd_auth_status()
    except Exception as e:
        print(json.dumps({"ok": False, "logged_in": False, "error": str(e)}))


def cmd_auth_logout():
    _clear_local_session_store()
    for p in (TOKEN_CONFIG_PATH, PKCE_VERIFIER_PATH, ACCOUNT_PROFILE_CACHE_PATH):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    print(json.dumps({"ok": True, "logged_in": False, "email": ""}))


@contextlib.contextmanager
def _quiet_colab_cli():
    """将 colab_cli 内部 typer.echo (如 Pruned stale local session) 重定向至 stderr，保持 stdout 纯净 JSON 输出"""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _safe_sync_sessions():
    for attempt in range(3):
        try:
            with _quiet_colab_cli():
                return state.sync_sessions()
        except Exception:
            if attempt == 2:
                try:
                    with _quiet_colab_cli():
                        listed = state.store.list()
                except Exception:
                    listed = []
                if isinstance(listed, dict):
                    cached = dict(listed)
                else:
                    cached = {s.name: s for s in listed if getattr(s, "name", None)}
                return cached, []
            time.sleep(1.2)


def cmd_auth_status():
    if not _has_valid_token():
        print(json.dumps({"ok": True, "logged_in": False}))
        return

    try:
        info = state.client.get_consumption_user_info()
        local_sessions, assignments = _safe_sync_sessions()
        name_by_ep = {s.endpoint: s.name for s in local_sessions.values()}
        sess_list = []
        for a in assignments:
            sess_list.append({
                "name": name_by_ep.get(a.endpoint, "colab-vm"),
                "endpoint": a.endpoint,
                "accelerator": a.accelerator.value,
                "variant": a.variant.name,
                "shape": a.machine_shape.name,
            })

        email = _get_account_email()
        print(json.dumps({
            "ok": True,
            "logged_in": True,
            "email": email,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
            "sessions": sess_list,
        }))
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "logged_in": True,
            "email": _get_account_email(),
            "error": str(e),
        }))


def _build_rclone_drive_conf() -> str:
    """
    复用本地已授权的 Google Colab CLI OAuth 凭据 (含 drive.file 权限域)，
    生成云端 rclone Google Drive 配置，使无浏览器唤醒的 G4 实例也能自动挂载/同步 Google Drive，
    持久化保存 ComfyUI 自定义节点、模型与工作流。
    """
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return ""
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        raw = json.loads(creds.to_json())
        scopes = raw.get("scopes") or []
        drive_scope = "drive" if any(s.endswith("/drive") for s in scopes) else (
            "drive.file" if any("drive.file" in s for s in scopes) else ""
        )
        if not drive_scope or not raw.get("token") or not raw.get("refresh_token"):
            return ""
        tok_obj = {
            "access_token": raw["token"],
            "token_type": "Bearer",
            "refresh_token": raw["refresh_token"],
        }
        if raw.get("expiry"):
            tok_obj["expiry"] = raw["expiry"]
        lines = [
            "[easycolab_gdrive]",
            "type = drive",
            f"client_id = {raw.get('client_id', '')}",
            f"client_secret = {raw.get('client_secret', '')}",
            f"scope = {drive_scope}",
            f"token = {json.dumps(tok_obj, separators=(',', ':'))}",
            "",
        ]
        return "\n".join(lines)
    except Exception:
        return ""


def cmd_stop_all():
    if not _has_valid_token():
        print(json.dumps({"ok": False, "error": "尚未登录 Google Colab CLI"}))
        return

    try:
        with _quiet_colab_cli():
            local_sessions, assignments = state.sync_sessions()
        stopped_count = 0
        for s in list(local_sessions.values()):
            try:
                with _quiet_colab_cli():
                    rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id)
                    try:
                        rt.execute_code(
                            "import urllib.request\n"
                            "try:\n"
                            "    urllib.request.urlopen(urllib.request.Request("
                            "'http://127.0.0.1:27860/minimax_api/comfyui/sync_drive', "
                            "data=b'{}', headers={'Content-Type':'application/json'}, method='POST'), timeout=18)\n"
                            "except Exception:\n"
                            "    pass\n",
                            allow_stdin=False,
                            timeout=22,
                        )
                    except Exception:
                        pass
                    rt.stop(shutdown_kernel=True)
            except Exception:
                pass
            try:
                state.client.unassign(s.endpoint)
                stopped_count += 1
            except Exception:
                pass
            with _quiet_colab_cli():
                state.store.remove(s.name)

        # 同时清理所有未在本地登记的云端孤儿实例 (Orphan VMs)
        for a in assignments:
            try:
                state.client.unassign(a.endpoint)
                stopped_count += 1
            except Exception:
                pass

        info = state.client.get_consumption_user_info()
        print(json.dumps({
            "ok": True,
            "stopped_count": stopped_count,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
            "message": f"🛑 已彻底释放全部云端实例！当前计费速率: {info.consumption_rate_hourly:.2f} CU/小时",
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


def _wait_for_runtime_ready(session: SessionState, timeout_s: int = 45) -> bool:
    """
    等待新分配的 Colab 容器内部 Jupyter Server (/api/kernels) 完成启动。
    规避容器刚启动前几秒出现的 SSLEOFError / ConnectionReset / 502 / 503。
    若检测到 404 (实例如已被销毁)，立即返回 False 以便重新分配新实例。
    """
    base_url = session.url.rstrip("/")
    probe_url = f"{base_url}/api/kernels"
    headers = {
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
        "X-Colab-Runtime-Proxy-Token": session.token,
        "Authorization": f"Bearer {session.token}",
    }
    params = {"authuser": "0", "colab-runtime-proxy-token": session.token}
    deadline = time.time() + timeout_s
    not_found_count = 0

    while time.time() < deadline:
        try:
            r = requests.get(probe_url, headers=headers, params=params, timeout=8)
            if r.status_code == 200:
                return True
            if r.status_code in (404, 410):
                not_found_count += 1
                if not_found_count >= 2:
                    return False
        except requests.exceptions.RequestException:
            pass
        time.sleep(2.0)
    return False


GPU_ACCEL_MAP = {
    "h100": Accelerator.H100,
    "g4": Accelerator.G4,
    "a100": Accelerator.A100,
    "l4": Accelerator.L4,
    "t4": Accelerator.T4,
}

# G4 (RTX PRO 6000 Blackwell 96GB)、H100、L4 在 Colab 后端均为固定单规格大内存实例，传 shape=hm 会报 400 Bad Request
SINGLE_SHAPE_GPUS = frozenset({Accelerator.G4, Accelerator.H100, Accelerator.L4})


def _resolve_gpu_accel(gpu_type: str) -> Accelerator:
    return GPU_ACCEL_MAP.get((gpu_type or "g4").strip().lower(), Accelerator.G4)


def _allocate_fresh_session(session_name: str, gpu_type: str) -> SessionState:
    accel = _resolve_gpu_accel(gpu_type)
    shape = None if accel in SINGLE_SHAPE_GPUS else resolve_assign_shape(accel, high_mem=True)
    gpu_desc = "G4 (RTX PRO 6000 · 96GB)" if accel == Accelerator.G4 else accel.value
    print(f"WAKE_LOG: 🚀 正在向 Google Colab 申请分配 {gpu_desc} 实例...", flush=True)
    try:
        res = state.client.assign(uuid.uuid4(), variant=Variant.GPU, accelerator=accel, shape=shape)
    except ColabRequestError as req_err:
        if shape is not None and getattr(getattr(req_err, "response", None), "status_code", None) == 400:
            shape = None
            res = state.client.assign(uuid.uuid4(), variant=Variant.GPU, accelerator=accel, shape=None)
        else:
            raise
    if isinstance(res, PostAssignmentResponse):
        token = res.runtime_proxy_info.token
        url = res.runtime_proxy_info.url
        endpoint = res.endpoint
        exp = res.runtime_proxy_info.expires_at()
    else:
        token = res.runtime_proxy_info.token if hasattr(res, "runtime_proxy_info") else getattr(res, "runtime_proxy_token", "")
        url = res.runtime_proxy_info.url if hasattr(res, "runtime_proxy_info") else ""
        endpoint = res.endpoint
        exp = res.runtime_proxy_info.expires_at() if hasattr(res, "runtime_proxy_info") else None

    s = SessionState(
        name=session_name,
        token=token,
        url=url,
        endpoint=endpoint,
        token_expires_at=exp,
        variant=Variant.GPU.value,
        accelerator=accel.value,
        machine_shape=Shape.HIGH_RAM.name if shape == Shape.HIGH_RAM else Shape.STANDARD.name,
    )
    state.store.add(s)
    print(f"WAKE_LOG: ✅ 成功分配 {accel.value} 云端虚拟机！正在等待容器内核就绪...", flush=True)
    return s


def cmd_wake(gpu_type: str, app_py_path: str):
    """
    无浏览器一键唤醒 Colab GPU 并自动部署 EasyColab 多模型服务端：
    1. 幂等检查：若已有运行中的会话且探活成功则直接复用，否则自动清理失效会话并分配新实例
    2. 等待容器 Jupyter 端点 (/api/kernels) TLS/HTTP 握手就绪，规避 SSLEOFError
    3. 上传本地最新 app.py 到 /content/app.py
    4. 在远端内核安装依赖并后台拉起 app.py，提取 https://xxxx.gradio.live 隧道链接返回给 Mac 客户端
    """
    if not _has_valid_token():
        print("WAKE_ERROR: 尚未授权登录 Google 账号，请先点击「🔑 绑定账号」", flush=True)
        return

    import getpass
    password = os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", "")
    raw_idle_min = os.environ.get("EASYCOLAB_IDLE_TIMEOUT_MINUTES", "").strip()
    try:
        initial_idle_minutes = int(raw_idle_min) if raw_idle_min != "" else None
    except ValueError:
        initial_idle_minutes = None
    if not password and sys.stdin.isatty():
        password = getpass.getpass("Management password (24+ URL-safe characters): ")
    if not password:
        print("WAKE_ERROR: Set a management password before deploying", flush=True)
        return

    try:
        session_name = "easycolab"
        print("WAKE_LOG: 🔍 正在检查账号下已有的 Colab 实例...", flush=True)
        local_sessions, assignments = _safe_sync_sessions()

        target_session = (
            local_sessions.get(session_name)
            or local_sessions.get("aevix")
            or local_sessions.get("minimax")
        )
        if target_session:
            session_name = target_session.name
        elif local_sessions:
            target_session = list(local_sessions.values())[0]
            session_name = target_session.name

        # 若本地未记录但云端已有活跃分配，直接纳管该云端实例
        if not target_session and assignments:
            a = assignments[0]
            target_session = SessionState(
                name=session_name,
                token=a.runtime_proxy_info.token,
                url=a.runtime_proxy_info.url,
                endpoint=a.endpoint,
                token_expires_at=a.runtime_proxy_info.expires_at(),
                variant=Variant.GPU.value,
                accelerator=a.accelerator.value,
                machine_shape=Shape.HIGH_RAM.name if a.machine_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
            )
            state.store.add(target_session)

        requested_accel = _resolve_gpu_accel(gpu_type)
        if target_session:
            cur_acc = (target_session.accelerator or "").strip().upper()
            if cur_acc != requested_accel.value.upper():
                print(
                    f"WAKE_LOG: 🔄 当前已分配实例型号为 [{cur_acc or 'CPU'}]，正在释放并切换为您选择的 [{requested_accel.value}]...",
                    flush=True,
                )
                try:
                    state.client.unassign(target_session.endpoint)
                except Exception:
                    pass
                state.prune_session(target_session.name)
                target_session = None
            else:
                print(f"WAKE_LOG: ♻️ 正在检查已有实例 [{target_session.name}] ({target_session.accelerator}) 连通性...", flush=True)
                if not _wait_for_runtime_ready(target_session, timeout_s=15):
                    print("WAKE_LOG: ⚠️ 旧实例已失效或正在销毁，正在自动清理并重新分配新实例...", flush=True)
                    try:
                        state.client.unassign(target_session.endpoint)
                    except Exception:
                        pass
                    state.prune_session(target_session.name)
                    target_session = None

        if not target_session:
            target_session = _allocate_fresh_session(session_name, gpu_type)
            if not _wait_for_runtime_ready(target_session, timeout_s=50):
                print("WAKE_LOG: ⏳ 容器仍在初始化，继续尝试连接内核...", flush=True)

        print("WAKE_LOG: 正在上传最新多模型开源服务端代码 (app.py)...", flush=True)
        contents = ContentsClient(target_session)
        for up_attempt in range(4):
            try:
                contents.upload(app_py_path, "content/app.py")
                break
            except Exception as up_err:
                if up_attempt == 3:
                    raise up_err
                time.sleep(2.5)

        contents.upload(os.path.join(os.path.dirname(app_py_path), "runtime_auth.py"), "content/runtime_auth.py")
        rclone_conf_text = _build_rclone_drive_conf()
        if rclone_conf_text:
            tmp_rc_fd, tmp_rc_path = tempfile.mkstemp(prefix="easycolab_rc_", suffix=".conf")
            try:
                with os.fdopen(tmp_rc_fd, "w", encoding="utf-8") as rcf:
                    rcf.write(rclone_conf_text)
                os.chmod(tmp_rc_path, 0o600)
                contents.upload(tmp_rc_path, "content/.easycolab_rclone.conf")
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_rc_path):
                    try:
                        os.remove(tmp_rc_path)
                    except Exception:
                        pass

        print("WAKE_LOG: 正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...", flush=True)
        boot_code = r"""
import os, sys, time, re, shutil, subprocess

with open(ADMIN_STAGE, encoding="ascii") as stream:
    admin_password = stream.read()
os.unlink(ADMIN_STAGE)

if not shutil.which("nginx"):
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "nginx-light"], check=True)
    # apt may start the distribution's default site.
    subprocess.run(["service", "nginx", "stop"], check=False)

# 0. 配置无网页模式下的 Google Drive 云盘自动挂载/高速同步 (供模型库与 ComfyUI 节点持久化)
rc_stage = "/content/.easycolab_rclone.conf"
if os.path.exists(rc_stage):
    rc_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(rc_dir, mode=0o700, exist_ok=True)
    rc_dst = os.path.join(rc_dir, "rclone.conf")
    shutil.move(rc_stage, rc_dst)
    os.chmod(rc_dst, 0o600)

if os.path.exists(os.path.expanduser("~/.config/rclone/rclone.conf")):
    if not shutil.which("rclone"):
        subprocess.run(
            ["bash", "-lc", "curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && unzip -q -o /tmp/rclone.zip -d /tmp && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone && rm -rf /tmp/rclone*"],
            check=False,
        )
    if shutil.which("rclone") and not os.path.ismount("/content/drive/MyDrive"):
        subprocess.run(["rclone", "mkdir", "easycolab_gdrive:EasyColab_Drive"], check=False)
        if os.path.exists("/dev/fuse") and (shutil.which("fusermount3") or shutil.which("fusermount")):
            os.makedirs("/content/drive/MyDrive", exist_ok=True)
            subprocess.Popen(
                [
                    "rclone", "mount", "easycolab_gdrive:EasyColab_Drive", "/content/drive/MyDrive",
                    "--vfs-cache-mode", "writes",
                    "--vfs-write-back", "3s",
                    "--dir-cache-time", "15s",
                    "--drive-chunk-size", "64M",
                    "--buffer-size", "32M",
                    "--allow-non-empty",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(10):
                if os.path.ismount("/content/drive/MyDrive"):
                    print("✅ 已通过 Google OAuth 自动挂载 Google Drive (/content/drive/MyDrive)！", flush=True)
                    break
                time.sleep(0.5)
            else:
                # 容器未开放 FUSE 挂载权限时清理空目录，由 app.py 自动启用 rclone 云盘高速双向同步
                try:
                    if not os.listdir("/content/drive/MyDrive"):
                        os.rmdir("/content/drive/MyDrive")
                except Exception:
                    pass

# 1. 如果尚未安装 DiffSynth-Studio，自动快速安装 (含最新 transformers 以支持 Qwen3.8-27B)
if not os.path.exists("/content/DiffSynth-Studio"):
    print("正在克隆 DiffSynth-Studio...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/modelscope/DiffSynth-Studio.git", "/content/DiffSynth-Studio"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", "huggingface_hub", "hf_transfer", "qwen-vl-utils", "einops", "kernels>=0.17.2,<0.18.0", "av", "gradio", "modelscope", "imageio", "imageio-ffmpeg", "scipy", "soundfile", "librosa"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fla-core", "flash-linear-attention"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--only-binary=:all:", "causal-conv1d"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/DiffSynth-Studio"], check=True)
else:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        import kernels  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "kernels>=0.17.2,<0.18.0"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "fla-core", "flash-linear-attention"], check=False)

# 2. 检查是否已有运行中的 app.py 或 llama-server，先清理旧进程释放 GPU 显存
subprocess.run(["pkill", "-f", "python.*/content/app.py"], check=False)
subprocess.run(["pkill", "-f", "llama-server"], check=False)
time.sleep(1)

# 3. 后台启动 /content/app.py 并等待 gradio.live 链接生成
log_path = "/content/easycolab_server.log"
if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "w") as out_f:
    child_env = {**os.environ, "EASYCOLAB_ADMIN_PASSWORD": admin_password}
    if INITIAL_IDLE_MINUTES is not None:
        child_env["EASYCOLAB_IDLE_TIMEOUT_MINUTES"] = str(INITIAL_IDLE_MINUTES)
    subprocess.Popen([sys.executable, "-u", "/content/app.py"], cwd="/content", env=child_env, stdout=out_f, stderr=subprocess.STDOUT)

url_re = re.compile(r"^EasyColab URL: (https://[a-zA-Z0-9-]+\.gradio\.live)$", re.MULTILINE)
found_url = None
for _ in range(90):
    time.sleep(2)
    if os.path.exists(log_path):
        txt = open(log_path, "r", errors="replace").read()
        m = url_re.search(txt)
        if m:
            found_url = m.group(1)
            break

if found_url:
    print("WAKE_URL:" + found_url, flush=True)
else:
    tail = open(log_path, "r", errors="replace").read()[-600:] if os.path.exists(log_path) else "no log"
    print("WAKE_ERROR: 启动超时，日志尾部: " + tail, flush=True)
"""

        def _on_kid(kid):
            target_session.kernel_id = kid
            state.store.add(target_session)

        def _on_sid(sid):
            target_session.session_id = sid
            state.store.add(target_session)

        def _run_boot_on_runtime(use_saved_kernel: bool):
            rt = ColabRuntime(
                target_session.url,
                target_session.token,
                session_name=target_session.name,
                history=state.history,
                kernel_id=target_session.kernel_id if use_saved_kernel else None,
                session_id=target_session.session_id if use_saved_kernel else None,
                on_kernel_started=_on_kid,
                on_session_started=_on_sid,
            )
            try:
                # Each retry gets its own single-use file. The password never
                # enters the execute_request source or the streamed wake log.
                stage = ".easycolab-management-" + uuid.uuid4().hex
                with tempfile.TemporaryDirectory() as directory:
                    password_file = os.path.join(directory, "management")
                    with open(password_file, "w", encoding="ascii") as stream:
                        stream.write(password)
                    contents.upload(password_file, "content/" + stage)
                code = (
                    "ADMIN_STAGE = " + repr("/content/" + stage) + "\n"
                    + "INITIAL_IDLE_MINUTES = " + repr(initial_idle_minutes) + "\n"
                    + boot_code
                )
                return rt.execute_code(code, allow_stdin=False, timeout=300)
            finally:
                try:
                    rt.stop()
                except Exception:
                    pass

        try:
            outputs = _run_boot_on_runtime(use_saved_kernel=True)
        except Exception:
            target_session.kernel_id = None
            target_session.session_id = None
            outputs = _run_boot_on_runtime(use_saved_kernel=False)

        for out in outputs:
            if "text" in out:
                for line in out["text"].splitlines():
                    print(line, flush=True)
            elif out.get("output_type") == "error":
                print("WAKE_ERROR: " + out.get("evalue", "Execution error"), flush=True)

    except Exception as e:
        print(f"WAKE_ERROR: {str(e)}", flush=True)


class _ProxyAuthExpiredError(RuntimeError):
    """Runtime Proxy Token 已过期或失效 (HTTP 401/403)，需立即通过 list_assignments 刷新"""


def _session_token_ttl_seconds(s: SessionState) -> float:
    exp = getattr(s, "token_expires_at", None)
    if exp is None:
        return 3600.0
    try:
        if isinstance(exp, (int, float)):
            return float(exp) - time.time()
        if hasattr(exp, "timestamp"):
            return float(exp.timestamp()) - time.time()
        if isinstance(exp, str) and exp.strip():
            dt = datetime.fromisoformat(exp.strip().replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp() - time.time()
    except Exception:
        pass
    return 3600.0


def _sync_and_adopt_sessions():
    local_sessions, assignments = _safe_sync_sessions()
    ep_map = {a.endpoint: a for a in assignments}

    if not local_sessions and assignments:
        a0 = assignments[0]
        s_auto = SessionState(
            name="easycolab",
            token=a0.runtime_proxy_info.token,
            url=a0.runtime_proxy_info.url,
            endpoint=a0.endpoint,
            token_expires_at=a0.runtime_proxy_info.expires_at(),
            session_id=uuid.uuid4().hex,
            variant=Variant.GPU.value,
            accelerator=a0.accelerator.value,
            machine_shape=Shape.HIGH_RAM.name if a0.machine_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
        )
        with _quiet_colab_cli():
            state.store.add(s_auto)
        local_sessions = {"easycolab": s_auto}

    for s in list(local_sessions.values()):
        changed = False
        if s.endpoint in ep_map:
            info_proxy = ep_map[s.endpoint].runtime_proxy_info
            if s.token != info_proxy.token or s.url != info_proxy.url:
                changed = True
            s.token = info_proxy.token
            s.url = info_proxy.url
            s.token_expires_at = info_proxy.expires_at()
        if not getattr(s, "session_id", None):
            s.session_id = uuid.uuid4().hex
            changed = True
        if changed:
            with _quiet_colab_cli():
                state.store.add(s)

    return local_sessions, assignments


def _touch_colab_tunnel_keepalive(endpoint: str) -> bool:
    if not endpoint:
        return False
    try:
        keepalive_url = f"{state.client.colab_domain.rstrip('/')}/tun/m/{endpoint}/keep-alive/"
        state.client._issue_request(
            keepalive_url,
            method="GET",
            headers={"X-Colab-Tunnel": "Google"},
        )
        return True
    except Exception:
        return False


def _resolve_or_start_remote_kernel(s: SessionState, timeout_s: float = 12.0) -> str:
    """
    通过 Colab 官方 Runtime Proxy 查询或启动远端 Jupyter 内核。
    严禁删除其他已有内核，避免误杀并发调试或子任务内核。
    """
    base_url = (s.url or "").rstrip("/")
    if not base_url or not s.token:
        raise RuntimeError("Session missing runtime proxy URL or token")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Colab-Client-Agent": "vscode",
        "X-Colab-Runtime-Proxy-Token": s.token,
        "Authorization": f"Bearer {s.token}",
    }
    params = {"authuser": "0", "colab-runtime-proxy-token": s.token}

    r = requests.get(f"{base_url}/api/kernels", headers=headers, params=params, timeout=timeout_s)
    if r.status_code in (401, 403):
        raise _ProxyAuthExpiredError(f"Runtime proxy HTTP {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"GET /api/kernels returned HTTP {r.status_code}")

    kernels = r.json() if r.text else []
    active_kid = None
    if isinstance(kernels, list) and kernels:
        known_ids = [str(k.get("id")) for k in kernels if isinstance(k, dict) and k.get("id")]
        if known_ids:
            active_kid = s.kernel_id if (s.kernel_id in known_ids) else known_ids[0]

    if not active_kid:
        pr = requests.post(
            f"{base_url}/api/kernels",
            headers=headers,
            params=params,
            json={"name": "python3"},
            timeout=timeout_s,
        )
        if pr.status_code in (401, 403):
            raise _ProxyAuthExpiredError(f"POST /api/kernels HTTP {pr.status_code}")
        if pr.status_code not in (200, 201):
            raise RuntimeError(f"POST /api/kernels returned HTTP {pr.status_code}")
        pdata = pr.json() if pr.text else {}
        active_kid = str(pdata.get("id") or "")

    if not active_kid:
        raise RuntimeError("Unable to obtain remote Jupyter kernel_id")

    changed = False
    if s.kernel_id != active_kid:
        s.kernel_id = active_kid
        changed = True
    if not s.session_id:
        s.session_id = uuid.uuid4().hex
        changed = True
    if changed:
        with _quiet_colab_cli():
            state.store.add(s)
    return active_kid


def _open_colab_proxy_ws(s: SessionState, kernel_id: str, timeout_s: float = 15.0) -> websocket.WebSocket:
    """
    建立直连 Google Colab 官方外部 Runtime Proxy (*.prod.r.colab.dev) 的常驻 WebSocket 通道，
    显式携带 colab-runtime-proxy-token、token、authuser=0 查询参数与认证请求头，
    确保 Google 外部代理层始终观测到活跃 WebSocket 连接 (connections >= 1)。
    """
    base_url = (s.url or "").rstrip("/")
    if base_url.startswith("https://"):
        ws_base = "wss://" + base_url[len("https://"):]
    elif base_url.startswith("http://"):
        ws_base = "ws://" + base_url[len("http://"):]
    else:
        ws_base = "wss://" + base_url

    sid = s.session_id or uuid.uuid4().hex
    qs = urllib.parse.urlencode({
        "session_id": sid,
        "token": s.token,
        "colab-runtime-proxy-token": s.token,
        "authuser": "0",
    })
    ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels?{qs}"
    headers = [
        f"X-Colab-Runtime-Proxy-Token: {s.token}",
        f"Authorization: Bearer {s.token}",
        "X-Colab-Client-Agent: vscode",
        "User-Agent: EasyColab/2.0",
    ]
    return websocket.create_connection(
        ws_url,
        timeout=timeout_s,
        origin=base_url,
        header=headers,
        enable_multithread=True,
    )


def _drain_proxy_ws_frames(ws: websocket.WebSocket, wait_s: float = 1.0, stop_on_msg_id: str = "") -> bool:
    """排空服务端下发的 Jupyter 消息与控制帧（自动回复 Ping -> Pong，遇到 Close 立即返回 False）"""
    deadline = time.time() + max(0.05, wait_s)
    while True:
        rem = deadline - time.time()
        if rem <= 0:
            return True
        try:
            ws.settimeout(min(rem, 1.0))
            opcode, frame = ws.recv_data(control_frame=True)
            if opcode == ABNF.OPCODE_CLOSE:
                return False
            if stop_on_msg_id and opcode == ABNF.OPCODE_TEXT and frame:
                raw_txt = frame.decode("utf-8", errors="replace") if isinstance(frame, (bytes, bytearray)) else str(frame)
                if stop_on_msg_id in raw_txt and '"execute_reply"' in raw_txt:
                    return True
        except websocket.WebSocketTimeoutException:
            return True
        except Exception:
            return False


def _send_proxy_ws_heartbeat(ws: websocket.WebSocket, session_id: str, touch_client_hb: bool = False) -> bool:
    """在已打开的外部 Runtime Proxy WebSocket 上发送 Ping 与 Jupyter execute_request，刷新代理与内核活跃度"""
    msg_id = uuid.uuid4().hex
    hb_file = "/tmp/.easycolab_client_hb" if touch_client_hb else "/tmp/.easycolab_proxy_hb"
    code = f"import pathlib; pathlib.Path('{hb_file}').touch(); _easycolab_hb = 1"
    msg = {
        "channel": "shell",
        "header": {
            "msg_id": msg_id,
            "username": "easycolab",
            "session": session_id,
            "date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": False,
        },
    }
    try:
        ws.settimeout(10.0)
        ws.ping("easycolab")
        ws.send(json.dumps(msg))
        return _drain_proxy_ws_frames(ws, wait_s=2.0, stop_on_msg_id=msg_id)
    except Exception:
        return False


def cmd_keepalive():
    """
    单次向活跃的 Colab Headless 实例发送官方 Runtime Proxy 探活与轻量 Jupyter 内核脉冲（兼容 CLI / Windows / Linux 调用）。
    macOS 客户端优先使用长连接常驻子进程 keepalive_daemon。
    """
    if not _has_valid_token():
        print(json.dumps({"ok": True, "logged_in": False}))
        return

    try:
        local_sessions, _ = _sync_and_adopt_sessions()
        touched = 0
        warnings = []

        for s in list(local_sessions.values()):
            _touch_colab_tunnel_keepalive(s.endpoint)
            if not s.url or not s.token:
                continue

            ws = None
            try:
                active_kid = _resolve_or_start_remote_kernel(s, timeout_s=12.0)
                ws = _open_colab_proxy_ws(s, active_kid, timeout_s=15.0)
                if _send_proxy_ws_heartbeat(ws, s.session_id or uuid.uuid4().hex, touch_client_hb=True):
                    touched += 1
                else:
                    warnings.append(f"{s.name}: heartbeat closed")
            except Exception as touch_err:
                warnings.append(f"{s.name}: {touch_err}")
            finally:
                if ws is not None:
                    try:
                        ws.close(timeout=1)
                    except Exception:
                        pass

        info = state.client.get_consumption_user_info()
        payload = {
            "ok": True,
            "logged_in": True,
            "touched_kernels": touched,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
        }
        if warnings:
            payload["warning"] = "; ".join(warnings)
        print(json.dumps(payload))
    except Exception as e:
        print(json.dumps({"ok": False, "logged_in": True, "error": str(e)}))


def cmd_keepalive_daemon():
    """
    macOS 原生客户端专属常驻保活守护进程：
    - 始终保持一条直连 Google Colab 官方外部 Runtime Proxy (wss://*.prod.r.colab.dev/api/kernels/<id>/channels)
      的真实 WebSocket 长连接，使代理层与 Jupyter 内核的 connections 持续 >= 1（杜绝无浏览器模式下 ~20 分钟孤儿实例强杀）；
    - 每 25 秒在长连接上发送一次 WebSocket Ping + Jupyter execute_request 脉冲；
    - 每 45 秒调用 /tun/m/<endpoint>/keep-alive/ 并拉取实时 CU 余额/速率，通过 KEEPALIVE_JSON: 行流式推送给客户端；
    - 在代理 Token 临近过期 (<10 分钟) 或连接断开时自动刷新 runtimeProxyInfo 并无缝重连；
    - 监听父进程 stdin 管道 EOF 与进程生命周期：一旦 EasyColab.app 断开实例或退出，立即关闭 WebSocket 并零残留退出。
    """
    stop_event = threading.Event()
    initial_ppid = os.getppid()

    def _handle_sig(_signum, _frame):
        stop_event.set()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig_obj = getattr(signal, sig_name, None)
        if sig_obj is not None:
            try:
                signal.signal(sig_obj, _handle_sig)
            except Exception:
                pass

    if not sys.stdin.isatty():
        def _watch_stdin_eof():
            try:
                while not stop_event.is_set():
                    data = sys.stdin.read(1)
                    if data == "":
                        stop_event.set()
                        break
            except Exception:
                stop_event.set()

        threading.Thread(target=_watch_stdin_eof, daemon=True).start()

    def _emit(obj: dict):
        try:
            print("KEEPALIVE_JSON:" + json.dumps(obj, ensure_ascii=False), flush=True)
        except Exception:
            stop_event.set()

    if not _has_valid_token():
        _emit({"ok": True, "logged_in": False, "ws_connected": False})
        return

    ws_map: dict[str, tuple[websocket.WebSocket, str, str]] = {}  # endpoint -> (ws, url, token)

    def _close_ws(ep: str):
        entry = ws_map.pop(ep, None)
        if entry is not None:
            try:
                entry[0].close(timeout=1)
            except Exception:
                pass

    def _close_all_ws():
        for ep in list(ws_map.keys()):
            _close_ws(ep)

    last_sync_ts = 0.0
    last_hb_ts = 0.0
    last_usage_ts = 0.0
    force_resync = True
    local_sessions: dict[str, SessionState] = {}

    try:
        while not stop_event.is_set():
            if os.name != "nt" and os.getppid() != initial_ppid:
                stop_event.set()
                break

            now = time.time()
            min_ttl = min((_session_token_ttl_seconds(s) for s in local_sessions.values()), default=3600.0)
            if force_resync or (now - last_sync_ts) >= 480.0 or min_ttl < 600.0:
                try:
                    local_sessions, _ = _sync_and_adopt_sessions()
                    last_sync_ts = time.time()
                    force_resync = False
                    active_eps = {s.endpoint for s in local_sessions.values() if s.endpoint}
                    for stale_ep in list(ws_map.keys()):
                        if stale_ep not in active_eps:
                            _close_ws(stale_ep)
                except Exception as sync_err:
                    _emit({"ok": False, "logged_in": True, "ws_connected": bool(ws_map), "warning": f"sync: {sync_err}"})

            if not local_sessions:
                if (now - last_usage_ts) >= 30.0:
                    try:
                        info = state.client.get_consumption_user_info()
                        last_usage_ts = time.time()
                        _emit({
                            "ok": True,
                            "logged_in": True,
                            "ws_connected": False,
                            "paid_balance": round(float(info.paid_compute_units_balance), 2),
                            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
                            "active_assignments": int(info.assignments_count),
                        })
                    except Exception:
                        pass
                stop_event.wait(6.0)
                force_resync = True
                continue

            # 确保每个活跃会话都持有常驻外部 Runtime Proxy WebSocket
            need_hb = (time.time() - last_hb_ts) >= 25.0
            any_ws_ok = False
            last_warn = ""

            for s in list(local_sessions.values()):
                if stop_event.is_set():
                    break
                ep = s.endpoint or s.name
                existing = ws_map.get(ep)
                if existing is not None:
                    old_ws, old_url, old_tok = existing
                    if not getattr(old_ws, "connected", False) or old_url != s.url or old_tok != s.token:
                        _close_ws(ep)
                        existing = None

                if existing is None:
                    try:
                        kid = _resolve_or_start_remote_kernel(s, timeout_s=12.0)
                        new_ws = _open_colab_proxy_ws(s, kid, timeout_s=15.0)
                        if _send_proxy_ws_heartbeat(new_ws, s.session_id or uuid.uuid4().hex, touch_client_hb=False):
                            ws_map[ep] = (new_ws, s.url, s.token)
                            any_ws_ok = True
                            last_hb_ts = time.time()
                        else:
                            try:
                                new_ws.close(timeout=1)
                            except Exception:
                                pass
                            force_resync = True
                    except _ProxyAuthExpiredError as auth_err:
                        last_warn = str(auth_err)
                        force_resync = True
                    except Exception as conn_err:
                        last_warn = str(conn_err)
                        force_resync = True
                else:
                    cur_ws = existing[0]
                    if need_hb:
                        if _send_proxy_ws_heartbeat(cur_ws, s.session_id or uuid.uuid4().hex, touch_client_hb=False):
                            any_ws_ok = True
                        else:
                            _close_ws(ep)
                            force_resync = True
                            last_warn = "ws heartbeat lost, reconnecting"
                    else:
                        if _drain_proxy_ws_frames(cur_ws, wait_s=1.0):
                            any_ws_ok = True
                        else:
                            _close_ws(ep)
                            force_resync = True
                            last_warn = "ws closed by peer, reconnecting"

            if need_hb and any_ws_ok:
                last_hb_ts = time.time()

            now2 = time.time()
            if (now2 - last_usage_ts) >= 45.0 or (last_usage_ts == 0.0):
                for s in list(local_sessions.values()):
                    _touch_colab_tunnel_keepalive(s.endpoint)
                try:
                    info = state.client.get_consumption_user_info()
                    last_usage_ts = now2
                    payload = {
                        "ok": True,
                        "logged_in": True,
                        "ws_connected": any_ws_ok,
                        "paid_balance": round(float(info.paid_compute_units_balance), 2),
                        "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
                        "active_assignments": int(info.assignments_count),
                    }
                    if last_warn:
                        payload["warning"] = last_warn
                    _emit(payload)
                except Exception as usage_err:
                    last_usage_ts = now2
                    _emit({"ok": False, "logged_in": True, "ws_connected": any_ws_ok, "warning": str(usage_err)})

            if not any_ws_ok and not stop_event.is_set():
                stop_event.wait(3.0)
            elif need_hb and not stop_event.is_set():
                stop_event.wait(1.0)
    finally:
        _close_all_ws()


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth_status")
    sub.add_parser("auth_url")
    sub.add_parser("auth_logout")

    p_code = sub.add_parser("auth_code")
    p_code.add_argument("code", type=str)

    p_wake = sub.add_parser("wake")
    p_wake.add_argument("--gpu", type=str, default="G4")
    p_wake.add_argument(
        "--app-py",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
    )

    sub.add_parser("keepalive")
    sub.add_parser("keepalive_daemon")
    sub.add_parser("stop_all")

    args = parser.parse_args(argv)
    if args.cmd == "auth_status":
        cmd_auth_status()
    elif args.cmd == "auth_url":
        cmd_auth_url()
    elif args.cmd == "auth_logout":
        cmd_auth_logout()
    elif args.cmd == "auth_code":
        cmd_auth_code(args.code)
    elif args.cmd == "wake":
        cmd_wake(args.gpu, args.app_py)
    elif args.cmd == "keepalive":
        cmd_keepalive()
    elif args.cmd == "keepalive_daemon":
        cmd_keepalive_daemon()
    elif args.cmd == "stop_all":
        cmd_stop_all()


def run_cli(argv=None):
    """供独立 CLI 或打包后的 Windows exe (--bridge-cli) 子进程模式统一调用"""
    try:
        main(argv)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    run_cli()
