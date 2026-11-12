"""
Core ViewModel (EasyColabViewModel) for the EasyColab Windows Client.
Handles settings persistence, asynchronous HTTP calls to Colab (/minimax_api/*),
headless colab_bridge.py subprocess orchestration (with CREATE_NO_WINDOW on Windows),
2.5s telemetry & keepalive watchdog loop, job polling, and local media management.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from PySide6.QtCore import QObject, QTimer, Signal

from windows_app.models import (
    IMAGE_RESOLUTIONS,
    MODEL_SPEC_MAP,
    OPEN_MODEL_CATALOG,
    PRESET_TEMPLATES,
    VIDEO_RESOLUTIONS,
    ChatTurnItem,
    ComfyOutputMediaItem,
    ComfyPresetPackItem,
    LoadedModelSummary,
    LocalMediaItem,
    ModelFileItem,
    ModelModality,
    OpenModelSpec,
    ResolutionOption,
    build_default_comfy_preset_packs,
    format_duration,
    get_app_config_dir,
    get_lang,
    get_legacy_media_dirs,
    get_local_save_dir,
    is_en,
    localize_server_text,
    mask_email_display,
    mask_url_display,
    open_path_in_os,
    resolve_resource_path,
    set_lang,
    snap_video_frames,
    strip_emoji,
    tr,
)

WIN_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


from windows_app.viewmodel_services import ViewModelServicesMixin


class _AdminSession(requests.Session):
    """Validate the prepared destination before each send, including redirects."""

    def __init__(self, base_url: str):
        super().__init__()
        self.origin = self._https_origin(requests.Request("GET", base_url).prepare().url)

    @staticmethod
    def _https_origin(url: str):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None:
            raise requests.exceptions.InvalidURL("Management requests require an HTTPS origin")
        return parsed.hostname, 443 if parsed.port is None else parsed.port

    def send(self, request, **kwargs):
        if self._https_origin(request.url) != self.origin:
            raise requests.exceptions.InvalidURL("Management request left the connected HTTPS origin")
        return super().send(request, **kwargs)


class EasyColabViewModel(ViewModelServicesMixin, QObject):
    """Central state & async controller emitting Qt signals onto the GUI thread."""

    # Signals emitted exclusively on the Qt main thread via _dispatch_ui
    _ui_callback_sig = Signal(object)

    state_changed = Signal()
    telemetry_changed = Signal()
    console_changed = Signal()
    chat_changed = Signal()
    media_list_changed = Signal()
    media_selected = Signal(object, object, bool)  # (Path | None, elapsed: float | None, auto_play: bool)
    theme_changed = Signal()
    language_changed = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._ui_callback_sig.connect(self._execute_ui_callback)

        self._config_path = get_app_config_dir() / "settings.json"
        self._settings = self._load_settings_file()

        # ── Language Setting (Default MUST be Simplified Chinese "zh") ──
        self.language: str = "en" if str(self._settings.get("language", "zh")) == "en" else "zh"
        set_lang(self.language)

        # ── Persisted Settings (Windows defaults to Light Mode) ──
        self.dark_mode: bool = bool(self._settings.get("darkMode", False))
        self.mask_server_url: bool = bool(self._settings.get("maskServerURL", False))
        self.colab_server_url: str = str(self._settings.get("colabServerURL", ""))
        self.wake_gpu_choice: str = str(self._settings.get("wakeGPUChoiceV2", "G4"))
        if self.wake_gpu_choice not in {"G4", "H100", "A100", "L4", "T4"}:
            self.wake_gpu_choice = "G4"
        self.total_budget_cu: float = float(self._settings.get("totalBudgetCU", 2000.0))
        self.idle_timeout_minutes: int = int(self._settings.get("idleTimeoutMinutes", 30))
        if self.idle_timeout_minutes not in {0, 15, 30, 60}:
            self.idle_timeout_minutes = 30
        self.context_length_k: int = max(4, min(256, int(self._settings.get("qwenContextLengthK", 32))))
        self.reasoning_effort: str = str(self._settings.get("reasoningEffort", "medium"))
        if self.reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            self.reasoning_effort = "medium"
        self.api_key_enabled = True
        self.api_access_key: str = str(self._settings.get("apiAccessKey", "")).strip()
        if not self.api_access_key:
            self.api_access_key = "sk-colab-" + secrets.token_hex(10)
            self._save_setting("apiAccessKey", self.api_access_key)
        self.media_elapsed_map: Dict[str, float] = {
            str(k): float(v)
            for k, v in (self._settings.get("mediaElapsedSecondsMap") or {}).items()
            if isinstance(v, (int, float))
        }
        self.has_seen_onboarding: bool = bool(self._settings.get("hasSeenOnboardingGuide_v1", False))
        self.combined_chart_mode: bool = bool(self._settings.get("combinedTelemetryChartMode", False))
        self.hf_token: str = str(self._settings.get("hfToken", ""))

        # ── Account & Headless Colab Bridge State ──
        self.cli_logged_in: bool = False
        self.account_email: str = ""
        self.official_cu_balance: Optional[float] = None
        self.official_burn_rate: Optional[float] = None
        self.active_vm_count: int = 0
        self.show_oauth_code_input: bool = False
        self.is_waking_colab: bool = False
        self.is_releasing_colab: bool = False
        self._auth_busy: bool = False

        # ── Remote Runtime & Telemetry State ──
        self.is_connected: bool = False
        self.gpu_name: str = tr("尚未连接云端 GPU", "Cloud GPU Not Connected")
        self.vram_used_gb: float = 0.0
        self.vram_total_gb: float = 0.0
        self.vram_free_gb: float = 0.0
        self.ram_used_gb: float = 0.0
        self.ram_total_gb: float = 0.0
        self.ram_free_gb: float = 0.0
        self.disk_used_gb: float = 0.0
        self.disk_total_gb: float = 0.0
        self.disk_free_gb: float = 0.0
        self.shm_total_gb: float = 0.0
        self.shm_free_gb: float = 0.0
        self.drive_total_gb: float = 0.0
        self.drive_free_gb: float = 0.0
        self.drive_persisted: bool = False

        self.vram_history_pct: List[float] = []
        self.ram_history_pct: List[float] = []
        self.disk_history_pct: List[float] = []
        self._last_telemetry_sample_ts: float = 0.0

        self.uptime_seconds: int = 0
        self.cu_burn_rate_per_hour: float = 0.0
        self.session_cu_used: float = 0.0
        self.idle_remaining_seconds: int = 1800
        self.api_request_count: int = 0

        self._consecutive_status_failures: int = 0
        self._status_check_inflight: bool = False
        self._remote_context_synced: bool = False
        self._remote_idle_synced: bool = False
        self._remote_api_key_synced: bool = False

        # ── Model Selection, Weights & VRAM State ──
        self.custom_models: List[OpenModelSpec] = []
        self._load_persisted_custom_models()
        self.selected_model_id: str = "minimax_h3"
        self.task_mode_id: str = "t2va"
        self.selected_precision: str = "bf16"
        self.selected_offload: str = "disk"

        self.model_files: List[ModelFileItem] = [f.clone() for f in MODEL_SPEC_MAP["minimax_h3"].default_files]
        self.downloaded_count: int = 0
        self.total_count: int = len(self.model_files)
        self.all_downloaded: bool = False
        self.total_size_gb: float = 0.0
        self.is_downloading_weights: bool = False
        self.is_loading_model: bool = False
        self.model_task_progress: float = 0.0

        self.loaded_model_ids: List[str] = []
        self.loaded_models_detail: List[LoadedModelSummary] = []
        self.loaded_context_length: int = 32768
        self.current_model_name: str = ""

        # ── Generation & Sampling State ──
        self.visual_prompt: str = PRESET_TEMPLATES[0].visual
        self.soundscape_prompt: str = PRESET_TEMPLATES[0].soundscape
        self.music_prompt: str = PRESET_TEMPLATES[0].music
        self.negative_prompt: str = ""

        self.selected_video_res: ResolutionOption = VIDEO_RESOLUTIONS[0]
        self.selected_image_res: ResolutionOption = IMAGE_RESOLUTIONS[0]
        self.num_frames: int = 124
        self.inference_steps: int = 35
        self.seed: int = 42
        self.randomize_seed: bool = True
        self.batch_count: int = 1
        self.denoising_strength: float = 0.70

        self.enable_thinking: bool = True
        self.max_new_tokens: int = 16384
        self.temperature: float = 1.0
        self.top_p: float = 0.95
        self.presence_penalty: float = 0.0

        self.is_generating: bool = False
        self.progress_value: float = 0.0
        self.current_job_elapsed: float = 0.0
        self._job_start_local_ts: float = 0.0
        self._active_generation_token: str = ""

        # ── ComfyUI State ──
        self.comfy_running: bool = False
        self.comfy_starting: bool = False
        self.comfy_syncing_drive: bool = False
        self.comfy_downloading_weight: bool = False
        self.comfy_installing_pack_id: str = ""
        self.comfy_progress: float = 0.0
        self.comfy_status_message: str = tr("云端 ComfyUI 尚未启动", "Cloud ComfyUI not started")
        self.comfy_public_url: str = ""
        self.comfy_queue_running: int = 0
        self.comfy_queue_pending: int = 0
        self.comfy_persistence_label: str = tr("Google Drive 自动软链持久化", "Google Drive Auto-Symlink Persistence")
        self.comfy_custom_nodes_count: int = 0
        self.comfy_models_count: int = 0
        self.comfy_workflows_count: int = 0
        self.comfy_unload_others_on_start: bool = False
        self.comfy_recent_outputs: List[ComfyOutputMediaItem] = []
        self.comfy_preset_packs: List[ComfyPresetPackItem] = build_default_comfy_preset_packs()

        # ── Chat, Logs & Local Gallery ──
        self.chat_history: List[ChatTurnItem] = []
        self.status_message: str = tr(
            "就绪：请先绑定 Google 账号并唤醒 G4 实例，或粘贴已有隧道链接连接。",
            "Ready: Bind your Google account and wake a G4 runtime, or paste an existing tunnel URL.",
        )
        _plat_label = "Windows" if sys.platform == "win32" else ("macOS" if sys.platform == "darwin" else "Linux")
        self.live_console_lines: List[str] = [
            tr(f"EasyColab {_plat_label} 工作台已就绪。", f"EasyColab {_plat_label} Studio is ready."),
            tr(
                f"本地成片自动保存目录: {get_local_save_dir()}",
                f"Local media auto-save folder: {get_local_save_dir()}",
            ),
        ]
        self.local_media_items: List[LocalMediaItem] = []
        self.current_media_path: Optional[Path] = None

        # ── Timers ──
        self._tick_counter: int = 0
        self._bg_timer = QTimer(self)
        self._bg_timer.setInterval(2500)
        self._bg_timer.timeout.connect(self._on_background_tick)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(200)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

    # ─── Initialization & Lifecycle ──────────────────────────────────────────

    def start(self) -> None:
        self.refresh_local_media_gallery(auto_select_first=True)
        self.refresh_official_colab_account_usage()
        if self.normalized_base_url:
            self.check_colab_status(silent=True)
        self._bg_timer.start()

    def _dispatch_ui(self, fn: Callable[[], None]) -> None:
        self._ui_callback_sig.emit(fn)

    def _execute_ui_callback(self, fn: object) -> None:
        if callable(fn):
            fn()

    # ─── Settings Helpers ────────────────────────────────────────────────────

    def _load_settings_file(self) -> dict:
        if self._config_path.exists():
            try:
                return json.loads(self._config_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_setting(self, key: str, value: Any) -> None:
        self._settings[key] = value
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps(self._settings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def toggle_dark_mode(self) -> None:
        self.dark_mode = not self.dark_mode
        self._save_setting("darkMode", self.dark_mode)
        self.theme_changed.emit()

    def set_language(self, lang: str) -> None:
        target = "en" if str(lang).strip().lower().startswith("en") else "zh"
        if target == self.language and get_lang() == target:
            return
        self.language = target
        set_lang(self.language)
        self._save_setting("language", self.language)
        self._translate_default_runtime_strings()
        self.language_changed.emit()

    def toggle_language(self) -> None:
        self.set_language("zh" if self.language == "en" else "en")

    def _translate_default_runtime_strings(self) -> None:
        to_en = is_en()

        def swap(val: str, pairs: List[tuple[str, str]]) -> str:
            for zh_txt, en_txt in pairs:
                if to_en and val == zh_txt:
                    return en_txt
                if (not to_en) and val == en_txt:
                    return zh_txt
            return localize_server_text(val) if to_en else val

        self.gpu_name = swap(
            self.gpu_name,
            [("尚未连接云端 GPU", "Cloud GPU Not Connected")],
        )
        self.comfy_status_message = swap(
            self.comfy_status_message,
            [
                ("云端 ComfyUI 尚未启动", "Cloud ComfyUI not started"),
                (
                    "尚未启动（支持一键拉起 ComfyUI + Google Drive 自动软链持久化）",
                    "Not started (1-click ComfyUI + Google Drive symlink persistence)",
                ),
            ],
        )
        self.comfy_persistence_label = swap(
            self.comfy_persistence_label,
            [
                ("Google Drive 自动软链持久化", "Google Drive Auto-Symlink Persistence"),
                ("Google Drive 实时软链挂载", "Google Drive Live Symlink Mounted"),
                ("Google Drive 云盘自动持久化", "Google Drive Cloud Auto-Persistence"),
                ("实例本地高速存储", "Instance Local High-Speed Storage"),
            ],
        )
        self.status_message = swap(
            self.status_message,
            [
                (
                    "就绪：请先绑定 Google 账号并唤醒 G4 实例，或粘贴已有隧道链接连接。",
                    "Ready: Bind your Google account and wake a G4 runtime, or paste an existing tunnel URL.",
                ),
                (
                    "请先输入 https://xxxx.gradio.live 服务地址，或点击「唤醒实例并部署」。",
                    "Please enter a https://xxxx.gradio.live server URL or click 'Wake & Deploy'.",
                ),
                ("请先连接或唤醒 Colab 云端实例。", "Please connect or wake a Colab cloud runtime first."),
                (
                    "尚未连接云端 Colab 服务，请先唤醒或连接实例。",
                    "Not connected to Colab server. Please wake or connect a runtime first.",
                ),
                ("已退出当前 Google 账号。", "Signed out of Google account."),
                ("已释放全部云端实例。", "All cloud runtimes released."),
                ("云端实例已释放，本地状态已同步归零。", "Cloud runtime released; local state reset."),
                ("已生成新的随机 API Key。", "Generated a new random API Key."),
                ("请先输入提示词或对话内容。", "Please enter a prompt or message first."),
            ],
        )

        save_dir = get_local_save_dir()
        _plat_label = "Windows" if sys.platform == "win32" else ("macOS" if sys.platform == "darwin" else "Linux")
        line_pairs = [
            (f"EasyColab {_plat_label} 工作台已就绪。", f"EasyColab {_plat_label} Studio is ready."),
            ("EasyColab Windows 工作台已就绪。", "EasyColab Windows Studio is ready."),
            (f"本地成片自动保存目录: {save_dir}", f"Local media auto-save folder: {save_dir}"),
            ("已清空历史输出，正在监听云端实时日志...", "Console cleared. Listening for live cloud logs..."),
        ]
        self.live_console_lines = [swap(ln, line_pairs) for ln in self.live_console_lines]

        for d in self.loaded_models_detail:
            if d.id in MODEL_SPEC_MAP:
                d.name = MODEL_SPEC_MAP[d.id].name

        for tpl in PRESET_TEMPLATES:
            if tpl.visual_en and tpl.visual_en != tpl.visual_zh:
                if to_en and self.visual_prompt == tpl.visual_zh:
                    self.visual_prompt = tpl.visual_en
                    break
                if (not to_en) and self.visual_prompt == tpl.visual_en:
                    self.visual_prompt = tpl.visual_zh
                    break

    def set_mask_server_url(self, enabled: bool) -> None:
        self.mask_server_url = bool(enabled)
        self._save_setting("maskServerURL", self.mask_server_url)
        self.state_changed.emit()
        self.console_changed.emit()

    def set_colab_server_url(self, url: str) -> None:
        cleaned = (url or "").strip()
        if cleaned != self.colab_server_url:
            self._remote_api_key_synced = False
            self.colab_server_url = cleaned
            self._save_setting("colabServerURL", cleaned)
            self.state_changed.emit()

    def set_wake_gpu_choice(self, gpu: str) -> None:
        if gpu in {"G4", "H100", "A100", "L4", "T4"}:
            self.wake_gpu_choice = gpu
            self._save_setting("wakeGPUChoiceV2", gpu)
            self.state_changed.emit()

    def set_combined_chart_mode(self, combined: bool) -> None:
        self.combined_chart_mode = bool(combined)
        self._save_setting("combinedTelemetryChartMode", self.combined_chart_mode)
        self.telemetry_changed.emit()

    def mark_onboarding_seen(self) -> None:
        if not self.has_seen_onboarding:
            self.has_seen_onboarding = True
            self._save_setting("hasSeenOnboardingGuide_v1", True)

    def set_enable_thinking(self, enabled: bool) -> None:
        self.enable_thinking = bool(enabled)
        if self.enable_thinking:
            self.temperature = 1.0
            self.top_p = 0.95
            self.presence_penalty = 0.0
        else:
            self.temperature = 0.7
            self.top_p = 0.80
            self.presence_penalty = 1.5
        self.state_changed.emit()

    def set_reasoning_effort(self, effort: str) -> None:
        if effort in {"low", "medium", "high", "xhigh"}:
            self.reasoning_effort = effort
            self._save_setting("reasoningEffort", effort)
            self.state_changed.emit()

    def set_context_length_k(self, k_val: int, sync_remote: bool = True) -> None:
        clamped = max(4, min(256, int(k_val)))
        self.context_length_k = clamped
        self._save_setting("qwenContextLengthK", clamped)
        self.state_changed.emit()
        if sync_remote and self.is_connected:
            self.sync_context_length_to_colab()

    # ─── Computed Properties ─────────────────────────────────────────────────

    @property
    def admin_password(self) -> str:
        return self._settings.get("adminPasswordsByURL", {}).get(self.normalized_base_url, "")

    def set_admin_password(self, password: str) -> None:
        address = self.normalized_base_url
        if not address:
            return
        passwords = dict(self._settings.get("adminPasswordsByURL", {}))
        if password:
            passwords[address] = password
        else:
            passwords.pop(address, None)
        self._save_setting("adminPasswordsByURL", passwords)

    def _admin_request(self, method: str, url: str, **kwargs):
        address = self.normalized_base_url
        password = self._settings.get("adminPasswordsByURL", {}).get(address, "")
        if not password:
            raise ValueError("Enter the management password or use Wake & Deploy")
        with _AdminSession(address) as session:
            return session.request(method, url, auth=("easycolab", password), **kwargs)

    @property
    def normalized_base_url(self) -> str:
        raw = self.colab_server_url.strip()
        while raw.endswith("/"):
            raw = raw[:-1]
        return raw

    @property
    def all_available_models(self) -> List[OpenModelSpec]:
        return OPEN_MODEL_CATALOG + self.custom_models

    @property
    def active_model_spec(self) -> OpenModelSpec:
        return MODEL_SPEC_MAP.get(self.selected_model_id, OPEN_MODEL_CATALOG[0])

    @property
    def context_length_tokens(self) -> int:
        return self.context_length_k * 1024

    @property
    def is_selected_model_loaded_in_vram(self) -> bool:
        return self.selected_model_id in self.loaded_model_ids

    @property
    def is_qwen_loaded_in_vram(self) -> bool:
        return ("qwen38_27b_gguf" in self.loaded_model_ids) or ("qwen38_27b" in self.loaded_model_ids)

    @property
    def display_account_label(self) -> str:
        if not self.cli_logged_in:
            return tr("未绑定", "Not bound")
        if self.account_email:
            return mask_email_display(self.account_email, self.mask_server_url)
        return tr("已连接", "Connected")

    @property
    def display_remaining_cu(self) -> float:
        if self.official_cu_balance is not None:
            return max(0.0, self.official_cu_balance)
        return max(0.0, self.total_budget_cu - self.session_cu_used)

    @property
    def display_burn_rate(self) -> float:
        if self.official_burn_rate is not None:
            return max(0.0, self.official_burn_rate)
        return max(0.0, self.cu_burn_rate_per_hour)

    def masked_url(self, text: str) -> str:
        return mask_url_display(text, self.mask_server_url)

    def append_console_line(self, line: str) -> None:
        clean = strip_emoji(self.masked_url(line))
        if not clean:
            return
        self.live_console_lines.append(clean)
        if len(self.live_console_lines) > 400:
            self.live_console_lines = self.live_console_lines[-400:]
        self.console_changed.emit()

    def clear_console_lines(self) -> None:
        self.live_console_lines = [
            tr("已清空历史输出，正在监听云端实时日志...", "Console cleared. Listening for live cloud logs...")
        ]
        self.console_changed.emit()

    # ─── Local Media Gallery Management ──────────────────────────────────────

    def refresh_local_media_gallery(self, auto_select_first: bool = False) -> None:
        exts = {".mp4", ".png", ".jpg", ".jpeg", ".webp"}
        seen_names = set()
        found: List[LocalMediaItem] = []

        for folder in get_legacy_media_dirs():
            if not folder.exists():
                continue
            try:
                for entry in folder.iterdir():
                    if not entry.is_file() or entry.suffix.lower() not in exts:
                        continue
                    if entry.name in seen_names:
                        continue
                    seen_names.add(entry.name)
                    try:
                        stat = entry.stat()
                        ctime = stat.st_mtime
                    except Exception:
                        ctime = 0.0
                    found.append(
                        LocalMediaItem(
                            path=entry,
                            filename=entry.name,
                            is_image=(entry.suffix.lower() != ".mp4"),
                            created_ts=ctime,
                            elapsed_seconds=self.media_elapsed_map.get(entry.name),
                        )
                    )
            except Exception:
                pass

        found.sort(key=lambda x: x.created_ts, reverse=True)
        self.local_media_items = found
        self.media_list_changed.emit()

        if auto_select_first and self.current_media_path is None and found:
            self.select_local_media(found[0].path, auto_play=False)

    def select_local_media(self, path: Optional[Path], auto_play: bool = True) -> None:
        self.current_media_path = path
        elapsed = self.media_elapsed_map.get(path.name) if path else None
        self.media_selected.emit(path, elapsed, auto_play)

    def export_or_reveal_notebook(self) -> None:
        nb_src = resolve_resource_path("EasyColab_Server.ipynb")
        if not nb_src.exists():
            self.status_message = tr(
                "未找到内置 EasyColab_Server.ipynb 文件。",
                "Built-in EasyColab_Server.ipynb file not found.",
            )
            self.state_changed.emit()
            return
        # If running inside a frozen bundle (_internal), copy to Downloads so user can easily drag into browser
        if getattr(sys, "frozen", False):
            dl_dir = Path.home() / "Downloads"
            dl_dir.mkdir(parents=True, exist_ok=True)
            dst = dl_dir / "EasyColab_Server.ipynb"
            try:
                shutil.copy2(nb_src, dst)
                nb_src = dst
            except Exception:
                pass
        self.status_message = tr(f"已定位笔记本文件：{nb_src}", f"Located notebook file: {nb_src}")
        self.state_changed.emit()
        open_path_in_os(nb_src, reveal=True)


    # ─── Background 2.5s Telemetry & Watchdog Loop ───────────────────────────

    def _on_elapsed_tick(self) -> None:
        if self.is_generating and self._job_start_local_ts > 0:
            local_el = time.time() - self._job_start_local_ts
            if local_el > self.current_job_elapsed:
                self.current_job_elapsed = local_el
                self.state_changed.emit()
        elif not self.is_generating:
            self._elapsed_timer.stop()

    def _on_background_tick(self) -> None:
        self._tick_counter += 1
        tick = self._tick_counter

        if self.is_connected:
            if not self.is_generating:
                self.check_colab_status(silent=True)
            elif tick % 6 == 0:
                self.check_colab_status(silent=True)
        elif self.normalized_base_url and (tick % 4 == 0):
            self.check_colab_status(silent=True)

        # Smart Keepalive & Official CU Ledger Sync
        if tick % 20 == 0 and self.cli_logged_in and not self.is_waking_colab:
            should_keepalive = self.is_connected and (
                self.is_generating
                or self.is_downloading_weights
                or self.is_loading_model
                or self.comfy_running
                or self.comfy_starting
                or self.comfy_downloading_weight
                or self.comfy_queue_running > 0
                or self.idle_timeout_minutes <= 0
                or self.idle_remaining_seconds > 75
            )
            if should_keepalive:
                self.keepalive_colab_runtime()
            else:
                self.refresh_official_colab_account_usage()
        elif tick % 16 == 0 and self.cli_logged_in and not self.is_connected:
            self.refresh_official_colab_account_usage()

    # ─── Remote Status & Telemetry Handling ──────────────────────────────────

    def reset_stale_remote_runtime_state(self, message: str = "") -> None:
        self.is_connected = False
        self.gpu_name = tr("尚未连接云端 GPU", "Cloud GPU Not Connected")
        self.vram_used_gb = 0.0
        self.vram_total_gb = 0.0
        self.vram_free_gb = 0.0
        self.ram_used_gb = 0.0
        self.ram_total_gb = 0.0
        self.ram_free_gb = 0.0
        self.disk_used_gb = 0.0
        self.disk_total_gb = 0.0
        self.disk_free_gb = 0.0
        self.shm_total_gb = 0.0
        self.shm_free_gb = 0.0
        self.drive_total_gb = 0.0
        self.drive_free_gb = 0.0
        self.drive_persisted = False
        self.vram_history_pct.clear()
        self.ram_history_pct.clear()
        self.disk_history_pct.clear()
        self.loaded_model_ids.clear()
        self.loaded_models_detail.clear()
        self.is_downloading_weights = False
        self.is_loading_model = False
        self.is_generating = False
        self.comfy_running = False
        self.comfy_starting = False
        self.comfy_public_url = ""
        self.cu_burn_rate_per_hour = 0.0
        self._remote_context_synced = False
        self._remote_idle_synced = False
        self._remote_api_key_synced = False
        if message:
            self.status_message = strip_emoji(self.masked_url(message))
        self.state_changed.emit()
        self.telemetry_changed.emit()

    def _record_telemetry_sample(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_telemetry_sample_ts) < 0.85:
            return
        self._last_telemetry_sample_ts = now

        def push(arr: List[float], used: float, total: float) -> None:
            if total <= 0:
                return
            pct = max(0.0, min(1.0, used / total))
            if not arr:
                arr.extend([pct] * 14)
            else:
                arr.append(pct)
                if len(arr) > 48:
                    del arr[: len(arr) - 48]

        push(self.vram_history_pct, self.vram_used_gb, self.vram_total_gb)
        push(self.ram_history_pct, self.ram_used_gb, self.ram_total_gb)
        push(self.disk_history_pct, self.disk_used_gb, self.disk_total_gb)

    def _apply_live_telemetry_snapshot(self, tel: dict) -> None:
        if not isinstance(tel, dict):
            return
        if tel.get("gpu_name"):
            self.gpu_name = strip_emoji(str(tel["gpu_name"]))
        for attr, key in (
            ("vram_free_gb", "vram_free_gb"),
            ("vram_total_gb", "vram_total_gb"),
            ("vram_used_gb", "vram_used_gb"),
            ("ram_total_gb", "ram_total_gb"),
            ("ram_used_gb", "ram_used_gb"),
            ("ram_free_gb", "ram_free_gb"),
            ("disk_total_gb", "disk_total_gb"),
            ("disk_used_gb", "disk_used_gb"),
            ("disk_free_gb", "disk_free_gb"),
            ("shm_total_gb", "shm_total_gb"),
            ("shm_free_gb", "shm_free_gb"),
            ("drive_total_gb", "drive_total_gb"),
            ("drive_free_gb", "drive_free_gb"),
            ("cu_burn_rate_per_hour", "cu_burn_rate_per_hour"),
        ):
            if tel.get(key) is not None:
                setattr(self, attr, float(tel[key]))
        self._record_telemetry_sample(force=False)
        self.telemetry_changed.emit()

    def _apply_status_response(self, d: dict, update_status_msg: bool = False) -> None:
        self._consecutive_status_failures = 0
        was_connected = self.is_connected
        self.is_connected = True

        if d.get("gpu_name"):
            self.gpu_name = strip_emoji(str(d["gpu_name"]))
        for attr in (
            "vram_free_gb",
            "vram_total_gb",
            "vram_used_gb",
            "ram_total_gb",
            "ram_used_gb",
            "ram_free_gb",
            "disk_total_gb",
            "disk_used_gb",
            "disk_free_gb",
            "shm_total_gb",
            "shm_free_gb",
            "drive_total_gb",
            "drive_free_gb",
            "cu_burn_rate_per_hour",
            "session_cu_used",
            "total_size_gb",
            "task_progress",
        ):
            if d.get(attr) is not None:
                val = float(d[attr])
                if attr == "task_progress":
                    self.model_task_progress = val
                else:
                    setattr(self, attr, val)

        if d.get("uptime_seconds") is not None:
            self.uptime_seconds = int(d["uptime_seconds"])
        if d.get("idle_remaining_seconds") is not None:
            self.idle_remaining_seconds = int(d["idle_remaining_seconds"])
        if d.get("api_request_count") is not None:
            self.api_request_count = int(d["api_request_count"])
        if d.get("drive_persisted") is not None:
            self.drive_persisted = bool(d["drive_persisted"])
        if d.get("loaded_context_length") is not None:
            self.loaded_context_length = int(d["loaded_context_length"])

        if isinstance(d.get("custom_models"), list):
            self._merge_remote_custom_models(d["custom_models"])

        if isinstance(d.get("loaded_model_ids"), list):
            self.loaded_model_ids = [str(x) for x in d["loaded_model_ids"]]
        elif d.get("loaded_model_id"):
            self.loaded_model_ids = [str(d["loaded_model_id"])]
        elif d.get("any_model_loaded") is False:
            self.loaded_model_ids = []

        if isinstance(d.get("loaded_models_detail"), list):
            self.loaded_models_detail = [LoadedModelSummary.from_dict(x) for x in d["loaded_models_detail"]]

        if d.get("current_model"):
            self.current_model_name = strip_emoji(str(d["current_model"]))

        # Update files list only if response matches currently selected model
        resp_model = d.get("selected_model_id")
        if (not resp_model or resp_model == self.selected_model_id) and isinstance(d.get("files"), list):
            self.model_files = [ModelFileItem.from_dict(x) for x in d["files"]]
            self.downloaded_count = int(d.get("downloaded_count", sum(1 for f in self.model_files if f.exists)))
            self.total_count = int(d.get("total_count", len(self.model_files)))
            self.all_downloaded = bool(d.get("all_downloaded", self.total_count > 0 and self.downloaded_count >= self.total_count))

        if d.get("is_downloading") is not None:
            self.is_downloading_weights = bool(d["is_downloading"])
        if d.get("is_loading_vram") is not None:
            self.is_loading_model = bool(d["is_loading_vram"])

        # Console lines from server
        if isinstance(d.get("console_lines"), list) and d["console_lines"]:
            cleaned_lines = [
                strip_emoji(self.masked_url(str(ln)))
                for ln in d["console_lines"]
                if str(ln).strip()
            ]
            if cleaned_lines:
                self.live_console_lines = cleaned_lines[-400:]
                self.console_changed.emit()

        # ComfyUI status object
        comfy = d.get("comfyui")
        if isinstance(comfy, dict):
            self.comfy_running = bool(comfy.get("is_running", False))
            self.comfy_starting = bool(comfy.get("is_starting", False))
            self.comfy_syncing_drive = bool(comfy.get("is_syncing_drive", False))
            self.comfy_downloading_weight = bool(comfy.get("is_downloading_weight", False))
            self.comfy_installing_pack_id = str(comfy.get("installing_pack_id") or "")
            self.comfy_progress = float(comfy.get("progress") or 0.0)
            if comfy.get("status_message"):
                self.comfy_status_message = localize_server_text(
                    strip_emoji(self.masked_url(str(comfy["status_message"])))
                )
            if comfy.get("public_url") is not None:
                self.comfy_public_url = str(comfy.get("public_url") or "")
            self.comfy_queue_running = int(comfy.get("queue_running") or 0)
            self.comfy_queue_pending = int(comfy.get("queue_pending") or 0)
            if comfy.get("persistence_label"):
                self.comfy_persistence_label = localize_server_text(strip_emoji(str(comfy["persistence_label"])))
            if comfy.get("drive_persisted") is not None:
                self.drive_persisted = self.drive_persisted or bool(comfy["drive_persisted"])
            self.comfy_custom_nodes_count = int(comfy.get("custom_nodes_count") or 0)
            self.comfy_models_count = int(comfy.get("models_count") or 0)
            self.comfy_workflows_count = int(comfy.get("workflows_count") or 0)
            if isinstance(comfy.get("recent_outputs"), list):
                self.comfy_recent_outputs = [ComfyOutputMediaItem.from_dict(x) for x in comfy["recent_outputs"]]
            if isinstance(comfy.get("preset_packs"), list) and comfy["preset_packs"]:
                self.comfy_preset_packs = [ComfyPresetPackItem.from_dict(x) for x in comfy["preset_packs"]]

        if d.get("action_message"):
            self.status_message = localize_server_text(strip_emoji(self.masked_url(str(d["action_message"]))))
        elif d.get("task_message") and (self.is_downloading_weights or self.is_loading_model):
            self.status_message = localize_server_text(strip_emoji(self.masked_url(str(d["task_message"]))))
        elif update_status_msg and d.get("message"):
            self.status_message = localize_server_text(strip_emoji(self.masked_url(str(d["message"]))))
        elif not was_connected:
            self.status_message = tr(f"已连接云端 {self.gpu_name}！", f"Connected to cloud {self.gpu_name}!")

        # One-time per-connection remote config syncs
        if not self._remote_context_synced:
            self._remote_context_synced = True
            if d.get("context_length") != self.context_length_tokens:
                self.sync_context_length_to_colab()
        if not self._remote_idle_synced:
            self._remote_idle_synced = True
            if d.get("idle_timeout_minutes") != self.idle_timeout_minutes:
                self.update_idle_timeout_config(self.idle_timeout_minutes)
        if not d.get("api_key_configured", False) and not self._remote_api_key_synced:
            self._remote_api_key_synced = True
            self.sync_api_key_to_colab(initialize_only=True)

        self._record_telemetry_sample(force=not was_connected)
        self.state_changed.emit()
        self.telemetry_changed.emit()

    def check_colab_status(self, silent: bool = False) -> None:
        base = self.normalized_base_url
        if not base:
            if not silent:
                self.status_message = tr(
                    "请先输入 https://xxxx.gradio.live 服务地址，或点击「唤醒实例并部署」。",
                    "Please enter a https://xxxx.gradio.live server URL or click 'Wake & Deploy'.",
                )
                self.state_changed.emit()
            return

        if silent and self._status_check_inflight:
            return
        self._status_check_inflight = True
        target_model = self.selected_model_id
        timeout_s = 28 if self.is_generating else (20 if silent else 22)

        def worker() -> None:
            try:
                r = self._admin_request("GET",
                    f"{base}/minimax_api/status",
                    params={"model_id": target_model},
                    timeout=timeout_s,
                )
                if r.status_code == 200:
                    payload = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(payload, update_status_msg=not silent))
                elif r.status_code == 404:

                    def on_404() -> None:
                        self._consecutive_status_failures += 1 if self.is_generating else 2
                        if not silent or self._consecutive_status_failures >= 6:
                            self.reset_stale_remote_runtime_state(
                                tr(
                                    "云端隧道地址已失效或实例已释放，请重新唤醒部署。",
                                    "Cloud tunnel expired or runtime terminated. Please wake & deploy again.",
                                )
                            )

                    self._dispatch_ui(on_404)
                else:
                    code = r.status_code

                    def on_http_err() -> None:
                        self._consecutive_status_failures += 1
                        limit = 8 if self.is_generating else 5
                        if not silent or self._consecutive_status_failures >= limit:
                            self.reset_stale_remote_runtime_state(
                                tr(f"连接云端服务异常 (HTTP {code})。", f"Cloud server error (HTTP {code}).")
                            )

                    self._dispatch_ui(on_http_err)
            except Exception as exc:
                err_str = str(exc)

                def on_net_err() -> None:
                    self._consecutive_status_failures += 1
                    limit = 8 if self.is_generating else 5
                    if not silent or self._consecutive_status_failures >= limit:
                        self.reset_stale_remote_runtime_state(
                            tr(f"无法连接云端隧道：{err_str[:120]}", f"Cannot reach cloud tunnel: {err_str[:120]}")
                        )

                self._dispatch_ui(on_net_err)
            finally:
                self._status_check_inflight = False

        threading.Thread(target=worker, daemon=True).start()


    # ─── Model Selection, Weight Download, Load & Unload ─────────────────────

    def select_model(self, model_id: str) -> None:
        if model_id not in MODEL_SPEC_MAP:
            return
        spec = MODEL_SPEC_MAP[model_id]
        self.selected_model_id = model_id
        if spec.modes:
            self.task_mode_id = spec.modes[0].id
        self.inference_steps = spec.default_steps
        self.model_files = [f.clone() for f in spec.default_files]
        self.downloaded_count = sum(1 for f in self.model_files if f.exists)
        self.total_count = len(self.model_files)
        self.all_downloaded = False
        self.state_changed.emit()

        base = self.normalized_base_url
        if self.is_connected and base and spec.modality != ModelModality.COMFY_UI:

            def worker() -> None:
                try:
                    r = self._admin_request("POST",
                        f"{base}/minimax_api/select_model",
                        json={"model_id": model_id},
                        timeout=30,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                except Exception:
                    pass

            threading.Thread(target=worker, daemon=True).start()
        elif self.is_connected and base:
            self.check_colab_status(silent=True)

    def set_task_mode(self, mode_id: str) -> None:
        self.task_mode_id = mode_id
        self.state_changed.emit()

    def download_model_weights(self) -> None:
        base = self.normalized_base_url
        if not self.is_connected or not base:
            self.status_message = tr("请先连接或唤醒 Colab 云端实例。", "Please connect or wake a Colab cloud runtime first.")
            self.state_changed.emit()
            return

        target_model = self.selected_model_id
        self.is_downloading_weights = True
        self.model_task_progress = 0.01
        self.status_message = tr(
            f"正在云端启动 8 线程极速下载 {self.active_model_spec.name} 权重...",
            f"Starting 8-thread cloud download for {self.active_model_spec.name} weights...",
        )
        self.state_changed.emit()

        def worker() -> None:
            try:
                r = self._admin_request("POST",
                    f"{base}/minimax_api/download_model",
                    json={"model_id": target_model, "hf_token": self.hf_token.strip()},
                    timeout=60,
                )
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                while True:
                    time.sleep(1.5)
                    sr = self._admin_request("GET",
                        f"{base}/minimax_api/status",
                        params={"model_id": target_model},
                        timeout=25,
                    )
                    if sr.status_code != 200:
                        break
                    sdata = sr.json()
                    self._dispatch_ui(lambda sd=sdata: self._apply_status_response(sd, update_status_msg=True))
                    if not sdata.get("is_downloading", False):
                        break
            except Exception as exc:
                msg = tr(f"下载权重请求异常：{exc}", f"Weight download request error: {exc}")
                self._dispatch_ui(lambda: self._set_status_and_log(msg))
            finally:

                def done() -> None:
                    self.is_downloading_weights = False
                    self.state_changed.emit()

                self._dispatch_ui(done)

        threading.Thread(target=worker, daemon=True).start()

    def load_model_to_vram(self) -> None:
        base = self.normalized_base_url
        if not self.is_connected or not base:
            self.status_message = tr("请先连接或唤醒 Colab 云端实例。", "Please connect or wake a Colab cloud runtime first.")
            self.state_changed.emit()
            return
        if not self.all_downloaded:
            self.status_message = tr(
                "请先点击「下载权重」将模型分片拉取到云盘/实例磁盘。",
                "Please click 'Download Weights' to fetch model shards first.",
            )
            self.state_changed.emit()
            return

        target_model = self.selected_model_id
        self.is_loading_model = True
        self.model_task_progress = 0.05
        self.status_message = tr(
            f"正在将 {self.active_model_spec.name} 装载至云端 GPU 显存...",
            f"Loading {self.active_model_spec.name} into cloud GPU VRAM...",
        )
        self.state_changed.emit()

        payload = {
            "model_id": target_model,
            "precision_mode": self.selected_precision,
            "offload_mode": self.selected_offload,
            "context_length": self.context_length_tokens,
            "hf_token": self.hf_token.strip(),
        }

        def worker() -> None:
            try:
                r = self._admin_request("POST", f"{base}/minimax_api/load_model", json=payload, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                while True:
                    time.sleep(1.5)
                    sr = self._admin_request("GET",
                        f"{base}/minimax_api/status",
                        params={"model_id": target_model},
                        timeout=25,
                    )
                    if sr.status_code != 200:
                        break
                    sdata = sr.json()
                    self._dispatch_ui(lambda sd=sdata: self._apply_status_response(sd, update_status_msg=True))
                    if not sdata.get("is_loading_vram", False):
                        break
            except Exception as exc:
                msg = tr(f"装载显存请求异常：{exc}", f"Load to VRAM request error: {exc}")
                self._dispatch_ui(lambda: self._set_status_and_log(msg))
            finally:

                def done() -> None:
                    self.is_loading_model = False
                    self.state_changed.emit()

                self._dispatch_ui(done)

        threading.Thread(target=worker, daemon=True).start()

    def _clear_local_loaded_record(self, model_id: str) -> None:
        if model_id == "all":
            self.loaded_model_ids.clear()
            self.loaded_models_detail.clear()
        else:
            self.loaded_model_ids = [m for m in self.loaded_model_ids if m != model_id]
            self.loaded_models_detail = [m for m in self.loaded_models_detail if m.id != model_id]

    def unload_model_from_vram(self, model_id: Optional[str] = None) -> None:
        target = model_id or self.selected_model_id
        base = self.normalized_base_url
        if not base:
            self._clear_local_loaded_record(target)
            self.state_changed.emit()
            return

        self.status_message = tr("正在从 GPU 显存卸载模型并回收缓存...", "Unloading model from GPU VRAM & freeing cache...")
        self.state_changed.emit()

        def worker() -> None:
            try:
                r = self._admin_request("POST",
                    f"{base}/minimax_api/unload_model",
                    json={"model_id": target},
                    timeout=20,
                )
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                    return
            except Exception:
                pass

            def fallback() -> None:
                self._clear_local_loaded_record(target)
                self.status_message = tr("已清除本地模型显存驻留记录。", "Cleared local VRAM residency record.")
                self.state_changed.emit()

            self._dispatch_ui(fallback)

        threading.Thread(target=worker, daemon=True).start()


    # ─── Generation & Job Polling State Machine ──────────────────────────────

    def start_generation(
        self,
        first_frame_b64: Optional[str] = None,
        last_frame_b64: Optional[str] = None,
    ) -> None:
        if self.is_generating:
            return

        spec = self.active_model_spec
        modality = spec.modality
        prompt = self.visual_prompt.strip()
        if not prompt:
            self.status_message = tr("请先输入提示词或对话内容。", "Please enter a prompt or message first.")
            self.state_changed.emit()
            return

        if self.task_mode_id == "fl2va" and not first_frame_b64 and not last_frame_b64:
            self.status_message = tr(
                "「首尾帧生视频」模式请至少上传起始首帧或结束尾帧参考图。",
                "First/Last-Frame Video mode requires at least a start or end reference frame.",
            )
            self.state_changed.emit()
            return
        if self.task_mode_id == "i2i_edit" and not first_frame_b64 and not last_frame_b64:
            self.status_message = tr(
                "指令修图/重绘模式请至少上传一张参考原图。",
                "Instruction Edit / Img2Img mode requires at least one reference image.",
            )
            self.state_changed.emit()
            return

        base = self.normalized_base_url
        if not self.is_connected or not base:
            self.status_message = tr(
                "尚未连接云端 Colab 服务，请先唤醒或连接实例。",
                "Not connected to Colab server. Please wake or connect a runtime first.",
            )
            self.state_changed.emit()
            return
        if not self.is_selected_model_loaded_in_vram:
            self.status_message = tr(
                f"请先在左侧模型中心点击「加载到显存」装载 {spec.name}。",
                f"Please click 'Load to VRAM' in the Model Center to load {spec.name} first.",
            )
            self.state_changed.emit()
            return

        token = str(uuid.uuid4())
        self._active_generation_token = token
        self.is_generating = True
        self.progress_value = 0.01
        self.current_job_elapsed = 0.0
        self._job_start_local_ts = time.time()
        self._elapsed_timer.start()

        width, height = (
            (self.selected_image_res.width, self.selected_image_res.height)
            if modality == ModelModality.IMAGE
            else (self.selected_video_res.width, self.selected_video_res.height)
        )

        assistant_idx: Optional[int] = None
        chat_messages_payload: Optional[List[Dict[str, str]]] = None
        if modality == ModelModality.LLM_VLM:
            has_img = bool(first_frame_b64 or last_frame_b64) and self.task_mode_id == "vlm"
            self.chat_history.append(ChatTurnItem(role="user", content=prompt, has_image=has_img))
            chat_messages_payload = [{"role": t.role, "content": t.content} for t in self.chat_history]
            self.chat_history.append(ChatTurnItem(role="assistant", content="", thinking=""))
            assistant_idx = len(self.chat_history) - 1
            self.visual_prompt = ""
            self.chat_changed.emit()

        payload: Dict[str, Any] = {
            "model_id": self.selected_model_id,
            "task_mode": self.task_mode_id,
            "visual_prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "soundscape_prompt": self.soundscape_prompt,
            "music_prompt": self.music_prompt,
            "width": width,
            "height": height,
            "num_frames": snap_video_frames(self.num_frames),
            "num_inference_steps": int(self.inference_steps),
            "fps": 24,
            "seed": int(self.seed),
            "randomize_seed": bool(self.randomize_seed),
            "batch_count": 1 if modality == ModelModality.LLM_VLM else int(self.batch_count),
            "denoising_strength": float(self.denoising_strength),
            "enable_thinking": bool(self.enable_thinking),
            "max_new_tokens": int(self.max_new_tokens),
            "context_length": self.context_length_tokens,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": float(self.presence_penalty),
            "repeat_penalty": 1.0,
            "reasoning_effort": self.reasoning_effort,
        }
        if chat_messages_payload is not None:
            payload["chat_messages"] = chat_messages_payload
        if self.task_mode_id in {"fl2va", "i2v", "i2i_edit", "vlm"}:
            if first_frame_b64:
                payload["first_frame_base64"] = first_frame_b64
            if last_frame_b64:
                payload["last_frame_base64"] = last_frame_b64

        self.status_message = tr(
            f"正在向云端提交 {spec.name} 推理任务...",
            f"Submitting {spec.name} inference job to cloud...",
        )
        self.state_changed.emit()

        def worker() -> None:
            try:
                r = self._admin_request("POST", f"{base}/minimax_api/generate", json=payload, timeout=30)
                if r.status_code != 200:
                    err_txt = r.text[:200]
                    raise RuntimeError(f"HTTP {r.status_code}: {err_txt}")
                job = r.json()
                if job.get("status") == "failed":
                    msg = localize_server_text(
                        strip_emoji(str(job.get("message") or tr("任务提交失败", "Failed to submit job")))
                    )

                    def on_fail() -> None:
                        if self._active_generation_token != token:
                            return
                        if assistant_idx is not None and assistant_idx < len(self.chat_history):
                            self.chat_history[assistant_idx].content = msg
                            self.chat_changed.emit()
                        self.is_generating = False
                        self.progress_value = 0.0
                        self.status_message = msg
                        self.state_changed.emit()

                    self._dispatch_ui(on_fail)
                    return

                job_id = str(job.get("job_id") or "")
                self._poll_colab_job(base, job_id, token, assistant_idx)
            except Exception as exc:
                err_msg = tr(f"提交推理任务失败：{exc}", f"Failed to submit inference job: {exc}")

                def on_exc() -> None:
                    if self._active_generation_token != token:
                        return
                    if assistant_idx is not None and assistant_idx < len(self.chat_history):
                        self.chat_history[assistant_idx].content = err_msg
                        self.chat_changed.emit()
                    self.is_generating = False
                    self.progress_value = 0.0
                    self.status_message = err_msg
                    self.state_changed.emit()

                self._dispatch_ui(on_exc)

        threading.Thread(target=worker, daemon=True).start()

    def _download_media_to_local(self, base_url: str, rel_or_abs_url: str, filename: str, elapsed: Optional[float]) -> Optional[Path]:
        full_url = rel_or_abs_url if rel_or_abs_url.startswith("http") else f"{base_url}{rel_or_abs_url}"
        save_dir = get_local_save_dir()
        safe_name = Path(filename).name
        dst = save_dir / safe_name
        r = self._admin_request("GET", full_url, timeout=180)
        if r.status_code == 200 and r.content:
            dst.write_bytes(r.content)
            if elapsed is not None and elapsed > 0:
                self.media_elapsed_map[safe_name] = float(elapsed)
                self._save_setting("mediaElapsedSecondsMap", self.media_elapsed_map)
            return dst
        return None

    def _poll_colab_job(
        self,
        base: str,
        job_id: str,
        token: str,
        assistant_idx: Optional[int],
    ) -> None:
        poll_interval = 0.6 if assistant_idx is not None else 1.0
        consecutive_404 = 0
        consecutive_errors = 0
        downloaded_filenames: set[str] = set()

        while self._active_generation_token == token:
            time.sleep(poll_interval)
            if self._active_generation_token != token:
                return

            try:
                r = self._admin_request("GET", f"{base}/minimax_api/job/{job_id}", timeout=30)
                if r.status_code == 404:
                    consecutive_404 += 1
                    if consecutive_404 >= 5:
                        self._dispatch_ui(
                            lambda: self.reset_stale_remote_runtime_state(
                                tr(
                                    "云端推理进程已中断（可能因显存溢出或实例释放），请重新唤醒实例。",
                                    "Cloud inference process interrupted (possible OOM or runtime release). Please wake the runtime again.",
                                )
                            )
                        )
                        return
                    continue
                if r.status_code != 200:
                    consecutive_errors += 1
                    if consecutive_errors >= 12:

                        def on_abort() -> None:
                            if self._active_generation_token == token:
                                self.is_generating = False
                                self.status_message = tr(
                                    "轮询云端任务多次失败，请检查网络连接。",
                                    "Polling cloud job failed repeatedly. Please check your network connection.",
                                )
                                self.state_changed.emit()

                        self._dispatch_ui(on_abort)
                        return
                    continue

                consecutive_404 = 0
                consecutive_errors = 0
                job = r.json()
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 12:

                    def on_net_abort() -> None:
                        if self._active_generation_token == token:
                            self.is_generating = False
                            self.status_message = tr(
                                "网络连接连续超时，已停止轮询。",
                                "Network connection timed out repeatedly. Stopped polling.",
                            )
                            self.state_changed.emit()

                    self._dispatch_ui(on_net_abort)
                    return
                continue

            # Progressive media download on worker thread before final UI update
            newly_saved: List[tuple[Path, Optional[float]]] = []
            for item in job.get("completed_videos") or []:
                fname = str(item.get("filename") or "")
                vurl = str(item.get("video_url") or "")
                el = float(item["elapsed"]) if item.get("elapsed") is not None else None
                if fname and vurl and fname not in downloaded_filenames:
                    try:
                        saved = self._download_media_to_local(base, vurl, fname, el)
                        if saved:
                            downloaded_filenames.add(fname)
                            newly_saved.append((saved, el))
                    except Exception:
                        pass

            status = str(job.get("status") or "")
            if status == "completed":
                fname = str(job.get("filename") or "")
                vurl = str(job.get("video_url") or "")
                el = float(job["elapsed"]) if job.get("elapsed") is not None else None
                if fname and vurl and fname not in downloaded_filenames:
                    try:
                        saved = self._download_media_to_local(base, vurl, fname, el)
                        if saved:
                            downloaded_filenames.add(fname)
                            newly_saved.append((saved, el))
                    except Exception:
                        pass

            def update_ui(
                j: dict = job,
                st: str = status,
                saved_files: List[tuple[Path, Optional[float]]] = newly_saved,
            ) -> None:
                if self._active_generation_token != token:
                    return

                if isinstance(j.get("telemetry"), dict):
                    self._apply_live_telemetry_snapshot(j["telemetry"])

                raw_prog = float(j.get("progress") or 0.0)
                if st != "completed":
                    self.progress_value = min(0.98, max(0.0, raw_prog))
                if j.get("message"):
                    self.status_message = localize_server_text(strip_emoji(self.masked_url(str(j["message"]))))
                if j.get("elapsed") is not None:
                    self.current_job_elapsed = max(self.current_job_elapsed, float(j["elapsed"]))
                if j.get("seed") is not None:
                    self.seed = int(j["seed"])

                if isinstance(j.get("console_lines"), list) and j["console_lines"]:
                    self.live_console_lines = [
                        strip_emoji(self.masked_url(str(ln)))
                        for ln in j["console_lines"]
                        if str(ln).strip()
                    ][-400:]
                    self.console_changed.emit()

                if assistant_idx is not None and assistant_idx < len(self.chat_history):
                    turn = self.chat_history[assistant_idx]
                    if j.get("thinking_output") is not None:
                        turn.thinking = str(j["thinking_output"])
                    if j.get("text_output") is not None:
                        turn.content = str(j["text_output"])
                    self.chat_changed.emit()

                if saved_files:
                    self.refresh_local_media_gallery(auto_select_first=False)
                    latest_path, latest_el = saved_files[-1]
                    self.select_local_media(latest_path, auto_play=True)

                if st == "completed":
                    self.is_generating = False
                    self.progress_value = 1.0
                    b_cnt = int(j.get("batch_count") or 1)
                    if b_cnt > 1 and saved_files:
                        self.status_message = tr(
                            f"批量生成全部完成（共 {b_cnt} 项），已自动回传保存至本地目录！",
                            f"Batch generation complete ({b_cnt} items) and auto-saved locally!",
                        )
                    self.state_changed.emit()
                    self.check_colab_status(silent=True)
                elif st == "failed":
                    msg = localize_server_text(
                        strip_emoji(str(j.get("message") or tr("云端推理失败", "Cloud inference failed")))
                    )
                    if assistant_idx is not None and assistant_idx < len(self.chat_history):
                        if not self.chat_history[assistant_idx].content:
                            self.chat_history[assistant_idx].content = msg
                            self.chat_changed.emit()
                    self.is_generating = False
                    self.progress_value = 0.0
                    self.status_message = msg
                    self.state_changed.emit()
                else:
                    self.state_changed.emit()

            self._dispatch_ui(update_ui)
            if status in {"completed", "failed"}:
                return

    def cancel_current_generation(self) -> None:
        self._active_generation_token = str(uuid.uuid4())
        self.is_generating = False
        self.progress_value = 0.0
        self.status_message = tr(
            "已发送停止信号，正在中断云端 GPU 当前采样任务...",
            "Stop signal sent. Interrupting active cloud GPU sampling job...",
        )
        self.live_console_lines = [
            ln for ln in self.live_console_lines if "实时下载/采样进度流" not in ln
        ]
        self.state_changed.emit()
        self.console_changed.emit()

        base = self.normalized_base_url
        if base:
            threading.Thread(
                target=lambda: self._safe_post_empty(f"{base}/minimax_api/cancel", timeout=12),
                daemon=True,
            ).start()

    def _safe_post_empty(self, url: str, timeout: int = 12) -> None:
        try:
            self._admin_request("POST", url, json={}, timeout=timeout)
        except Exception:
            pass

