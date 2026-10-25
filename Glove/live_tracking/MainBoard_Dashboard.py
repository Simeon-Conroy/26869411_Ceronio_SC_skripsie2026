"""Dashboard for separate MainBoard calibration and calibrated live tracking."""

from __future__ import annotations

import ctypes
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
LIVE_FOLDER = Path(__file__).resolve().parent
if str(PROJECT_FOLDER) not in sys.path: sys.path.insert(0, str(PROJECT_FOLDER))
try:
    from PIL import Image, ImageGrab, ImageTk
except ModuleNotFoundError:
    Image = ImageGrab = ImageTk = None
from live_tracking.live_common.Live_IMU_Axis_Controls import AXIS_DEFAULTS, AXIS_LABELS
from live_tracking.live_common.Live_IMU_Mapping import CHANNEL_TO_SITE
from live_tracking.live_common.Live_IMU_Display_Controls import DEFAULT_DISPLAY_CONTROLS, read_display_controls, save_display_controls
from live_tracking.live_common.Live_IMU_Serial import LiveIMUSerial
from common.Const_INFO import IMU_ORDER

IMU_NAMES = list(IMU_ORDER)
DEFAULT_DISABLED_IMUS = set()
MADGWICK_FILTER = "Madgwick"
MADGWICK_GAIN = "0.1"
CALIBRATION_STEP_NAMES = {"stationary": "Long reference hold: bias and gravity", "neutral": "Short reference hold: joint zero", "straight_thumb": "Thumb local straight reference", "flexed_thumb": "Thumb full-flex endpoint", "straight_index": "Index local straight reference", "flexed_index": "Index full-flex endpoint", "straight_middle": "Middle local straight reference", "flexed_middle": "Middle full-flex endpoint", "straight_ring": "Ring local straight reference", "flexed_ring": "Ring full-flex endpoint", "straight_pinky": "Pinky local straight reference", "flexed_pinky": "Pinky full-flex endpoint", "fingers_together": "All fingers together", "all_spread": "All fingers spread", "return_reference": "Final reference and drift diagnostic"}
for _finger in ("thumb", "index", "middle", "ring", "pinky"):
    CALIBRATION_STEP_NAMES[f"abduction_left_{_finger}"] = f"{_finger.title()} leftmost abduction"
    CALIBRATION_STEP_NAMES[f"abduction_neutral_{_finger}"] = f"{_finger.title()} neutral abduction"
    CALIBRATION_STEP_NAMES[f"abduction_right_{_finger}"] = f"{_finger.title()} rightmost abduction"

DARK_THEME = {"background": "#121416", "panel": "#1b1f23", "surface": "#252b31", "border": "#46515b", "text": "#f3f0e8", "muted": "#a9b1b8", "primary": "#f2a900", "secondary": "#20b8d4", "viewer": "#101214", "health_card": "#101214", "field": "#20252a", "trough": "#30373d", "active": "#5b4307", "disabled": "#292e33", "disabled_text": "#6f787f", "good": "#2e9d62", "bad": "#d94b4b", "waiting": "#5f6b73", "red": "#d94b4b", "purple": "#8e5bd9", "blue": "#3182ce", "white": "#f1f1f1", "pink": "#e878a9", "yellow": "#d6a500", "grey": "#737b84", "good_text": "#f3f0e8", "bad_text": "#f3f0e8", "waiting_text": "#f3f0e8", "red_text": "#f3f0e8", "purple_text": "#f3f0e8", "blue_text": "#f3f0e8", "white_text": "#151515", "pink_text": "#151515", "yellow_text": "#151515", "grey_text": "#f3f0e8", "good_detail": "#8fa79a", "waiting_detail": "#a9b1b8", "bad_detail": "#f0a0a0"}


class MainBoardDashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.project_folder = PROJECT_FOLDER
        self.live_folder = LIVE_FOLDER
        self.dashboard_log_folder = self.project_folder / "dashboard_logs"
        self.dashboard_log_folder.mkdir(exist_ok=True)
        self.dashboard_log_file = self.dashboard_log_folder / f"Dashboard_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.dashboard_log_handle = self.dashboard_log_file.open("a", encoding="utf-8", errors="replace", buffering=1)
        self.calibration_file = self.live_folder / "live_imu_calibration.json"
        self.candidate_file = self.live_folder / "live_imu_calibration_candidate.json"
        self.stage1_file = self.live_folder / "live_imu_calibration_stage1.json"
        self.axis_control_file = self.live_folder / "joint_axis_controls.json"
        self.display_control_file = self.live_folder / "display_controls.json"
        self.calibrations_folder = self.project_folder / "calibrations"
        self.full_recordings_folder = self.project_folder / "recordings" / "dashboard_full"
        self.full_recordings_folder.mkdir(parents=True, exist_ok=True)
        self.full_recording_process: subprocess.Popen[str] | None = None
        self.full_recording_folder: Path | None = None
        self.process: subprocess.Popen[str] | None = None
        self.process_mode: str | None = None
        self.mediapipe_processes: dict[int, subprocess.Popen[str]] = {}
        self.embedded_mediapipe_hwnds: dict[int, int] = {}
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.health_queue: queue.Queue[dict] = queue.Queue()
        self.health_receiver: LiveIMUSerial | None = None
        self.embedded_viewer_hwnd: int | None = None
        self.startup_check_running = True
        self.closing = False
        self.calibration_completed_this_run = False
        self.stage1_complete = False
        self.calibration_step_ready = False
        self.calibration_step_review = False
        self.calibration_can_go_back = False
        self.stage1_final_review = False
        self.current_calibration_step = ""
        self.calibration_ready = False
        self.status = tk.StringVar(value="Calibration required")
        self.connection = tk.StringVar(value="Listener inactive")
        self.last_frame = tk.StringVar(value="No frames received")
        self.full_recording_status = tk.StringVar(value="Full recording stopped")
        self.calibration_text = tk.StringVar(value="Hold the glove in the reference pose, then press Calibrate.")
        self.calibration_stage_text = tk.StringVar(value="Stage: waiting")
        self.calibration_motion_text = tk.StringVar(value="Stability: not measuring")
        self.calibration_samples_text = tk.StringVar(value="Valid samples: 0")
        self.calibration_quality_text = tk.StringVar(value="Quality check: pending")
        self.calibration_progress = tk.DoubleVar(value=0.0)
        self.calibration_overall_progress = tk.DoubleVar(value=0.0)
        self.calibration_step_counter = tk.StringVar(value="No calibration running")
        self.filter_name = tk.StringVar(value=MADGWICK_FILTER)
        self.filter_gain = tk.StringVar(value=MADGWICK_GAIN)
        self.gain_slider_value = tk.DoubleVar(value=math.log10(float(MADGWICK_GAIN)))
        display_controls = read_display_controls(self.display_control_file)
        # Raw tracking is an explicit one-session action; never restore it
        # from the settings file when the dashboard opens.
        self.raw_mode = tk.BooleanVar(value=False)
        self.abduction_enabled = tk.BooleanVar(value=not display_controls.get("disable_abduction", DEFAULT_DISPLAY_CONTROLS["disable_abduction"]))
        self.middle_abduction_enabled = tk.BooleanVar(value=not display_controls.get("disable_middle_abduction", DEFAULT_DISPLAY_CONTROLS["disable_middle_abduction"]))
        self.display_mode_text = tk.StringVar(value="")
        self.use_gyro_bias = tk.BooleanVar(value=True)
        self.calibration_seconds = tk.StringVar(value="10")
        self.pose_snapshot_file = tk.StringVar(value="No pose snapshot file selected")
        self.full_calibration_flow = False
        self.active_calibration_steps = []
        self.active_calibration_names = {}
        self.reference_photo = None
        try: disabled_imus = set(json.loads((self.live_folder / "disabled_imus.json").read_text(encoding="utf-8")))
        except (FileNotFoundError, OSError, ValueError): disabled_imus = DEFAULT_DISABLED_IMUS.copy()
        self.imu_enabled = {name: tk.BooleanVar(value=name not in disabled_imus) for name in IMU_NAMES}
        saved_axes = self._read_axis_controls()
        self.axis_enabled = {name: tk.BooleanVar(value=saved_axes[name]) for name in AXIS_DEFAULTS}
        self._build_ui()
        self.root.bind_all("<Control-s>", self.capture_dashboard_views)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._start_startup_health_check()
        self.root.after(100, self._poll)

    def _collapsible(self, parent, title: str, expanded: bool = True, fill: str = "x", pady=0):
        container = ttk.Frame(parent)
        container.pack(fill=fill, pady=pady)
        body = ttk.Frame(container, padding=10)
        def toggle() -> None:
            if body.winfo_manager(): body.pack_forget(); button.configure(text=f"▶ {title}")
            else: body.pack(fill=fill, expand=fill == "both"); button.configure(text=f"▼ {title}")
        button = ttk.Button(container, text=f"▼ {title}" if expanded else f"▶ {title}", command=toggle)
        button.pack(fill="x")
        if expanded: body.pack(fill=fill, expand=fill == "both")
        return body

    def _apply_theme(self) -> None:
        palette = DARK_THEME
        self.palette = palette
        style = self.style
        background, panel, surface, border = palette["background"], palette["panel"], palette["surface"], palette["border"]
        text, muted, primary, secondary = palette["text"], palette["muted"], palette["primary"], palette["secondary"]
        self.root.configure(bg=background)
        style.configure("TFrame", background=background)
        style.configure("Panel.TFrame", background=panel)
        style.configure("TLabel", background=background, foreground=text, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=background, foreground=muted)
        style.configure("Primary.TLabel", background=background, foreground=primary)
        style.configure("Secondary.TLabel", background=background, foreground=secondary)
        style.configure("TButton", background=surface, foreground=text, bordercolor=border, lightcolor=surface, darkcolor=surface, font=("Segoe UI", 10), padding=(10, 7))
        style.map("TButton", background=[("active", palette["active"]), ("pressed", primary), ("disabled", palette["disabled"])], foreground=[("disabled", palette["disabled_text"])])
        style.configure("TLabelframe", background=background, foreground=primary, bordercolor=border)
        style.configure("TLabelframe.Label", background=background, foreground=primary, font=("Segoe UI", 10, "bold"))
        style.configure("TCheckbutton", background=background, foreground=text)
        style.map("TCheckbutton", background=[("active", background)], foreground=[("disabled", palette["disabled_text"])])
        style.configure("TEntry", fieldbackground=palette["field"], foreground=text, insertcolor=text, bordercolor=border)
        style.configure("TCombobox", fieldbackground=palette["field"], background=surface, foreground=text, arrowcolor=primary, bordercolor=border)
        style.map("TCombobox", fieldbackground=[("readonly", palette["field"])], foreground=[("readonly", text)])
        style.configure("TScrollbar", background=surface, troughcolor=background, bordercolor=background, arrowcolor=muted)
        style.configure("Horizontal.TProgressbar", troughcolor=palette["trough"], background=primary, bordercolor=border)
        style.configure("Treeview", background=palette["field"], fieldbackground=palette["field"], foreground=text, bordercolor=border, rowheight=26)
        style.configure("Treeview.Heading", background=surface, foreground=text, bordercolor=border, font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", primary)], foreground=[("selected", palette["good_text"])])
        self.page_canvas.configure(bg=background)
        self.right_canvas.configure(bg=background)
        self.reference_card.configure(bg=panel, highlightbackground=secondary)
        self.reference_title.configure(bg=panel, fg=secondary)
        self.reference_image.configure(bg=palette["viewer"], fg=muted)
        self.viewer_host.configure(bg=palette["viewer"], highlightbackground=border)
        self.viewer_placeholder.configure(bg=palette["viewer"], fg=muted)
        if hasattr(self, "mediapipe_hosts"):
            for host in self.mediapipe_hosts.values(): host.configure(bg=palette["viewer"], highlightbackground=border)
            for label in self.mediapipe_placeholders.values(): label.configure(bg=palette["viewer"], fg=muted)
        self.log.configure(bg=palette["viewer"], fg=text, insertbackground=text)
        for name, card in self.imu_health_cards.items():
            card.configure(bg=palette["health_card"], highlightbackground=border)
            label, detail = self.imu_health_labels[name], self.imu_error_labels[name]
            status = label.cget("text").rsplit("\n", 1)[-1]
            health = "good" if status == "GOOD" else "waiting" if status == "WAITING" else next((c for c, marker in (("red", "WHO_AM_I"), ("purple", "REINITIALISATION"), ("blue", "SPI READ"), ("white", "STALE"), ("pink", "CONFIGURATION"), ("yellow", "NO DATA"), ("grey", "HELD")) if marker in status), "red")
            label.configure(bg=palette[health], fg=palette.get(f"{health}_text", palette["text"]))
            detail.configure(bg=palette["health_card"], fg=palette.get(f"{health}_detail", palette["bad_detail"]))

    def _build_ui(self) -> None:
        self.root.title("Glove Dashboard")
        self.style = ttk.Style(self.root)
        if "clam" in self.style.theme_names(): self.style.theme_use("clam")
        palette = DARK_THEME
        self.palette = palette
        background, panel, surface, border = palette["background"], palette["panel"], palette["surface"], palette["border"]
        text, muted, teal, purple = palette["text"], palette["muted"], palette["primary"], palette["secondary"]
        style = self.style
        self.root.configure(bg=background)
        style.configure("TFrame", background=background)
        style.configure("Panel.TFrame", background=panel)
        style.configure("TLabel", background=background, foreground=text, font=("Segoe UI", 10))
        style.configure("TButton", background=surface, foreground=text, bordercolor=border, lightcolor=surface, darkcolor=surface, font=("Segoe UI", 10), padding=(10, 7))
        style.map("TButton", background=[("active", palette["active"]), ("pressed", teal), ("disabled", palette["disabled"])], foreground=[("disabled", palette["disabled_text"])])
        style.configure("TLabelframe", background=background, foreground=teal, bordercolor=border)
        style.configure("TLabelframe.Label", background=background, foreground=teal, font=("Segoe UI", 10, "bold"))
        style.configure("TCheckbutton", background=background, foreground=text)
        style.map("TCheckbutton", background=[("active", background)], foreground=[("disabled", palette["disabled_text"])])
        style.configure("TEntry", fieldbackground=palette["field"], foreground=text, insertcolor=text, bordercolor=border)
        style.configure("TCombobox", fieldbackground=palette["field"], background=surface, foreground=text, arrowcolor=teal, bordercolor=border)
        style.map("TCombobox", fieldbackground=[("readonly", palette["field"])], foreground=[("readonly", text)])
        style.configure("TScrollbar", background=surface, troughcolor=background, bordercolor=background, arrowcolor=muted)
        style.configure("Horizontal.TProgressbar", troughcolor=palette["trough"], background=teal, bordercolor=border)
        style.configure("Treeview", background=palette["field"], fieldbackground=palette["field"], foreground=text, bordercolor=border, rowheight=26)
        style.configure("Treeview.Heading", background=surface, foreground=text, bordercolor=border, font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", teal)], foreground=[("selected", palette["good_text"])])
        style.configure("Muted.TLabel", background=background, foreground=muted)
        style.configure("Primary.TLabel", background=background, foreground=teal)
        style.configure("Secondary.TLabel", background=background, foreground=purple)
        if os.name == "nt":
            self.root.geometry("1920x1080+0+0")
            self.root.state("zoomed")
        else: self.root.geometry("1920x1080")
        self.root.minsize(960, 700)
        page = ttk.Frame(self.root, padding=8)
        page.pack(fill="both", expand=True)
        self.page_grid = page
        page.columnconfigure(0, weight=3, minsize=480)
        page.columnconfigure(1, weight=6, minsize=700)
        page.columnconfigure(2, weight=3, minsize=460)
        page.rowconfigure(0, weight=1)
        left_panel, centre_panel, right_panel = ttk.Frame(page), ttk.Frame(page), ttk.Frame(page)
        self.left_panel, self.centre_panel, self.right_panel = left_panel, centre_panel, right_panel
        self.left_panel_visible, self.right_panel_visible = True, True
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        centre_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        self.page_canvas = tk.Canvas(left_panel, bg=background, highlightthickness=0)
        self.page_scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=self.page_canvas.yview)
        self.page_canvas.configure(yscrollcommand=self.page_scrollbar.set)
        self.page_scrollbar.pack(side="right", fill="y")
        self.page_canvas.pack(side="left", fill="both", expand=True)
        main = ttk.Frame(self.page_canvas, padding=12)
        self.page_window = self.page_canvas.create_window((0, 0), window=main, anchor="nw")
        main.bind("<Configure>", lambda event: self.page_canvas.configure(scrollregion=self.page_canvas.bbox("all")))
        self.page_canvas.bind("<Configure>", lambda event: self.page_canvas.itemconfigure(self.page_window, width=event.width))
        self.right_canvas = tk.Canvas(right_panel, bg=background, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=right_scrollbar.set)
        right_scrollbar.pack(side="right", fill="y")
        self.right_canvas.pack(side="left", fill="both", expand=True)
        right_main = ttk.Frame(self.right_canvas, padding=12)
        right_window = self.right_canvas.create_window((0, 0), window=right_main, anchor="nw")
        right_main.bind("<Configure>", lambda event: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")))
        self.right_canvas.bind("<Configure>", lambda event: self.right_canvas.itemconfigure(right_window, width=event.width))
        self.root.bind_all("<MouseWheel>", self._scroll_dashboard)
        panel_controls = ttk.Frame(centre_panel)
        panel_controls.pack(fill="x", pady=(0, 4))
        self.left_panel_button = ttk.Button(panel_controls, text="◀", width=3, command=self.toggle_left_panel)
        self.left_panel_button.pack(side="left")
        self.right_panel_button = ttk.Button(panel_controls, text="▶", width=3, command=self.toggle_right_panel)
        self.right_panel_button.pack(side="right")
        self.viewer_row = ttk.Frame(centre_panel)
        self.viewer_row.pack(fill="both", expand=True)
        self.viewer_row.columnconfigure(0, weight=0)
        self.viewer_row.columnconfigure(1, weight=1)
        self.viewer_row.rowconfigure(0, weight=1)
        self.reference_card = tk.Frame(self.viewer_row, width=360, bg=panel, highlightbackground=purple, highlightthickness=2)
        self.reference_card.grid_propagate(False)
        self.reference_title = tk.Label(self.reference_card, text="REFERENCE POSE", bg=panel, fg=purple, font=("Segoe UI", 12, "bold"))
        self.reference_title.pack(pady=(7, 3))
        self.reference_image = tk.Label(self.reference_card, bg=palette["viewer"], fg=muted, text="Reference pose image", justify="center")
        self.reference_image.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.viewer_host = tk.Frame(self.viewer_row, bg=palette["viewer"], highlightbackground=border, highlightthickness=2)
        self.viewer_host.grid(row=0, column=1, sticky="nsew")
        self.viewer_placeholder = tk.Label(self.viewer_host, text="MuJoCo viewer will appear here\nwhen calibration or tracking starts", bg=palette["viewer"], fg=muted, font=("Segoe UI", 16, "bold"), justify="center")
        self.viewer_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self.viewer_host.bind("<Configure>", self._resize_embedded_viewer)
        self.mediapipe_dock_frame = ttk.Frame(centre_panel)
        self.mediapipe_visible = False
        self.mediapipe_hosts, self.mediapipe_placeholders = {}, {}
        for slot, camera in enumerate((0, 3)):
            host = tk.Frame(self.mediapipe_dock_frame, height=320, bg=palette["viewer"], highlightbackground=border, highlightthickness=2)
            host.grid(row=0, column=slot, sticky="nsew", padx=(0, 4) if slot == 0 else (4, 0))
            host.grid_propagate(False)
            placeholder = tk.Label(host, text=f"Webcam {camera} dock", bg=palette["viewer"], fg=muted, font=("Segoe UI", 11, "bold"), justify="center")
            placeholder.place(relx=0.5, rely=0.5, anchor="center")
            host.bind("<Configure>", self._resize_embedded_mediapipe_windows)
            self.mediapipe_hosts[slot] = host
            self.mediapipe_placeholders[slot] = placeholder
        self.mediapipe_dock_frame.columnconfigure(0, weight=1)
        self.mediapipe_dock_frame.columnconfigure(1, weight=1)
        self.activity_frame = ttk.LabelFrame(centre_panel, text="Activity log / terminal output", padding=6)
        self.activity_frame.pack(fill="x", pady=(8, 0))
        self.log = tk.Text(self.activity_frame, height=9, wrap="word", state="disabled", font=("Consolas", 9), bg=palette["viewer"], fg=text, insertbackground=text, relief="flat")
        self.log.pack(fill="x")
        self._append_log(f"DASHBOARD_LOG_FILE {self.dashboard_log_file}\n")

        status_frame = self._collapsible(main, "Status", expanded=True)
        ttk.Label(status_frame, textvariable=self.status, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.connection).pack(anchor="w", pady=(4, 0))
        ttk.Label(status_frame, textvariable=self.last_frame).pack(anchor="w")
        self.recheck_button = ttk.Button(status_frame, text="Reconnect ESP and rerun IMU check", command=self._start_startup_health_check)
        self.recheck_button.pack(fill="x", pady=(10, 0))

        health_frame = self._collapsible(right_main, "Live IMU health", expanded=True)
        self.imu_health_cards, self.imu_health_labels, self.imu_error_labels = {}, {}, {}
        for index, name in enumerate(IMU_NAMES):
            card = tk.Frame(health_frame, bg=palette["health_card"], highlightbackground=border, highlightthickness=1)
            card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=3, pady=3)
            label = tk.Label(card, text=f"{name.removeprefix('imu_').replace('_', ' ').title()}\nWAITING", bg=palette["waiting"], fg=palette["waiting_text"], font=("Segoe UI", 8, "bold"), height=2, relief="flat", padx=3, pady=2)
            label.pack(fill="x")
            error_label = tk.Label(card, text="Waiting for data", bg=palette["health_card"], fg=muted, font=("Segoe UI", 7), height=2, anchor="w", justify="left", padx=4, pady=1, wraplength=185)
            error_label.pack(fill="x")
            self.imu_health_cards[name] = card
            self.imu_health_labels[name] = label
            self.imu_error_labels[name] = error_label
        for column in range(2): health_frame.columnconfigure(column, weight=1)
        legend = tk.Frame(health_frame, bg=palette["health_card"])
        legend.grid(row=(len(IMU_NAMES) + 1) // 2, column=0, columnspan=2, sticky="ew", padx=3, pady=(8, 3))
        ttk.Label(legend, text="Health key (highest active priority)", style="Muted.TLabel").pack(anchor="w")
        key_items = [("red", "WHO_AM_I / no response"), ("purple", "reinitialisation failed"), ("blue", "SPI read failed"), ("white", "persistent stale scans"), ("pink", "configuration changed"), ("yellow", "no data-ready"), ("grey", "held/outlier samples"), ("good", "healthy")]
        key_row = tk.Frame(legend, bg=palette["health_card"])
        key_row.pack(fill="x")
        for colour, text in key_items:
            tk.Label(key_row, text=f"  {text}  ", bg=palette[colour], fg=palette.get(f"{colour}_text", palette["text"]), font=("Segoe UI", 7), padx=2, pady=2).pack(side="left", padx=(0, 3), pady=3)

        calibration_frame = self._collapsible(main, "Calibration", expanded=True, pady=12)
        method_frame = ttk.Frame(calibration_frame)
        method_frame.pack(fill="x", pady=(0, 8))
        self.pose_file_button = ttk.Button(method_frame, text="Using reference + fist calibration", state="disabled")
        self.pose_file_button.pack(fill="x")
        ttk.Label(calibration_frame, textvariable=self.calibration_stage_text, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(calibration_frame, textvariable=self.calibration_step_counter, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(6, 0))
        self.calibration_instruction_label = ttk.Label(calibration_frame, textvariable=self.calibration_text, font=("Segoe UI", 17, "bold"), wraplength=400, justify="left")
        self.calibration_instruction_label.pack(anchor="w", fill="x", pady=(8, 8))
        calibration_frame.bind("<Configure>", lambda event: self.calibration_instruction_label.configure(wraplength=max(260, event.width - 24)))
        ttk.Label(calibration_frame, textvariable=self.calibration_motion_text).pack(anchor="w")
        ttk.Label(calibration_frame, textvariable=self.calibration_samples_text).pack(anchor="w")
        ttk.Label(calibration_frame, textvariable=self.calibration_quality_text).pack(anchor="w")
        ttk.Label(calibration_frame, text="Overall calibration progress", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        ttk.Progressbar(calibration_frame, variable=self.calibration_overall_progress, maximum=100).pack(fill="x", pady=(2, 6))
        ttk.Label(calibration_frame, text="Current capture progress", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        ttk.Progressbar(calibration_frame, variable=self.calibration_progress, maximum=100).pack(fill="x", pady=(2, 8))
        calibration_controls = ttk.Frame(calibration_frame)
        calibration_controls.pack(fill="x")
        ttk.Label(calibration_controls, text="Stationary time:").grid(row=0, column=0, sticky="w")
        self.duration_entry = ttk.Entry(calibration_controls, textvariable=self.calibration_seconds, width=6)
        self.duration_entry.grid(row=0, column=1, sticky="w", padx=(6, 3))
        ttk.Label(calibration_controls, text="seconds").grid(row=0, column=2, sticky="w")
        self.calibrate_button = ttk.Button(calibration_controls, text="▶  Start pose calibration", command=self.calibrate)
        self.calibrate_button.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        calibration_controls.columnconfigure(2, weight=1)
        step_controls = ttk.LabelFrame(calibration_frame, text="Current calibration step", padding=8)
        step_controls.pack(fill="x", pady=(10, 0))
        self.calibration_step_button = ttk.Button(step_controls, text="▶  Start this step", command=self.start_calibration_step, state="disabled")
        self.calibration_step_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.back_step_button = ttk.Button(step_controls, text="←  Go back", command=self.go_back_calibration_step, state="disabled")
        self.back_step_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.skip_step_button = ttk.Button(step_controls, text="Skip this step  →", command=self.skip_calibration_step, state="disabled")
        self.skip_step_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))
        for column in range(3): step_controls.columnconfigure(column, weight=1)
        self.bias_check = ttk.Checkbutton(calibration_frame, text="Apply measured gyro-bias correction during tracking", variable=self.use_gyro_bias)
        self.bias_check.pack(anchor="w", pady=(8, 0))
        step_review_controls = ttk.Frame(calibration_frame)
        step_review_controls.pack(fill="x", pady=(8, 0))
        ttk.Label(step_review_controls, text="Calibration step review", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.accept_step_button = ttk.Button(step_review_controls, text="Accept this step", command=lambda: self.review_calibration_step("ACCEPT"), state="disabled")
        self.accept_step_button.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        self.redo_step_button = ttk.Button(step_review_controls, text="Redo only this step", command=lambda: self.review_calibration_step("REDO"), state="disabled")
        self.redo_step_button.grid(row=1, column=1, sticky="ew", padx=(4, 0))
        step_review_controls.columnconfigure(0, weight=1)
        step_review_controls.columnconfigure(1, weight=1)
        self.final_review_controls = ttk.LabelFrame(calibration_frame, text="Final calibration review", padding=8)
        self.final_review_controls.pack(fill="x", pady=(8, 0))
        self.finish_stage1_button = ttk.Button(self.final_review_controls, text="✓  Accept and save", command=self.finish_stage1, state="disabled")
        self.finish_stage1_button.grid(row=0, column=0, sticky="ew")
        self.final_review_controls.columnconfigure(0, weight=1)
        tracking = self._collapsible(right_main, "Live tracking", expanded=True, pady=(12, 0))
        selector = ttk.Frame(tracking)
        selector.pack(fill="x")
        ttk.Label(selector, text="Filter:").grid(row=0, column=0, sticky="w")
        ttk.Label(selector, text=MADGWICK_FILTER, style="Primary.TLabel").grid(row=0, column=1, columnspan=4, sticky="w", padx=(6, 0))
        ttk.Label(selector, text="Gain:").grid(row=1, column=0, sticky="w", pady=(7, 0))
        self.gain_slider = ttk.Scale(selector, from_=-5, to=3, variable=self.gain_slider_value, command=self._gain_slider_changed)
        self.gain_slider.grid(row=1, column=1, sticky="ew", padx=(6, 5), pady=(7, 0))
        ttk.Button(selector, text="÷10", width=4, command=lambda: self._multiply_gain(0.1)).grid(row=1, column=2, padx=2, pady=(7, 0))
        ttk.Button(selector, text="×10", width=4, command=lambda: self._multiply_gain(10.0)).grid(row=1, column=3, padx=2, pady=(7, 0))
        self.gain_entry = ttk.Entry(selector, textvariable=self.filter_gain, width=9)
        self.gain_entry.grid(row=1, column=4, sticky="e", padx=(5, 0), pady=(7, 0))
        selector.columnconfigure(1, weight=1)
        self.gain_entry.bind("<Return>", lambda event: self._custom_gain_changed())
        self.gain_entry.bind("<FocusOut>", lambda event: self._custom_gain_changed())
        tracking_actions = ttk.Frame(tracking)
        tracking_actions.pack(fill="x", pady=(10, 0))
        self.start_button = ttk.Button(tracking_actions, text="▶  Start tracking", command=self.start_tracking, state="disabled")
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(tracking_actions, text="■  Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        self.skip_calibration_button = ttk.Button(tracking, text="Skip calibration and start raw tracking", command=self.skip_calibration)
        self.skip_calibration_button.pack(fill="x", pady=(8, 0))
        self.mediapipe_button = ttk.Button(tracking, text="Show Cameras", command=self.toggle_camera_display)
        self.mediapipe_button.pack(fill="x", pady=(8, 0))

        recording_frame = self._collapsible(right_main, "Full session recording", expanded=True, pady=(12, 0))
        ttk.Label(recording_frame, textvariable=self.full_recording_status, wraplength=430, justify="left").pack(anchor="w")
        recording_buttons = ttk.Frame(recording_frame)
        recording_buttons.pack(fill="x", pady=(8, 0))
        self.start_full_recording_button = ttk.Button(recording_buttons, text="●  Start full recording", command=self.start_full_recording)
        self.start_full_recording_button.pack(side="left", fill="x", expand=True)
        self.stop_full_recording_button = ttk.Button(recording_buttons, text="■  Stop recording", command=self.stop_full_recording, state="disabled")
        self.stop_full_recording_button.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self._refresh_stage1_resume()

        axis_frame = self._collapsible(right_main, "Display locks — axes and abduction", expanded=True, pady=(12, 0))
        ttk.Label(axis_frame, text="Disable a control to hold that movement at zero in preview/tracking. Saved calibration data is preserved.", wraplength=430, justify="left").grid(row=0, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 5))
        for index, (name, label) in enumerate(AXIS_LABELS.items()):
            ttk.Checkbutton(axis_frame, text=label, variable=self.axis_enabled[name], command=self.axis_controls_changed).grid(row=1 + index // 3, column=index % 3, sticky="w", padx=4, pady=3)
        self.disable_abduction_check = ttk.Checkbutton(axis_frame, text="Finger abduction", variable=self.abduction_enabled, command=self.display_controls_changed)
        self.disable_abduction_check.grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 2))
        self.middle_abduction_check = ttk.Checkbutton(axis_frame, text="Middle finger abduction", variable=self.middle_abduction_enabled, command=self.display_controls_changed)
        self.middle_abduction_check.grid(row=4, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 2))
        self.display_mode_label = ttk.Label(axis_frame, textvariable=self.display_mode_text, style="Warning.TLabel", wraplength=430)
        self.display_mode_label.grid(row=10, column=0, columnspan=3, sticky="ew", padx=4, pady=(4, 3))
        axis_buttons = ttk.Frame(axis_frame)
        axis_buttons.grid(row=11, column=0, columnspan=3, sticky="w", pady=(7, 0))
        ttk.Button(axis_buttons, text="Enable all", command=lambda: self._set_all_axes(True)).pack(side="left")
        ttk.Button(axis_buttons, text="Disable all", command=lambda: self._set_all_axes(False)).pack(side="left", padx=8)
        for column in range(3): axis_frame.columnconfigure(column, weight=1)

        imu_frame = self._collapsible(right_main, "IMUs used by calibration and tracking", expanded=False, pady=(12, 0))
        for index, name in enumerate(IMU_NAMES): ttk.Checkbutton(imu_frame, text=name.removeprefix("imu_"), variable=self.imu_enabled[name], command=self._imu_selection_changed).grid(row=index // 4, column=index % 4, sticky="w", padx=8, pady=3)
        imu_buttons = ttk.Frame(imu_frame)
        imu_buttons.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(imu_buttons, text="Enable all", command=lambda: self._set_all_imus(True)).pack(side="left")
        ttk.Button(imu_buttons, text="Set all to zero", command=lambda: self._set_all_imus(False)).pack(side="left", padx=8)

    def _set_imu_health(self, name: str, health: str, status: str, detail: str = "") -> None:
        if name not in self.imu_health_labels: return
        title = name.removeprefix("imu_").replace("_", " ").title()
        self.imu_health_labels[name].configure(bg=self.palette[health], fg=self.palette[f"{health}_text"], text=f"{title}\n{status.upper()}")
        if health == "good": self.imu_error_labels[name].configure(text="No errors", fg=self.palette["good_detail"])
        elif health == "waiting": self.imu_error_labels[name].configure(text=detail or "Waiting for data", fg=self.palette["waiting_detail"])
        else: self.imu_error_labels[name].configure(text=detail or status, fg=self.palette["bad_detail"])

    def toggle_left_panel(self) -> None:
        self.left_panel_visible = not self.left_panel_visible
        if self.left_panel_visible:
            self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            self.page_grid.columnconfigure(0, weight=3, minsize=480)
            self.left_panel_button.configure(text="◀")
        else:
            self.left_panel.grid_remove()
            self.page_grid.columnconfigure(0, weight=0, minsize=0)
            self.left_panel_button.configure(text="▶")
        self._resize_embedded_viewer()

    def toggle_right_panel(self) -> None:
        self.right_panel_visible = not self.right_panel_visible
        if self.right_panel_visible:
            self.right_panel.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
            self.page_grid.columnconfigure(2, weight=3, minsize=460)
            self.right_panel_button.configure(text="▶")
        else:
            self.right_panel.grid_remove()
            self.page_grid.columnconfigure(2, weight=0, minsize=0)
            self.right_panel_button.configure(text="◀")
        self._resize_embedded_viewer()

    def _scroll_dashboard(self, event) -> None:
        try: widget = self.root.winfo_containing(*self.root.winfo_pointerxy())
        except (KeyError, tk.TclError): return
        while widget is not None and widget not in {self.page_canvas, self.right_canvas}: widget = widget.master
        canvas = self.right_canvas if widget == self.right_canvas else self.page_canvas
        if canvas.winfo_exists(): canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _read_axis_controls(self) -> dict[str, bool]:
        values = AXIS_DEFAULTS.copy()
        try:
            saved = json.loads(self.axis_control_file.read_text(encoding="utf-8"))
            for name in values:
                if name in saved: values[name] = bool(saved[name])
        except (FileNotFoundError, OSError, ValueError, TypeError): pass
        return values

    def _save_axis_controls(self) -> None:
        values = {name: enabled.get() for name, enabled in self.axis_enabled.items()}
        self.axis_control_file.write_text(json.dumps(values, indent=2), encoding="utf-8")
        enabled = sum(values.values())
        self.calibration_quality_text.set(f"Wrist/Palm model axes: {enabled} of 6 enabled; changes are live and calibration remains saved")
        self.refresh_display_mode_text()

    def axis_controls_changed(self) -> None:
        self._save_axis_controls()
        disabled_count = sum(1 for value in self.axis_enabled.values() if not value.get())
        if disabled_count and not self.calibration_ready and not self.raw_mode.get():
            messagebox.showwarning("NO wrist/palm axis display", f"{disabled_count} Wrist/Palm axis control(s) are disabled. Calibration preview and live tracking will hold those axes at zero until you enable them again.")
        self._set_running_controls(False)

    def _set_all_axes(self, enabled: bool) -> None:
        for value in self.axis_enabled.values(): value.set(enabled)
        self.abduction_enabled.set(enabled)
        self.middle_abduction_enabled.set(enabled)
        self._save_axis_controls()
        self.display_controls_changed(show_warning=True)

    def _python_command(self, script: str, *arguments: str) -> list[str]:
        project_python = self.project_folder / "skripsie_venv311" / "Scripts" / "python.exe"
        return [str(project_python if project_python.exists() else Path(sys.executable)), "-u", str(self.project_folder / script), *arguments]

    def _find_ffmpeg(self) -> str | None:
        explicit = os.environ.get("MAINBOARD_FFMPEG")
        if explicit and Path(explicit).exists(): return explicit
        path_hit = shutil.which("ffmpeg")
        if path_hit: return path_hit
        candidates = [Path("C:/ffmpeg/bin/ffmpeg.exe"), Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"), Path("C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe")]
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            winget_root = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
            try: candidates.extend(sorted(winget_root.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"), reverse=True))
            except OSError: pass
        direct = Path("C:/Users/ceron/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin/ffmpeg.exe")
        candidates.append(direct)
        for candidate in candidates:
            if candidate.exists(): return str(candidate)
        return None

    def _write_full_recording_metadata(self, folder: Path, status: str) -> None:
        metadata = {
            "status": status,
            "created_at": datetime.now().astimezone().isoformat(),
            "dashboard_log_file": str(self.dashboard_log_file),
            "filter": self.filter_name.get(),
            "gain": self.filter_gain.get(),
            "raw_mode": bool(self.raw_mode.get()),
            "abduction_enabled": bool(self.abduction_enabled.get()),
            "middle_abduction_enabled": bool(self.middle_abduction_enabled.get()),
            "use_gyro_bias": bool(self.use_gyro_bias.get()),
            "calibration_file": str(self.calibration_file if self.calibration_file.exists() else ""),
            "mediapipe_cameras": [0, 3]
        }
        try:
            (folder / "session_settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except OSError as error:
            self._append_log(f"FULL_RECORDING_METADATA_ERROR {error}\n")

    def _copy_full_recording_sidecars(self, folder: Path) -> None:
        try: self.dashboard_log_handle.flush()
        except OSError: pass
        for source, target_name in ((self.dashboard_log_file, "dashboard.log"), (self.calibration_file, "calibration_used.json"), (self.display_control_file, "display_controls.json"), (self.axis_control_file, "joint_axis_controls.json")):
            try:
                if source.exists(): shutil.copy2(source, folder / target_name)
            except OSError as error:
                self._append_log(f"FULL_RECORDING_COPY_ERROR {source.name}: {error}\n")

    def start_full_recording(self) -> bool:
        if self.full_recording_process is not None and self.full_recording_process.poll() is None:
            self.status.set("Full recording is already running")
            return True
        ffmpeg_path = self._find_ffmpeg()
        if not ffmpeg_path:
            messagebox.showerror("FFmpeg not found", "Install FFmpeg or set MAINBOARD_FFMPEG to ffmpeg.exe. Tracking was not started because this workflow records every live run.")
            return False
        folder = self.full_recordings_folder / f"Session_{datetime.now():%Y%m%d_%H%M%S}"
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "dashboard_video.mp4"
        self.root.update_idletasks()
        capture_x, capture_y = max(0, self.root.winfo_rootx()), max(0, self.root.winfo_rooty())
        capture_w, capture_h = max(2, self.root.winfo_width()), max(2, self.root.winfo_height())
        capture_w -= capture_w % 2
        capture_h -= capture_h % 2
        command = [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-thread_queue_size", "512", "-f", "gdigrab", "-framerate", "15", "-offset_x", str(capture_x), "-offset_y", str(capture_y), "-video_size", f"{capture_w}x{capture_h}", "-i", "desktop"]
        command += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p", "-fps_mode", "cfr"]
        command += ["-movflags", "+faststart", str(output)]
        self._write_full_recording_metadata(folder, "recording")
        self._copy_full_recording_sidecars(folder)
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.full_recording_process = subprocess.Popen(command, cwd=self.project_folder, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=creation_flags)
        except OSError as error:
            messagebox.showerror("Recording could not start", str(error))
            self.full_recording_process = None
            return False
        self.full_recording_folder = folder
        threading.Thread(target=self._read_output, args=(self.full_recording_process,), daemon=True).start()
        self.full_recording_status.set(f"Recording to {folder.name}")
        self._append_log(f"FULL_RECORDING_STARTED {folder}; area={capture_w}x{capture_h}+{capture_x}+{capture_y}\n")
        self._set_running_controls(False)
        return True

    def stop_full_recording(self) -> None:
        process = self.full_recording_process
        folder = self.full_recording_folder
        if process is None or process.poll() is not None:
            self.full_recording_process = None
            self.full_recording_status.set("Full recording stopped")
            self._set_running_controls(False)
            return
        try:
            if process.stdin:
                process.stdin.write("q\n")
                process.stdin.flush()
            process.wait(timeout=8)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill()
        self.full_recording_process = None
        if folder:
            self._write_full_recording_metadata(folder, "stopped")
            self._copy_full_recording_sidecars(folder)
            self.full_recording_status.set(f"Saved full recording: {folder}")
            self._append_log(f"FULL_RECORDING_STOPPED {folder}\n")
        else:
            self.full_recording_status.set("Full recording stopped")
        self._set_running_controls(False)

    def _launch(self, mode: str, command: list[str], embed_viewer: bool = True) -> None:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(command, cwd=self.project_folder, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=creation_flags)
        self.process_mode = mode
        self.embedded_viewer_hwnd = None
        if embed_viewer:
            self.viewer_placeholder.configure(text="Starting MuJoCo viewer…")
            self.viewer_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()
        if embed_viewer and os.name == "nt": self.root.after(250, lambda: self._embed_mujoco_window(0))

    def _embed_mujoco_window(self, attempt: int) -> None:
        if self.process is None or self.process.poll() is not None or self.embedded_viewer_hwnd is not None: return
        user32, windows = ctypes.windll.user32, []
        user32.SetParent.argtypes, user32.SetParent.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p
        user32.GetWindowLongW.argtypes, user32.GetWindowLongW.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_long
        user32.SetWindowLongW.argtypes, user32.SetWindowLongW.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long], ctypes.c_long
        user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def collect(window, parameter):
            process_id, title = ctypes.c_ulong(), ctypes.create_unicode_buffer(256)
            user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
            user32.GetWindowTextW(window, title, len(title))
            if process_id.value == self.process.pid and user32.IsWindowVisible(window) and title.value: windows.append(int(window))
            return True

        user32.EnumWindows(collect, 0)
        if not windows:
            if attempt < 80: self.root.after(250, lambda: self._embed_mujoco_window(attempt + 1))
            else: self.viewer_placeholder.configure(text="MuJoCo opened separately\n(the native window could not be embedded)")
            return
        self._dock_mujoco_window(windows[0])

    def _dock_mujoco_window(self, window: int) -> None:
        if os.name != "nt": return
        if self.embedded_viewer_hwnd is not None and self.embedded_viewer_hwnd != window:
            self._resize_embedded_viewer()
            return
        user32 = ctypes.windll.user32
        user32.SetParent.argtypes, user32.SetParent.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p
        user32.GetWindowLongW.argtypes, user32.GetWindowLongW.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_long
        user32.SetWindowLongW.argtypes, user32.SetWindowLongW.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long], ctypes.c_long
        user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        host = self.viewer_host.winfo_id()
        user32.SetParent(ctypes.c_void_p(window), ctypes.c_void_p(host))
        style = user32.GetWindowLongW(ctypes.c_void_p(window), -16)
        child_style = (style | 0x40000000) & ~(0x80000000 | 0x00C00000 | 0x00040000 | 0x00080000 | 0x00020000 | 0x00010000)
        user32.SetWindowLongW(ctypes.c_void_p(window), -16, child_style)
        self.embedded_viewer_hwnd = window
        self.viewer_placeholder.place_forget()
        user32.ShowWindow(ctypes.c_void_p(window), 5)
        self._resize_embedded_viewer()
        for delay in (80, 250, 600, 1200):
            self.root.after(delay, self._resize_embedded_viewer)

    def _resize_embedded_viewer(self, event=None) -> None:
        if os.name != "nt" or self.embedded_viewer_hwnd is None: return
        width, height = max(1, self.viewer_host.winfo_width()), max(1, self.viewer_host.winfo_height())
        ctypes.windll.user32.SetWindowPos(ctypes.c_void_p(self.embedded_viewer_hwnd), 0, 0, 0, width, height, 0x0060)

    def _clear_embedded_viewer(self) -> None:
        self.embedded_viewer_hwnd = None
        if hasattr(self, "viewer_placeholder"):
            self.viewer_placeholder.configure(text="MuJoCo viewer will appear here\nwhen calibration or tracking starts")
            self.viewer_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def _embed_mediapipe_window(self, slot: int, process: subprocess.Popen[str], attempt: int) -> None:
        if os.name != "nt" or process.poll() is not None or slot in self.embedded_mediapipe_hwnds: return
        user32, windows = ctypes.windll.user32, []
        user32.SetParent.argtypes, user32.SetParent.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p
        user32.GetWindowLongW.argtypes, user32.GetWindowLongW.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_long
        user32.SetWindowLongW.argtypes, user32.SetWindowLongW.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long], ctypes.c_long
        user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def collect(window, parameter):
            process_id, title = ctypes.c_ulong(), ctypes.create_unicode_buffer(256)
            user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
            user32.GetWindowTextW(window, title, len(title))
            if process_id.value == process.pid and user32.IsWindowVisible(window) and title.value: windows.append(int(window))
            return True

        user32.EnumWindows(collect, 0)
        if not windows:
            if attempt < 80: self.root.after(250, lambda: self._embed_mediapipe_window(slot, process, attempt + 1))
            else: self.mediapipe_placeholders[slot].configure(text="MediaPipe opened separately\n(the native window could not be embedded)")
            return
        self._dock_mediapipe_window(slot, windows[0])

    def _dock_mediapipe_window(self, slot: int, window: int) -> None:
        if os.name != "nt" or slot in self.embedded_mediapipe_hwnds or slot not in self.mediapipe_hosts: return
        user32 = ctypes.windll.user32
        user32.SetParent.argtypes, user32.SetParent.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p
        user32.GetWindowLongW.argtypes, user32.GetWindowLongW.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_long
        user32.SetWindowLongW.argtypes, user32.SetWindowLongW.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long], ctypes.c_long
        user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        host = self.mediapipe_hosts[slot].winfo_id()
        user32.SetParent(ctypes.c_void_p(window), ctypes.c_void_p(host))
        style = user32.GetWindowLongW(window, -16)
        child_style = (style | 0x40000000) & ~(0x80000000 | 0x00C00000 | 0x00040000 | 0x00080000 | 0x00020000 | 0x00010000)
        user32.SetWindowLongW(window, -16, child_style)
        self.embedded_mediapipe_hwnds[slot] = window
        self.mediapipe_placeholders[slot].place_forget()
        self._resize_embedded_mediapipe_windows()
        user32.ShowWindow(ctypes.c_void_p(window), 5)

    def _resize_embedded_mediapipe_windows(self, event=None) -> None:
        if os.name != "nt": return
        for slot, window in list(self.embedded_mediapipe_hwnds.items()):
            host = self.mediapipe_hosts.get(slot)
            if host is None: continue
            width, height = max(1, host.winfo_width()), max(1, host.winfo_height())
            ctypes.windll.user32.SetWindowPos(window, 0, 0, 0, width, height, 0x0060)

    def _clear_embedded_mediapipe_windows(self) -> None:
        self.embedded_mediapipe_hwnds.clear()
        if hasattr(self, "mediapipe_placeholders"):
            for slot, label in self.mediapipe_placeholders.items():
                label.configure(text=f"Camera {slot} dock")
                label.place(relx=0.5, rely=0.5, anchor="center")

    def _choose_pose_file(self) -> str | None:
        selected = filedialog.askopenfilename(title="Choose named pose snapshots", initialdir=str(self.project_folder), filetypes=[("HDF5 pose snapshots", "*.h5 *.hdf5"), ("All files", "*.*")])
        if selected:
            self.pose_snapshot_file.set(selected)
            self.pose_file_button.configure(text=f"H5: {Path(selected).name}")
            return selected
        return None

    def _show_reference_image(self, payload: dict) -> None:
        try:
            if Image is not None and ImageTk is not None:
                image = Image.open(payload["path"])
                image.thumbnail((340, 620), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
            else:
                photo = tk.PhotoImage(file=payload["path"])
            self.reference_photo = photo
            self.reference_title.configure(text=f"REFERENCE POSE — {payload.get('label', '')}")
            self.reference_image.configure(image=photo, text="")
            if not self.reference_card.winfo_manager():
                self.reference_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
                self.viewer_host.grid_configure(row=0, column=1, columnspan=1, sticky="nsew")
                self.viewer_row.columnconfigure(0, minsize=360, weight=0)
                self.viewer_row.columnconfigure(1, weight=1)
                self.root.after(50, self._resize_embedded_viewer)
        except (OSError, tk.TclError, KeyError) as error:
            self.reference_image.configure(image="", text=f"Could not display reference pose: {error}")

    def _hide_reference_image(self) -> None:
        if hasattr(self, "reference_card"): self.reference_card.grid_forget()
        if hasattr(self, "viewer_row"):
            self.viewer_row.columnconfigure(0, minsize=0, weight=0)
            self.viewer_row.columnconfigure(1, weight=1)
        if hasattr(self, "viewer_host"):
            self.viewer_host.grid_configure(row=0, column=0, columnspan=2, sticky="nsew")
        self.reference_photo = None
        if hasattr(self, "root"): self.root.after(50, self._resize_embedded_viewer)

    def calibrate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.stop()
            self.root.after(300, self.calibrate)
            return
        # Release the dashboard's idle health-monitor serial reader before
        # launching calibration.  Both readers cannot consume the same COM
        # stream; leaving this alive makes calibration receive no frames.
        if self.health_receiver is not None:
            self.health_receiver.stop()
            self.health_receiver = None
        try: duration = float(self.calibration_seconds.get())
        except ValueError: messagebox.showerror("Invalid duration", "Enter a numeric calibration duration."); return
        if duration <= 0: messagebox.showerror("Invalid duration", "Calibration duration must be positive."); return
        if not any(value.get() for value in self.imu_enabled.values()): messagebox.showerror("No IMUs enabled", "Enable at least one IMU before calibration."); return
        self._save_disabled_imus()
        snapshot_path = str(self.project_folder / "datasets" / "snapshots" / "USE_THIS_ONE.h5")
        if not Path(snapshot_path).is_file(): messagebox.showerror("Calibration file missing", f"Missing fixed calibration file:\\n{snapshot_path}"); return
        self.active_calibration_steps = []
        self.active_calibration_names = {}
        self.calibration_step_counter.set("Loading snapshot poses")
        self.calibration_stage_text.set("Pose calibration")
        self.calibration_motion_text.set("Stability: waiting for measurements")
        self.calibration_samples_text.set("Valid samples: 0")
        self.calibration_quality_text.set("Quality check: pending")
        self.status.set("Calibrating")
        self.connection.set("Opening MainBoard serial port")
        self.calibration_text.set("Loading the named reference poses.")
        try: gain = float(self.filter_gain.get())
        except ValueError: messagebox.showerror("Invalid gain", "Enter a numeric Madgwick gain."); return
        self.calibration_ready = False
        self._force_calibrated_mode()
        self.calibration_completed_this_run = False
        self.full_calibration_flow = True
        self.stage1_complete = False
        self.calibration_step_ready = False
        self.calibration_step_review = False
        self.stage1_final_review = False
        self.calibration_progress.set(0)
        self.calibration_overall_progress.set(0)
        bias_option = "--use-gyro-bias" if self.use_gyro_bias.get() else "--no-use-gyro-bias"
        self._launch("pose_calibration", self._python_command("live_tracking/calibration/Pose_Snapshot_Calibration.py", "--snapshots", snapshot_path, "--duration", str(duration), "--filter", MADGWICK_FILTER, "--gain", str(gain), bias_option))
        self._set_running_controls(True)

    def start_calibration_step(self) -> None:
        if not self.calibration_step_ready or self.process is None or self.process.stdin is None: return
        self.process.stdin.write("START\n")
        self.process.stdin.flush()
        self.calibration_step_ready = False
        self.calibration_step_button.configure(state="disabled")

    def skip_calibration_step(self) -> None:
        skippable = self.process_mode == "pose_calibration" and self.current_calibration_step != "stationary"
        if not self.calibration_step_ready or not skippable or self.process is None or self.process.stdin is None: return
        self.process.stdin.write("SKIP\n")
        self.process.stdin.flush()
        self.calibration_step_ready = False
        self.skip_step_button.configure(state="disabled")
        self.calibration_text.set(f"Skipped {CALIBRATION_STEP_NAMES.get(self.current_calibration_step, self.current_calibration_step)}.")

    def go_back_calibration_step(self) -> None:
        if not self.calibration_step_ready or not self.calibration_can_go_back or self.process is None or self.process.stdin is None: return
        self.process.stdin.write("BACK\n")
        self.process.stdin.flush()
        self.calibration_step_ready = False
        self.calibration_can_go_back = False
        self.back_step_button.configure(state="disabled")
        self.calibration_text.set("Returning to the previously accepted calibration step.")

    def review_calibration_step(self, decision: str) -> None:
        if not self.calibration_step_review or self.process is None or self.process.stdin is None: return
        self.process.stdin.write(f"{decision}\n")
        self.process.stdin.flush()
        self.calibration_step_review = False
        self.accept_step_button.configure(state="disabled")
        self.redo_step_button.configure(state="disabled")
        self.calibration_motion_text.set("Step accepted; preparing the next step" if decision == "ACCEPT" else "Redo selected; preparing this step again")

    def finish_stage1(self) -> None:
        if not self.stage1_final_review or self.process is None or self.process.stdin is None: return
        self.process.stdin.write("ACCEPT_STAGE1\n")
        self.process.stdin.flush()
        self.stage1_final_review = False
        self._set_running_controls(True)

    def save_display_controls(self) -> None:
        save_display_controls(self.display_control_file, self.raw_mode.get(), not self.abduction_enabled.get(), not self.middle_abduction_enabled.get())
        self.refresh_display_mode_text()

    def _force_calibrated_mode(self) -> None:
        if self.raw_mode.get():
            self.raw_mode.set(False)
        self.save_display_controls()

    def display_controls_changed(self, show_warning: bool = True) -> None:
        self.save_display_controls()
        if show_warning and not self.abduction_enabled.get() and not self.calibration_ready and not self.raw_mode.get():
            messagebox.showwarning("NO abduction calibration", "Finger abduction is disabled. Calibration preview and live tracking will hold all finger abduction joints at zero until you enable it again.")
        self._set_running_controls(False)

    def refresh_display_mode_text(self) -> None:
        messages = []
        disabled_axes = sum(1 for value in self.axis_enabled.values() if not value.get())
        if self.raw_mode.get(): messages.append("Raw tracking mode: saved calibration is bypassed.")
        if disabled_axes and not self.calibration_ready and not self.raw_mode.get(): messages.append(f"NO wrist/palm axis display: {disabled_axes} axis control(s) held at zero.")
        elif disabled_axes: messages.append(f"{disabled_axes} wrist/palm axis control(s) disabled for preview/tracking.")
        if not self.abduction_enabled.get() and not self.calibration_ready and not self.raw_mode.get(): messages.append("NO abduction calibration: finger abduction is held at zero.")
        elif not self.abduction_enabled.get(): messages.append("Finger abduction disabled for preview/tracking; saved calibration is unchanged.")
        elif not self.middle_abduction_enabled.get(): messages.append("Middle finger abduction is post-zeroed in final qpos.")
        self.display_mode_text.set(" ".join(messages))

    def skip_calibration(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.status.set("Stop the current process before starting raw tracking")
            return
        self.raw_mode.set(True)
        self.status.set("Raw tracking mode")
        self.connection.set("Calibration skipped; starting raw live tracking")
        self.last_frame.set("Waiting for raw IMU packets")
        self.calibration_stage_text.set("Stage: raw tracking")
        self.calibration_text.set("Calibration skipped. Raw live tracking is starting.")
        self.calibration_samples_text.set("Raw mode uses live IMU packets directly")
        self.calibration_quality_text.set("Raw/no calibration")
        self.calibration_overall_progress.set(100)
        self.calibration_progress.set(100)
        self._set_running_controls(False)
        self.start_tracking(raw_requested=True)

    def toggle_camera_display(self) -> None:
        if any(process.poll() is None for process in self.mediapipe_processes.values()):
            self.stop_mediapipe()
            self._hide_camera_docks()
        else:
            self.start_mediapipe_recording()

    def _show_camera_docks(self) -> None:
        if self.mediapipe_visible: return
        self.mediapipe_dock_frame.pack(fill="x", pady=(0, 4), before=self.activity_frame)
        self.mediapipe_visible = True
        self.mediapipe_button.configure(text="Hide Cameras")
        self._resize_embedded_viewer()
        self._resize_embedded_mediapipe_windows()

    def _hide_camera_docks(self) -> None:
        if not self.mediapipe_visible: return
        self.mediapipe_dock_frame.pack_forget()
        self.mediapipe_visible = False
        self.mediapipe_button.configure(text="Show Cameras")
        self._resize_embedded_viewer()

    def start_mediapipe_recording(self) -> None:
        self._show_camera_docks()
        if any(process.poll() is None for process in self.mediapipe_processes.values()):
            self._resize_embedded_mediapipe_windows()
            return
        cameras = [0, 3]
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.embedded_mediapipe_hwnds.clear()
        for slot, camera_index in enumerate(cameras):
            title = f"MainBoard Camera {slot} Camera {camera_index}"
            command = self._python_command("live_tracking/MediaPipe_Marker_Display.py", "--camera", str(camera_index), "--dock-slot", str(slot), "--window-title", title, "--width", "960", "--height", "540", "--fps", "12")
            process = subprocess.Popen(command, cwd=self.project_folder, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=creation_flags)
            self.mediapipe_processes[slot] = process
            self.mediapipe_placeholders[slot].configure(text=f"Starting camera {camera_index}…")
            self.mediapipe_placeholders[slot].place(relx=0.5, rely=0.5, anchor="center")
            threading.Thread(target=self._read_output, args=(process,), daemon=True).start()
            if os.name == "nt": self.root.after(250, lambda s=slot, p=process: self._embed_mediapipe_window(s, p, 0))
        self.connection.set("Cameras 0 and 3 opening")
        self._append_log("Started camera display on cameras [0, 3].\n")
        self._set_running_controls(False)

    def choose_saved_calibration_for_tracking(self) -> bool:
        self.calibrations_folder.mkdir(exist_ok=True)
        initial = self.calibrations_folder if any(self.calibrations_folder.glob("*.json")) else self.project_folder
        selected = filedialog.askopenfilename(title="Choose calibration for live tracking", initialdir=initial, filetypes=[("Calibration files", "*.json"), ("All files", "*.*")])
        if not selected: return False
        source = Path(selected)
        saved = self._read_calibration(source)
        if saved is None:
            messagebox.showerror("Calibration cannot be used", "This file is invalid or was recorded with a different IMU selection.")
            return False
        try:
            for target in (self.calibration_file, self.stage1_file):
                if source.resolve() != target.resolve(): shutil.copy2(source, target)
        except OSError as error:
            messagebox.showerror("Calibration could not be loaded", str(error))
            return False
        self.calibration_ready = True
        self._force_calibrated_mode()
        self.calibration_completed_this_run = True
        self.status.set(f"Calibrated — {source.stem}")
        self.connection.set("Saved calibration loaded; starting live tracking")
        self.calibration_stage_text.set("Calibration selected")
        self.calibration_text.set(f"Using saved calibration: {source.stem}")
        self.calibration_quality_text.set("Ready for live tracking")
        self._append_log(f"CALIBRATION_SELECTED_FOR_TRACKING {source}\n")
        self._set_running_controls(False)
        return True

    def start_tracking(self, raw_requested: bool = False) -> None:
        self.raw_mode.set(bool(raw_requested))
        if not raw_requested:
            if not self.choose_saved_calibration_for_tracking(): return
        self.save_display_controls()
        # The dashboard owns the serial port while idle; release it before the
        # tracking subprocess opens the same port.
        if self.health_receiver is not None:
            self.health_receiver.stop()
            self.health_receiver = None
        if self.process is not None and self.process.poll() is None:
            self.status.set("Waiting for the previous process to close")
            return
        try: gain = float(self.filter_gain.get())
        except ValueError: messagebox.showerror("Invalid gain", "Enter a numeric Madgwick gain."); return
        bias_option = "--use-gyro-bias" if self.use_gyro_bias.get() else "--no-use-gyro-bias"
        if self.full_recording_process is None or self.full_recording_process.poll() is not None:
            if not self.start_full_recording(): return
        if not any(process.poll() is None for process in self.mediapipe_processes.values()):
            self.start_mediapipe_recording()
        self._hide_reference_image()
        command = self._python_command("live_tracking/Live_Tracking.py", "--filter", MADGWICK_FILTER, "--gain", str(gain), bias_option)
        if self.full_recording_folder:
            command += ["--recording-folder", str(self.full_recording_folder)]
        self._launch("tracking", command)
        self.status.set("Tracking")
        self.connection.set("Opening MainBoard serial port")
        self.last_frame.set("Waiting for IMU frames")
        self._set_running_controls(True)
        self._append_log(f"Started calibrated {MADGWICK_FILTER} tracking and live HDF5 recording.\n")

    def stop_mediapipe(self) -> None:
        if not self.mediapipe_processes:
            return
        stopped = False
        for process in list(self.mediapipe_processes.values()):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired: process.kill()
                stopped = True
        self.mediapipe_processes.clear()
        self._clear_embedded_mediapipe_windows()
        self._hide_camera_docks()
        if stopped: self._append_log("Stopped camera display.\n")
        self._set_running_controls(False)

    def stop(self) -> None:
        self.stop_mediapipe()
        self._terminate_process()
        self.stop_full_recording()
        self._clear_embedded_viewer()
        self.status.set("Raw tracking mode" if self.raw_mode.get() else ("Calibrated" if self.calibration_ready else "Calibration required"))
        self.connection.set("Saved calibration loaded — press Start tracking to connect")
        self.last_frame.set("No frames being processed")
        self._set_running_controls(False)
        self._append_log("Stopped.\n")
        self._refresh_stage1_resume()

    def _terminate_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.kill()
        self.process = None
        self.process_mode = None

    def _run_startup_health_check(self) -> None:
        receiver = LiveIMUSerial()
        self.health_receiver = receiver
        receiver.start()
        deadline = time.monotonic() + 12.0
        latest_missing: list[str] | None = None
        consecutive_complete_frames = 0
        while not self.closing and time.monotonic() < deadline:
            frame = receiver.get_nowait()
            if frame is None:
                time.sleep(0.05)
                continue
            latest_missing = [CHANNEL_TO_SITE[channel_id] for channel_id, record in sorted(frame.records.items()) if not record.available]
            consecutive_complete_frames = consecutive_complete_frames + 1 if not latest_missing else 0
            if consecutive_complete_frames >= 3: break
            time.sleep(0.05)
        error = receiver.error
        port = receiver.port_name
        # Keep the receiver alive after the startup check.  The dashboard can
        # therefore continue showing live IMU health while tracking is idle.
        # It is released by start_tracking() before the tracking subprocess
        # opens the serial port.
        if self.closing: return
        if consecutive_complete_frames >= 3: self.health_queue.put({"result": "passed", "port": port})
        elif latest_missing is not None: self.health_queue.put({"result": "failed", "port": port, "missing": latest_missing})
        else: self.health_queue.put({"result": "no_data", "port": port, "error": error})

    def _start_startup_health_check(self) -> None:
        if self.startup_check_running and self.health_receiver is not None: return
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Tracking is running", "Stop calibration or live tracking before reconnecting to the ESP serial port.")
            return
        self.startup_check_running = True
        self.status.set(f"Checking all {len(IMU_NAMES)} IMUs")
        self.connection.set("Opening MainBoard serial port")
        self.last_frame.set("Waiting for firmware startup results")
        for name in self.imu_health_labels: self._set_imu_health(name, "waiting", "Checking", "Running startup check")
        self._set_running_controls(False)
        threading.Thread(target=self._run_startup_health_check, name="startup-imu-check", daemon=True).start()

    def _finish_startup_health_check(self, result: dict) -> None:
        self.startup_check_running = False
        self._set_running_controls(False)
        if result["result"] == "passed":
            for name in self.imu_health_labels: self._set_imu_health(name, "good", "Good")
            self.status.set(f"Hardware ready — {len(IMU_NAMES)}/{len(IMU_NAMES)} IMUs passed")
            self.connection.set(f"Startup check completed on {result.get('port') or 'serial'}")
            self.last_frame.set("All channels configured and streaming")
            self._append_log(f"STARTUP_IMU_CHECK PASS {len(IMU_NAMES)}/{len(IMU_NAMES)} IMUs configured and streaming.\n")
            messagebox.showinfo("IMU startup check", f"All {len(IMU_NAMES)} IMUs passed the startup check and are streaming correctly.")
        elif result["result"] == "failed":
            missing = result.get("missing", [])
            for name in self.imu_health_labels: self._set_imu_health(name, "bad", "Error", "Startup check: no response") if name in missing else self._set_imu_health(name, "good", "Good")
            names = "\n".join(f"• {name}" for name in missing)
            self.status.set(f"IMU check: {len(IMU_NAMES) - len(missing)}/{len(IMU_NAMES)} passed")
            self.connection.set(f"Startup check completed on {result.get('port') or 'serial'}")
            self.last_frame.set("Unavailable: " + ", ".join(missing))
            self._append_log(f"STARTUP_IMU_CHECK FAIL unavailable={','.join(missing)}\n")
            messagebox.showwarning("IMU startup check", f"{16 - len(missing)} of 16 IMUs passed.\n\nUnavailable IMUs:\n{names}")
        else:
            detail = result.get("error") or "No valid MainBoard frames were received within 12 seconds."
            for name in self.imu_health_labels: self._set_imu_health(name, "waiting", "Unknown", "Startup check received no data")
            self.status.set("IMU startup check could not run")
            self.connection.set(detail)
            self.last_frame.set("No valid firmware frame received")
            self._append_log(f"STARTUP_IMU_CHECK NO_DATA {detail}\n")
            messagebox.showerror("IMU startup check", f"The dashboard could not complete the IMU check.\n\n{detail}")

    def _set_running_controls(self, running: bool) -> None:
        running = self.process is not None and self.process.poll() is None
        blocked = running or self.startup_check_running
        self.recheck_button.configure(state="disabled" if blocked else "normal")
        self.calibrate_button.configure(state="disabled" if blocked else "normal")
        self.skip_calibration_button.configure(state="disabled" if running else "normal")
        self.mediapipe_button.configure(state="normal")
        full_recording_running = self.full_recording_process is not None and self.full_recording_process.poll() is None
        self.start_full_recording_button.configure(state="disabled" if full_recording_running else "normal")
        self.stop_full_recording_button.configure(state="normal" if full_recording_running else "disabled")
        calibration_running = running and self.process_mode in {"calibration", "pose_calibration"}
        self.calibration_step_button.configure(state="normal" if calibration_running and self.calibration_step_ready else "disabled")
        skippable = self.process_mode == "pose_calibration" and self.current_calibration_step != "stationary"
        self.skip_step_button.configure(state="normal" if self.process_mode in {"calibration", "pose_calibration"} and self.calibration_step_ready and skippable else "disabled")
        self.back_step_button.configure(state="normal" if self.process_mode in {"calibration", "pose_calibration"} and self.calibration_step_ready and self.calibration_can_go_back else "disabled")
        self.accept_step_button.configure(state="normal" if calibration_running and self.calibration_step_review else "disabled")
        self.redo_step_button.configure(state="normal" if calibration_running and self.calibration_step_review else "disabled")
        final_review = running and self.process_mode in {"calibration", "pose_calibration"} and self.stage1_final_review
        self.finish_stage1_button.configure(state="normal" if final_review else "disabled")
        self.duration_entry.configure(state="disabled" if blocked else "normal")
        self.gain_entry.configure(state="disabled" if blocked else "normal")
        self.gain_slider.configure(state="disabled" if blocked else "normal")
        self.bias_check.configure(state="disabled" if blocked else "normal")
        # Keep this runtime diagnostic control available while live tracking
        # is active; changing it only affects the final displayed qpos.
        self.middle_abduction_check.configure(state="normal")
        self.pose_file_button.configure(state="disabled" if blocked else "normal")
        self.start_button.configure(state="disabled" if blocked else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _has_saved_stage1(self) -> bool:
        return self._find_saved_stage1() is not None

    def _read_calibration(self, path: Path) -> dict | None:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            enabled_now = [name for name in IMU_NAMES if self.imu_enabled[name].get()]
            return saved if saved.get("enabled_imus") == enabled_now else None
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError): return None

    def _find_saved_stage1(self) -> dict | None:
        for path in (self.stage1_file, self.candidate_file, self.calibration_file):
            saved = self._read_calibration(path)
            if saved and saved.get("guided_imu", {}).get("flexion") and saved.get("guided_imu", {}).get("ranges"):
                saved["calibration_mode"] = "stage1_only"
                return saved
        return None

    def _activate_calibration(self, saved: dict, description: str) -> None:
        guided = saved.get("guided_imu", {})
        saved["guided_imu"] = {key: value for key, value in guided.items() if key in {"sample_rate_hz", "neutral_mean_accel_m_s2", "neutral_mean_gyro_rad_s", "neutral_qpos", "flexion", "abduction", "ranges", "endpoint_holds", "sensor_to_segment_alignment", "stage1_endpoint_mapping", "pose_snapshot_calibration"}}
        saved["calibration_mode"] = "imu_only"
        self.stage1_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        self.calibration_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        self.calibration_ready = True
        self._force_calibrated_mode()
        self.calibration_completed_this_run = True
        self.stage1_complete = False
        self.status.set(f"Calibrated — {description}")
        self.connection.set("Listener inactive")
        self.calibration_stage_text.set(f"Calibration selected: {description}")
        self.calibration_text.set(f"Using {description}. Live tracking is ready.")
        self.calibration_motion_text.set("IMU-only calibration loaded")
        self.calibration_quality_text.set("Selected calibration loaded successfully")
        self.calibration_progress.set(100)
        self.calibration_overall_progress.set(100)
        self._set_running_controls(False)

    def _save_stage1_version(self) -> None:
        saved = self._read_calibration(self.candidate_file)
        if saved is None: return
        guided = saved.get("guided_imu", {})
        saved["guided_imu"] = {key: value for key, value in guided.items() if key in {"sample_rate_hz", "neutral_mean_accel_m_s2", "neutral_mean_gyro_rad_s", "neutral_qpos", "flexion", "abduction", "ranges", "endpoint_holds", "sensor_to_segment_alignment", "stage1_endpoint_mapping", "pose_snapshot_calibration"}}
        saved["calibration_mode"] = "stage1_only"
        self.stage1_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        # Stage-1 calibration is the active calibration for direct Start
        # Tracking; keep the runtime file in sync with the saved candidate.
        self.calibration_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        self.calibrations_folder.mkdir(parents=True, exist_ok=True)
        named_file = self.calibrations_folder / f"Calibration_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
        named_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        self._append_log(f"CALIBRATION_NAMED_SAVED {named_file}\n")

    def _refresh_stage1_resume(self) -> None:
        return

    def _refresh_saved_controls(self) -> None: self._refresh_stage1_resume()

    def _save_disabled_imus(self) -> None:
        disabled = [name for name, enabled in self.imu_enabled.items() if not enabled.get()]
        (self.live_folder / "disabled_imus.json").write_text(json.dumps(disabled), encoding="utf-8")

    def _imu_selection_changed(self) -> None:
        self._save_disabled_imus()
        running = self.process is not None and self.process.poll() is None
        if running and self.process_mode == "tracking":
            self.status.set("Tracking — IMU selection updated")
            self.calibration_text.set("Live IMU selection changed. Disabled sensors are now fed to the filters as zero.")
            self._append_log("LIVE_IMU_SELECTION_UPDATED\n")
            self._set_running_controls(True)
            return
        self.calibration_ready = False
        self.calibration_completed_this_run = False
        self.stage1_complete = False
        self.calibration_progress.set(0)
        self.status.set("Calibration required")
        self.calibration_text.set("IMU selection changed; recalibration is required.")
        self.calibration_stage_text.set("Stage: waiting")
        self.calibration_motion_text.set("Stability: not measuring")
        self.calibration_samples_text.set("Valid samples: 0")
        self.calibration_quality_text.set("Quality check: pending")
        self._set_running_controls(False)
    def _set_all_imus(self, enabled: bool) -> None:
        for value in self.imu_enabled.values(): value.set(enabled)
        self._imu_selection_changed()

    def _gain_slider_changed(self, exponent: str) -> None:
        self.filter_gain.set(f"{10.0 ** float(exponent):.6g}")

    def _multiply_gain(self, factor: float) -> None:
        try: value = float(self.filter_gain.get()) * factor
        except ValueError: messagebox.showerror("Invalid value", "Madgwick gain must be numeric."); return
        self.filter_gain.set(f"{value:.6g}")
        self._custom_gain_changed()

    def _custom_gain_changed(self) -> None:
        try: value = float(self.filter_gain.get())
        except ValueError: messagebox.showerror("Invalid value", "Madgwick gain must be numeric."); return
        if value < 0: messagebox.showerror("Invalid value", "Madgwick gain cannot be negative."); return
        if value > 0: self.gain_slider_value.set(max(-5.0, min(3.0, math.log10(value))))

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None: return
        for line in process.stdout: self.output_queue.put(line)

    def _display_step(self, local_step: int, local_total: int) -> tuple[int, int]:
        return local_step, local_total

    def _set_overall_step_progress(self, local_step: int, local_total: int, fraction: float = 0.0) -> None:
        display_step, display_total = self._display_step(local_step, local_total)
        completed = max(0.0, display_step - 1 + fraction)
        self.calibration_overall_progress.set(100.0 * completed / display_total)

    def _poll(self) -> None:
        while True:
            try: result = self.health_queue.get_nowait()
            except queue.Empty: break
            self._finish_startup_health_check(result)
        if not self.startup_check_running and self.health_receiver is not None and self.process is None:
            while True:
                frame = self.health_receiver.get_nowait()
                if frame is None: break
                for channel_id, record in frame.records.items():
                    name = CHANNEL_TO_SITE.get(channel_id)
                    if name not in self.imu_health_labels: continue
                    if record.diagnostic_code == 2 or (not record.available and record.diagnostic_code == 0): colour, label = "red", "WHO_AM_I / no response"
                    elif record.diagnostic_code == 5: colour, label = "purple", "Reinitialisation failed"
                    elif record.diagnostic_code == 4: colour, label = "blue", "SPI read failed"
                    elif record.stale_scans > 0: colour, label = "white", "Stale samples"
                    elif record.diagnostic_code == 3: colour, label = "pink", "Configuration changed"
                    elif record.diagnostic_code == 1 or not (record.acc_fresh and record.gyro_fresh): colour, label = "yellow", "No data ready"
                    else: colour, label = "good", "Good"
                    detail = f"{label} · WHO {record.who_am_i:02X} XL {record.ctrl1_xl:02X} G {record.ctrl2_g:02X} S {record.status_register:02X} · stale {record.stale_scans}"
                    self._set_imu_health(name, colour, label, detail)
        while True:
            try: line = self.output_queue.get_nowait()
            except queue.Empty: break
            stripped = line.strip()
            self._write_log_file(line)
            if not stripped.startswith("IMU_HEALTH "): self._append_log(line, write_file=False)
            if stripped.startswith("CALIBRATION_PLAN "):
                plan = json.loads(stripped.removeprefix("CALIBRATION_PLAN "))
                self.active_calibration_steps = [item["code"] for item in plan]
                self.active_calibration_names = {item["code"]: item["name"] for item in plan}
                step, total = self._display_step(0, len(plan))
                self.calibration_step_counter.set(f"Step {step} of {total}")
            elif stripped == "CALIBRATION_REFERENCE_HIDE": self._hide_reference_image()
            elif stripped.startswith("CALIBRATION_REFERENCE_IMAGE "):
                self._show_reference_image(json.loads(stripped.removeprefix("CALIBRATION_REFERENCE_IMAGE ")))
            elif stripped.startswith("CALIBRATION_PROGRESS "):
                parts = stripped.split()
                self.calibration_progress.set(float(parts[1]))
                progress_total = max(1, len(self.active_calibration_steps))
                self._set_overall_step_progress(1, progress_total, float(parts[1]) / 100.0)
                self.calibration_text.set(f"Collecting stationary measurements: {parts[1]}% ({parts[2]}/{parts[3]} frames)")
                self.calibration_samples_text.set(f"Valid samples: {parts[2]} of {parts[3]}")
                self.calibration_motion_text.set("Stability: stable; collecting")
            elif stripped.startswith("CALIBRATION_STEP_READY "):
                parts = stripped.split(" ", 3)
                steps = self.active_calibration_steps
                names = self.active_calibration_names
                step_index = steps.index(parts[1]) + 1 if parts[1] in steps else 0
                self.calibration_step_ready = True
                self.current_calibration_step = parts[1]
                self.calibration_step_review = False
                self.calibration_can_go_back = False
                self.stage1_final_review = False
                display_step, display_total = self._display_step(step_index, len(steps))
                self.calibration_stage_text.set("Calibration")
                self.calibration_step_counter.set(f"Step {display_step} of {display_total} — {names.get(parts[1], parts[1].replace('_', ' '))}")
                self.calibration_text.set(parts[3])
                self.calibration_motion_text.set("Ready: position your hand, then press Start this step")
                self.calibration_samples_text.set(f"Capture duration: {parts[2]} seconds")
                self.calibration_progress.set(0)
                if step_index: self._set_overall_step_progress(step_index, len(steps))
                self._set_running_controls(True)
            elif stripped == "CALIBRATION_CAN_GO_BACK":
                self.calibration_can_go_back = True
                self._set_running_controls(True)
            elif stripped.startswith("CALIBRATION_WENT_BACK "):
                self.calibration_can_go_back = False
                self.calibration_quality_text.set(f"Returned from {stripped.removeprefix('CALIBRATION_WENT_BACK ')}; the previous capture will be replaced")
            elif stripped.startswith("CALIBRATION_STEP_PROGRESS "):
                parts = stripped.split()
                steps = self.active_calibration_steps
                step_index = steps.index(parts[1]) + 1 if parts[1] in steps else 0
                self.calibration_step_ready = False
                self.calibration_step_review = False
                self.calibration_progress.set(float(parts[2]))
                if step_index: self._set_overall_step_progress(step_index, len(steps), float(parts[2]) / 100.0)
                self.calibration_samples_text.set(f"{parts[1].replace('_', ' ')}: {parts[3]} of {parts[4]} frames")
                self.calibration_motion_text.set("Capturing movement")
                self.calibration_step_button.configure(state="disabled")
            elif stripped.startswith("CALIBRATION_STEP_REVIEW "):
                parts = stripped.split(" ", 2)
                self.calibration_step_ready = False
                self.calibration_step_review = True
                self.calibration_text.set(parts[2])
                self.calibration_motion_text.set("Review the live MuJoCo movement before deciding")
                self.calibration_quality_text.set("Choose Accept this step or Redo only this step")
                self.calibration_progress.set(100)
                self._set_running_controls(True)
            elif stripped.startswith("CALIBRATION_STEP_REDO "):
                self.calibration_step_review = False
                self.calibration_progress.set(0)
                self.calibration_quality_text.set("Previous capture discarded; repeat this step")
            elif stripped.startswith("CALIBRATION_STEP_SKIPPED "):
                code = stripped.removeprefix("CALIBRATION_STEP_SKIPPED ")
                self.calibration_step_ready = False
                self.calibration_step_review = False
                self.calibration_quality_text.set(f"Skipped: {self.active_calibration_names.get(code, code)}; this pose contributes no correction to the fit")
            elif stripped.startswith("CALIBRATION_MAPPING_WARNING "):
                self.calibration_quality_text.set(stripped.removeprefix("CALIBRATION_MAPPING_WARNING "))
            elif stripped.startswith("CALIBRATION_PALM_WRIST "):
                self.calibration_quality_text.set(json.loads(stripped.removeprefix("CALIBRATION_PALM_WRIST "))["summary"])
            elif stripped.startswith("CALIBRATION_STAGE1_FINAL_REVIEW "):
                self.calibration_step_ready = False
                self.calibration_step_review = False
                if self.process_mode == "pose_calibration" and self.full_calibration_flow:
                    self.stage1_final_review = True
                    self.calibration_stage_text.set("Calibration complete: final acceptance")
                    self.calibration_text.set("All required holds are complete. Accept or redo a step.")
                    self.calibration_motion_text.set("Press Accept and save to finish.")
                    self.finish_stage1_button.configure(text="✓  Accept and save")
                elif self.process_mode == "calibration" and self.full_calibration_flow:
                    self.stage1_final_review = False
                    self._hide_reference_image()
                    self.calibration_stage_text.set("Calibration capture complete")
                    self.calibration_step_counter.set(f"All {len(self.active_calibration_steps)} calibration steps captured")
                    self.calibration_text.set("Opening calibration review.")
                    self.calibration_motion_text.set("Saving the captured and skipped steps")
                    if self.process is not None and self.process.stdin is not None:
                        self.process.stdin.write("ACCEPT_STAGE1\n")
                        self.process.stdin.flush()
                else:
                    self.stage1_final_review = True
                    self.calibration_stage_text.set("Calibration complete: final review")
                    self.calibration_text.set(stripped.removeprefix("CALIBRATION_STAGE1_FINAL_REVIEW "))
                    self.calibration_motion_text.set("Move the glove in the viewer, then save calibration or redo one selected block")
                    self.finish_stage1_button.configure(text="Open calibration review" if self.full_calibration_flow else "✓  Accept and save")
                self._set_running_controls(True)
            elif stripped.startswith("CALIBRATION_STEP_FAILED "):
                parts = stripped.split(" ", 2)
                self.calibration_motion_text.set(f"Step failed: {parts[2]}")
                self.calibration_quality_text.set("Quality check: repeat this step")
                self.calibration_progress.set(0)
            elif stripped.startswith("CALIBRATION_RESET "):
                self.calibration_progress.set(0)
                self.calibration_text.set(stripped.removeprefix("CALIBRATION_RESET "))
                self.calibration_samples_text.set("Valid samples: 0; progress restarted")
                self.calibration_motion_text.set("Stability: significant sustained movement")
            elif stripped.startswith("CALIBRATION_MOTION "):
                self.calibration_motion_text.set(stripped.removeprefix("CALIBRATION_MOTION "))
            elif stripped.startswith("CALIBRATION_STABLE "):
                self.calibration_motion_text.set(stripped.removeprefix("CALIBRATION_STABLE "))
            elif stripped.startswith("CALIBRATION_WAITING "): self.connection.set(stripped.removeprefix("CALIBRATION_WAITING "))
            elif stripped.startswith("CALIBRATION_VALIDITY "):
                parts = stripped.split()
                self.calibration_quality_text.set(f"Quality check: {parts[1]} {parts[2]}")
            elif stripped.startswith("CALIBRATION_STATIONARY_WARNING "):
                self.calibration_quality_text.set(stripped.removeprefix("CALIBRATION_STATIONARY_WARNING "))
            elif stripped.startswith("CALIBRATION_COMPLETE "):
                self.stage1_complete = True
                self._save_stage1_version()
                self.calibration_progress.set(100)
                saved = self._find_saved_stage1()
                if saved is not None: self._activate_calibration(saved, "IMU calibration")
                self.calibration_overall_progress.set(100)
                completed = len(self.active_calibration_steps)
                self.calibration_step_counter.set(f"All {completed} calibration steps complete")
                self.calibration_text.set("Calibration saved. Live tracking is ready.")
                self.status.set("Calibrated")
                self.calibration_stage_text.set("Calibration complete")
                self.calibration_quality_text.set("Calibration accepted and saved")
            elif stripped.startswith("QUICK_CALIBRATION_COMPLETE "):
                self.calibration_ready = True
                if self.raw_mode.get():
                    self.raw_mode.set(False)
                    self.save_display_controls()
                self.calibration_completed_this_run = True
                self.calibration_progress.set(100)
                self.calibration_overall_progress.set(100)
                self.calibration_step_counter.set("Quick calibration: 2 of 2 steps complete")
                self.calibration_text.set("Quick calibration saved and activated. Live tracking is ready.")
                self.status.set("Quick calibration ready")
                self.calibration_stage_text.set("Quick calibration complete")
                self.calibration_quality_text.set("Stationary bias and neutral reference saved")
            elif stripped.startswith("CALIBRATION_ERROR "):
                self.calibration_quality_text.set(stripped.removeprefix("CALIBRATION_ERROR "))
            elif stripped.startswith("MUJOCO_HWND "):
                try: self._dock_mujoco_window(int(stripped.split()[1]))
                except (ValueError, IndexError): self._append_log("Could not read the MuJoCo native window handle.\n")
            elif stripped.startswith("MEDIAPIPE_HWND "):
                try:
                    _, slot, hwnd = stripped.split()
                    self._dock_mediapipe_window(int(slot), int(hwnd))
                except (ValueError, IndexError): self._append_log("Could not read the MediaPipe native window handle.\n")
            elif "MainBoard serial connected" in stripped: self.connection.set(stripped)
            elif stripped.startswith("IMU_HEALTH "):
                _, name, health, stale, outliers, code, who, ctrl1, ctrl2, sensor_status = stripped.split()
                if name in self.imu_health_labels:
                    if code == "2" or health != "GOOD" and who == "00": colour, label = "red", "WHO_AM_I / no response"
                    elif code == "5": colour, label = "purple", "Reinitialisation failed"
                    elif code == "4": colour, label = "blue", "SPI read failed"
                    elif int(stale) > 0: colour, label = "white", "Stale samples"
                    elif code == "3": colour, label = "pink", "Configuration changed"
                    elif code == "1": colour, label = "yellow", "No data ready"
                    elif int(outliers) > 0: colour, label = "grey", "Held/outlier samples"
                    else: colour, label = "good", "Good"
                    detail = f"{label} · WHO {who} XL {ctrl1} G {ctrl2} S {sensor_status} · stale {stale} · held {outliers}"
                    self._set_imu_health(name, colour, label, detail)
            elif stripped.startswith("Serial "):
                parts = stripped.split(" | ")
                self.connection.set(parts[0])
                if len(parts) > 1: self.last_frame.set(parts[1])
        finished_mediapipe_slots = [slot for slot, process in self.mediapipe_processes.items() if process.poll() is not None]
        for slot in finished_mediapipe_slots:
            self.mediapipe_processes.pop(slot, None)
            self.embedded_mediapipe_hwnds.pop(slot, None)
            if hasattr(self, "mediapipe_placeholders") and slot in self.mediapipe_placeholders:
                self.mediapipe_placeholders[slot].configure(text=f"Camera {slot} dock")
                self.mediapipe_placeholders[slot].place(relx=0.5, rely=0.5, anchor="center")
        if finished_mediapipe_slots: self._set_running_controls(False)
        if self.full_recording_process is not None and self.full_recording_process.poll() is not None:
            folder = self.full_recording_folder
            self.full_recording_process = None
            if folder:
                self._write_full_recording_metadata(folder, "ended")
                self._copy_full_recording_sidecars(folder)
                self.full_recording_status.set(f"Full recording ended: {folder.name}")
                self._append_log(f"FULL_RECORDING_ENDED {folder}\\n")
            self._set_running_controls(False)
        if self.process is not None and self.process.poll() is not None:
            return_code = self.process.returncode
            finished_mode = self.process_mode
            self.process = None
            self.process_mode = None
            self._clear_embedded_viewer()
            if finished_mode in {"calibration", "pose_calibration"} and return_code == 0:
                self.status.set("Calibrated" if self.calibration_ready else "Calibration complete")
            elif finished_mode == "tracking" and return_code == 0: self.status.set("Calibrated")
            else: self.status.set(f"{finished_mode or 'Process'} failed (code {return_code})")
            self.connection.set("Listener inactive")
            self._set_running_controls(False)
            if finished_mode == "calibration" and return_code == 0: self.full_calibration_flow = False
            elif return_code != 0: self.full_calibration_flow = False
        self.root.after(100, self._poll)

    def _write_log_file(self, text: str) -> None:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if not text.endswith("\n"):
                text += "\n"
            self.dashboard_log_handle.write(f"[{timestamp}] {text}")
        except OSError:
            pass

    def capture_dashboard_views(self, _event=None) -> str:
        """Save the dashboard, MuJoCo, and MediaPipe views in one timestamped folder."""
        if ImageGrab is None:
            self._append_log("Screenshot capture unavailable: Pillow is not installed.\n")
            return "break"
        folder = self.project_folder / "screenshots" / f"Capture_{datetime.now():%Y%m%d_%H%M%S}"
        folder.mkdir(parents=True, exist_ok=True)
        targets = [("Dashboard", int(self.root.winfo_id()))]
        if self.embedded_viewer_hwnd:
            targets.append(("MuJoCo", int(self.embedded_viewer_hwnd)))
        for slot, hwnd in sorted(self.embedded_mediapipe_hwnds.items()):
            targets.append((f"MediaPipe_{slot + 1}", int(hwnd)))
        saved = []
        for name, hwnd in targets:
            try:
                image = ImageGrab.grab(window=hwnd)
                path = folder / f"{name}.png"
                image.save(path)
                saved.append(path.name)
            except Exception as exc:
                self._append_log(f"Screenshot failed for {name}: {exc}\n")
        self._append_log(f"Screenshots saved to {folder} ({', '.join(saved)})\n")
        return "break"

    def _append_log(self, text: str, write_file: bool = True) -> None:
        if write_file:
            self._write_log_file(text)
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def close(self) -> None:
        self.closing = True
        if self.health_receiver is not None: self.health_receiver.stop()
        self.stop_full_recording()
        self.stop()
        try: self.dashboard_log_handle.close()
        except OSError: pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    MainBoardDashboard(root)
    root.mainloop()


if __name__ == "__main__": main()






