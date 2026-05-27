from __future__ import annotations

import base64
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.camera_utils import AUTO_CAMERA_SOURCE, list_available_cameras, open_camera_source
from src.config import CAPTURES_DIR, LOGS_DIR, KNOWN_FACES_DIR, ensure_project_dirs, resource_path
from src.face_store import (
    POSES,
    KnownPerson,
    display_name_from_path,
    list_known_people,
    save_pose_set,
)

NOTIFICATION_ICON_PATH = resource_path("assets/notifications.png")
APP_ICON_PATH = resource_path("assets/app_icon.ico")
AUTO_CAMERA_LABEL = "auto - Tự động"

APP_BG = "#eef2f6"
SURFACE = "#ffffff"
SURFACE_2 = "#f8fafc"
TEXT = "#17202a"
MUTED = "#64748b"
PRIMARY = "#0f766e"
PRIMARY_DARK = "#115e59"
DANGER = "#b91c1c"
BORDER = "#d8e0ea"
OVERLAY_FONT_PATHS = [
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]


class KnownFacesAdminApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        ensure_project_dirs()
        self.title("Quản lý người quen")
        if APP_ICON_PATH.exists():
            self.iconbitmap(str(APP_ICON_PATH))
        self.geometry("1080x660")
        self.minsize(980, 600)
        self.configure(bg=APP_BG)
        self.selected_path: Path | None = None
        self.preview_image: tk.PhotoImage | None = None
        self.people: list[KnownPerson] = []
        self.search_var = tk.StringVar()
        self.camera_source_options: dict[str, str] = {AUTO_CAMERA_LABEL: AUTO_CAMERA_SOURCE}
        self.camera_source_var = tk.StringVar(value=AUTO_CAMERA_LABEL)
        self.camera_hint_var = tk.StringVar(value="")
        self.notifications: list[dict[str, str]] = []
        self.notification_keys: set[str] = set()
        self.has_unread_notifications = False
        self.bell_canvas: tk.Canvas | None = None
        self.bell_icon: tk.PhotoImage | None = None

        self._build_layout()
        self.refresh_camera_devices()
        self.refresh_faces()
        self.load_notifications(initial=True)
        self.poll_notifications()

    def _build_layout(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", rowheight=34, font=("Segoe UI", 10), background=SURFACE, fieldbackground=SURFACE)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#e8eef5", foreground=TEXT)
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("Small.TButton", font=("Segoe UI", 9), padding=(8, 4))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 9), background=PRIMARY, foreground="white")
        style.map("Primary.TButton", background=[("active", PRIMARY_DARK)], foreground=[("active", "white")])
        style.configure("Danger.TButton", font=("Segoe UI", 10), padding=(12, 8), foreground=DANGER)
        style.configure("Search.TEntry", padding=(8, 8))
        style.configure("TLabel", background=APP_BG, font=("Segoe UI", 10), foreground=TEXT)

        header = tk.Frame(self, bg="#0f172a", padx=22, pady=18)
        header.pack(fill=tk.X)
        title_block = tk.Frame(header, bg="#0f172a")
        title_block.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            title_block,
            text="Quản lý dữ liệu người quen",
            bg="#0f172a",
            fg="white",
            font=("Segoe UI", 19, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            title_block,
            text="Thêm, chụp và quản lý mẫu khuôn mặt dùng cho hệ thống giám sát.",
            bg="#0f172a",
            fg="#cbd5e1",
            font=("Segoe UI", 10),
        ).pack(anchor=tk.W, pady=(4, 0))
        self.status_label = tk.Label(
            header,
            text="",
            bg="#0f172a",
            fg="#e2e8f0",
            font=("Segoe UI", 11, "bold"),
        )
        self.status_label.pack(side=tk.RIGHT, padx=(16, 0))
        self.bell_canvas = tk.Canvas(
            header,
            width=52,
            height=46,
            bg="#0f172a",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.bell_canvas.pack(side=tk.RIGHT)
        self.bell_canvas.bind("<Button-1>", lambda _event: self.open_notifications())
        self.draw_bell()

        body = tk.Frame(self, bg=APP_BG, padx=18, pady=18)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=APP_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls = tk.Frame(left, bg=SURFACE, padx=14, pady=14, highlightbackground=BORDER, highlightthickness=1)
        controls.pack(fill=tk.X, pady=(0, 12))

        action_row = tk.Frame(controls, bg=SURFACE)
        action_row.pack(fill=tk.X)
        ttk.Button(action_row, text="+ Thêm từ ảnh", style="Primary.TButton", command=self.add_from_image).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Quét bằng camera", style="Primary.TButton", command=self.add_from_camera).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Mở camera giám sát", command=self.open_monitoring_camera).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Xóa người", style="Danger.TButton", command=self.delete_selected).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_row, text="Làm mới", command=self.refresh_faces).pack(side=tk.LEFT)

        camera_row = tk.Frame(controls, bg=SURFACE)
        camera_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(camera_row, text="Thiết bị camera", bg=SURFACE, fg=MUTED, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.camera_source_combo = ttk.Combobox(
            camera_row,
            textvariable=self.camera_source_var,
            values=tuple(self.camera_source_options),
            width=32,
            state="readonly",
        )
        self.camera_source_combo.pack(side=tk.LEFT, padx=(10, 8))
        ttk.Button(camera_row, text="Quét lại", style="Small.TButton", command=self.refresh_camera_devices).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(
            camera_row,
            textvariable=self.camera_hint_var,
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

        search_row = tk.Frame(controls, bg=SURFACE)
        search_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(search_row, text="Tìm kiếm", bg=SURFACE, fg=MUTED, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        search_entry = ttk.Entry(search_row, textvariable=self.search_var, style="Search.TEntry")
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        self.search_var.trace_add("write", lambda *_args: self.apply_filter())

        columns = ("name", "images", "folder", "modified")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("name", text="Tên")
        self.tree.heading("images", text="Số ảnh")
        self.tree.heading("folder", text="Thư mục")
        self.tree.heading("modified", text="Cập nhật")
        self.tree.column("name", minwidth=180, width=260)
        self.tree.column("images", minwidth=80, width=90, anchor=tk.CENTER)
        self.tree.column("folder", minwidth=220, width=310)
        self.tree.column("modified", minwidth=150, width=170)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.tag_configure("odd", background="#f8fafc")
        self.tree.tag_configure("even", background="#ffffff")

        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        right = tk.Frame(body, bg=SURFACE, padx=16, pady=16, width=390, highlightbackground=BORDER, highlightthickness=1)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))
        right.pack_propagate(False)

        tk.Label(
            right,
            text="Thông tin người quen",
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            right,
            text="Chọn một dòng trong danh sách để xem ảnh và thông tin lưu trữ.",
            bg=SURFACE,
            fg=MUTED,
            wraplength=340,
            justify=tk.LEFT,
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(4, 10))
        self.preview_canvas = tk.Canvas(
            right,
            bg=SURFACE_2,
            width=340,
            height=270,
            bd=0,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        self.preview_canvas.pack(fill=tk.X, pady=(12, 12))
        self.preview_canvas.create_text(170, 135, text="Chưa chọn ảnh", fill=MUTED, font=("Segoe UI", 10))

        self.detail_label = tk.Label(
            right,
            text="",
            bg=SURFACE,
            fg=TEXT,
            justify=tk.LEFT,
            anchor=tk.NW,
            font=("Segoe UI", 10),
            wraplength=340,
        )
        self.detail_label.pack(fill=tk.BOTH, expand=True)

        detail_actions = tk.Frame(right, bg=SURFACE)
        detail_actions.pack(fill=tk.X, pady=(10, 0))
        tk.Button(
            detail_actions,
            text="Mở thư mục",
            command=self.open_selected_folder,
            bg="#e8eef5",
            fg=TEXT,
            activebackground="#dbe5f0",
            activeforeground=TEXT,
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            bd=0,
            height=2,
            cursor="hand2",
        ).pack(fill=tk.X)

    def refresh_faces(self) -> None:
        self.people = list_known_people(KNOWN_FACES_DIR)
        self.apply_filter()
        if self.selected_path and not self.selected_path.exists():
            self.clear_preview()

    def apply_filter(self) -> None:
        self.tree.delete(*self.tree.get_children())
        keyword = self.search_var.get().strip().lower()
        filtered_people = [
            person
            for person in self.people
            if not keyword
            or keyword in person.name.lower()
            or keyword in person.folder.name.lower()
        ]
        for index, person in enumerate(filtered_people):
            newest = max((path.stat().st_mtime for path in person.images), default=person.folder.stat().st_mtime)
            modified = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
            self.tree.insert(
                "",
                tk.END,
                iid=str(person.folder),
                values=(person.name, len(person.images), person.folder.name, modified),
                tags=("even" if index % 2 == 0 else "odd",),
            )
        if keyword:
            self.status_label.configure(text=f"{len(filtered_people)}/{len(self.people)} người quen")
        else:
            self.status_label.configure(text=f"{len(self.people)} người quen")

    def add_from_image(self) -> None:
        name = self.ask_name()
        if not name:
            return

        pose_images = {}
        for pose, label in POSES:
            image_path = filedialog.askopenfilename(
                title=f"Chọn ảnh {label}",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                    ("All files", "*.*"),
                ],
            )
            if not image_path:
                return
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("Lỗi", f"Không đọc được ảnh {label}.")
                return
            pose_images[pose] = image

        saved_paths = save_pose_set(name, pose_images, KNOWN_FACES_DIR)
        self.refresh_faces()
        self.select_path(saved_paths[0].parent)
        messagebox.showinfo("Thành công", f"Đã thêm {name} với {len(saved_paths)} góc mặt.")

    def add_from_camera(self) -> None:
        name = self.ask_name()
        if not name:
            return
        CameraCaptureWindow(self, name, self.get_camera_source())

    def get_camera_source(self) -> str:
        selected = self.camera_source_var.get().strip()
        if not selected:
            return AUTO_CAMERA_SOURCE
        return self.camera_source_options.get(selected, selected)

    def refresh_camera_devices(self) -> None:
        current_source = self.get_camera_source()
        options: dict[str, str] = {AUTO_CAMERA_LABEL: AUTO_CAMERA_SOURCE}
        for device in list_available_cameras():
            options[device.label] = str(device.source)

        self.camera_source_options = options
        values = tuple(options)
        self.camera_source_combo.configure(values=values)

        selected_label = AUTO_CAMERA_LABEL
        for label, source in options.items():
            if source == current_source:
                selected_label = label
                break
        self.camera_source_var.set(selected_label)

        camera_count = max(0, len(options) - 1)
        if camera_count:
            self.camera_hint_var.set(f"Tìm thấy {camera_count} camera. Cắm/rút thiết bị xong bấm Quét lại.")
        else:
            self.camera_hint_var.set("Chưa tìm thấy camera có hình. Mở app webcam trên điện thoại rồi bấm Quét lại.")

    def ask_name(self) -> str | None:
        dialog = NameDialog(self)
        self.wait_window(dialog)
        return dialog.value

    def delete_selected(self) -> None:
        path = self.get_selected_path()
        if path is None:
            messagebox.showwarning("Chưa chọn", "Hãy chọn một người cần xóa.")
            return
        if not messagebox.askyesno("Xác nhận", f"Xóa {display_name_from_path(path)}?"):
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        self.selected_path = None
        self.clear_preview()
        self.refresh_faces()

    def open_selected_folder(self) -> None:
        path = self.selected_path or KNOWN_FACES_DIR
        target = path if path.is_dir() else path.parent
        if not target.exists():
            messagebox.showwarning("Không tìm thấy", "Thư mục không tồn tại.")
            return
        os.startfile(target)

    def open_monitoring_camera(self) -> None:
        project_dir = Path(__file__).resolve().parent
        python_executable = sys.executable
        camera_source = self.get_camera_source()
        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW
            if getattr(sys, "frozen", False):
                command = [python_executable, "--monitor", "--source", camera_source]
                cwd = Path(sys.executable).resolve().parent
            else:
                command = [python_executable, str(project_dir / "main.py"), "--source", camera_source]
                cwd = project_dir
            ensure_project_dirs()
            stdout_log = (LOGS_DIR / "monitor_stdout.log").open("a", encoding="utf-8")
            stderr_log = (LOGS_DIR / "monitor_stderr.log").open("a", encoding="utf-8")
            subprocess.Popen(
                command,
                cwd=cwd,
                creationflags=creationflags,
                stdout=stdout_log,
                stderr=stderr_log,
            )
        except Exception as exc:
            messagebox.showerror("Không mở được camera", f"Lỗi khi chạy main.py:\n{exc}")

    def load_notifications(self, initial: bool = False) -> None:
        log_file = LOGS_DIR / "detections.csv"
        if not log_file.exists():
            self.notifications = []
            self.notification_keys = set()
            self.draw_bell()
            return

        previous_keys = set(self.notification_keys)
        notifications: list[dict[str, str]] = []
        keys: set[str] = set()
        with log_file.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("event") != "stranger":
                    continue
                image_name = row.get("image", "")
                timestamp = row.get("timestamp", "")
                key = f"{timestamp}|{image_name}"
                keys.add(key)
                notifications.append(
                    {
                        "key": key,
                        "timestamp": timestamp,
                        "message": self.format_notification_message(timestamp),
                        "image": image_name,
                    }
                )

        self.notifications = list(reversed(notifications))
        if not initial and any(key not in previous_keys for key in keys):
            self.has_unread_notifications = True
        self.notification_keys = keys
        self.draw_bell()

    def poll_notifications(self) -> None:
        self.load_notifications()
        self.after(2500, self.poll_notifications)

    def draw_bell(self) -> None:
        if self.bell_canvas is None:
            return
        canvas = self.bell_canvas
        canvas.delete("all")
        canvas.create_oval(5, 4, 45, 44, fill="#1e293b", outline="#334155", width=1)
        if self.bell_icon is None and NOTIFICATION_ICON_PATH.exists():
            self.bell_icon = tk.PhotoImage(file=str(NOTIFICATION_ICON_PATH))
        if self.bell_icon is not None:
            canvas.create_image(25, 24, image=self.bell_icon, anchor=tk.CENTER)
        else:
            canvas.create_text(25, 25, text="!", fill="#f8fafc", font=("Segoe UI", 17, "bold"))
        if self.has_unread_notifications:
            canvas.create_oval(33, 4, 50, 21, fill="#ef4444", outline="#0f172a", width=2)
            canvas.create_text(41.5, 12.5, text="!", fill="white", font=("Segoe UI", 10, "bold"))

    def open_notifications(self) -> None:
        self.has_unread_notifications = False
        self.draw_bell()
        NotificationWindow(self, self.notifications)

    def format_notification_message(self, timestamp: str) -> str:
        try:
            captured_at = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            formatted = captured_at.strftime("%d/%m/%Y %H:%M:%S")
        except ValueError:
            formatted = timestamp or "không rõ thời gian"
        return f"Đã chụp ảnh được người lạ vào thời điểm {formatted}."

    def open_stranger_folder(self) -> None:
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(CAPTURES_DIR)

    def on_select(self, _event=None) -> None:
        path = self.get_selected_path()
        if path is None:
            return
        self.selected_path = path
        self.show_preview(path)

    def get_selected_path(self) -> Path | None:
        selection = self.tree.selection()
        if not selection:
            return None
        path = Path(selection[0])
        return path if path.exists() else None

    def select_path(self, path: Path) -> None:
        iid = str(path)
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)
        self.selected_path = path
        self.show_preview(path)

    def show_preview(self, path: Path) -> None:
        preview_path = self.first_image_for_path(path)
        if preview_path is None:
            self.clear_preview("Chưa có ảnh")
            return

        image = cv2.imread(str(preview_path))
        if image is None:
            self.clear_preview("Không đọc được ảnh")
            return

        self.preview_image = image_to_photo(image, max_width=340, max_height=270)
        self.preview_canvas.delete("all")
        self.preview_canvas.configure(bg=SURFACE_2)
        self.preview_canvas.create_image(170, 135, image=self.preview_image, anchor=tk.CENTER)
        images = self.images_for_path(path)
        size_kb = sum(image_path.stat().st_size for image_path in images) / 1024
        shown_names = [image_path.name for image_path in images[:6]]
        image_names = ", ".join(shown_names) or "Chưa có"
        if len(images) > 6:
            image_names += f" ... (+{len(images) - 6} ảnh)"
        self.detail_label.configure(
            text=(
                f"Tên: {display_name_from_path(path)}\n"
                f"Số ảnh: {len(images)}\n"
                f"Ảnh: {image_names}\n"
                f"Dung lượng: {size_kb:.1f} KB\n"
                f"Thư mục: {path.name}"
            )
        )

    def clear_preview(self, text: str = "Chưa chọn ảnh") -> None:
        self.preview_image = None
        self.preview_canvas.delete("all")
        self.preview_canvas.configure(bg=SURFACE_2)
        self.preview_canvas.create_text(170, 135, text=text, fill=MUTED, font=("Segoe UI", 10))
        self.detail_label.configure(text="")

    def images_for_path(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        return [
            image_path
            for image_path in sorted(path.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

    def first_image_for_path(self, path: Path) -> Path | None:
        images = self.images_for_path(path)
        if not images:
            return None
        front_images = [image_path for image_path in images if image_path.stem.startswith("front")]
        return front_images[0] if front_images else images[0]


class NotificationWindow(tk.Toplevel):
    def __init__(self, parent: KnownFacesAdminApp, notifications: list[dict[str, str]]) -> None:
        super().__init__(parent)
        self.parent = parent
        self.title("Thông báo người lạ")
        self.geometry("560x420")
        self.minsize(480, 340)
        self.configure(bg=APP_BG, padx=16, pady=16)
        self.transient(parent)

        header = tk.Frame(self, bg=APP_BG)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="Thông báo",
            bg=APP_BG,
            fg=TEXT,
            font=("Segoe UI", 17, "bold"),
        ).pack(side=tk.LEFT)
        tk.Button(
            header,
            text="Mở thư mục người lạ",
            command=parent.open_stranger_folder,
            bg=PRIMARY,
            fg="white",
            activebackground=PRIMARY_DARK,
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        list_frame = tk.Frame(self, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(14, 0))

        canvas = tk.Canvas(list_frame, bg=SURFACE, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=canvas.yview)
        content = tk.Frame(canvas, bg=SURFACE)
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        if not notifications:
            tk.Label(
                content,
                text="Chưa có thông báo người lạ.",
                bg=SURFACE,
                fg=MUTED,
                font=("Segoe UI", 11),
                padx=18,
                pady=18,
            ).pack(anchor=tk.W)
            return

        for notification in notifications:
            item = tk.Frame(content, bg=SURFACE, padx=14, pady=12)
            item.pack(fill=tk.X)
            tk.Label(
                item,
                text=notification["message"],
                bg=SURFACE,
                fg=TEXT,
                font=("Segoe UI", 10),
                wraplength=470,
                justify=tk.LEFT,
            ).pack(anchor=tk.W)
            if notification.get("image"):
                tk.Label(
                    item,
                    text=f"Ảnh: {notification['image']}",
                    bg=SURFACE,
                    fg=MUTED,
                    font=("Segoe UI", 9),
                ).pack(anchor=tk.W, pady=(4, 0))
            tk.Frame(content, bg=BORDER, height=1).pack(fill=tk.X, padx=12)


class NameDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk) -> None:
        super().__init__(parent)
        self.value: str | None = None
        self.title("Nhập tên")
        self.geometry("360x150")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.configure(bg="#f4f6f8", padx=16, pady=14)

        tk.Label(self, text="Tên người quen", bg="#f4f6f8", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.entry = ttk.Entry(self, font=("Segoe UI", 11))
        self.entry.pack(fill=tk.X, pady=(8, 12))
        self.entry.focus_set()

        actions = tk.Frame(self, bg="#f4f6f8")
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="Hủy", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Lưu", command=self.save).pack(side=tk.RIGHT, padx=(0, 8))
        self.bind("<Return>", lambda _event: self.save())
        self.bind("<Escape>", lambda _event: self.destroy())

    def save(self) -> None:
        name = self.entry.get().strip()
        if not name:
            messagebox.showwarning("Thiếu tên", "Hãy nhập tên người quen.", parent=self)
            return
        self.value = name
        self.destroy()


class CameraCaptureWindow(tk.Toplevel):
    def __init__(self, parent: KnownFacesAdminApp, name: str, camera_source: str) -> None:
        super().__init__(parent)
        self.parent = parent
        self.name = name
        self.camera_source = camera_source
        camera_result = open_camera_source(camera_source)
        self.camera = camera_result.camera
        self.current_frame = None
        self.preview_image: tk.PhotoImage | None = None
        self.after_id: str | None = None
        self.pose_index = 0
        self.pose_images = {}
        self.pose_buffer = []
        self.stable_since: float | None = None
        self.last_auto_capture_at = 0.0
        self.required_stable_seconds = 1.4
        self.samples_per_pose = 5
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        self.title(f"Chụp khuôn mặt - {name}")
        self.geometry("720x610")
        self.configure(bg="#111820", padx=14, pady=14)
        self.protocol("WM_DELETE_WINDOW", self.close)

        self.instruction_label = tk.Label(
            self,
            bg="#111820",
            fg="#ffffff",
            font=("Segoe UI", 14, "bold"),
            text=self.current_instruction(),
        )
        self.instruction_label.pack(fill=tk.X, pady=(0, 10))

        self.video_label = tk.Label(self, bg="#000000", fg="white", text="Đang mở camera...")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        actions = tk.Frame(self, bg="#111820", pady=12)
        actions.pack(fill=tk.X)
        self.capture_button = ttk.Button(actions, text="Chụp thủ công", command=self.capture_manual)
        self.capture_button.pack(side=tk.LEFT)
        self.progress_label = tk.Label(
            actions,
            bg="#111820",
            fg="#d6e4f0",
            font=("Segoe UI", 10),
            text="0/3",
        )
        self.progress_label.pack(side=tk.LEFT, padx=12)
        self.scan_label = tk.Label(
            actions,
            bg="#111820",
            fg="#d6e4f0",
            font=("Segoe UI", 10),
            text="Đưa mặt vào khung",
        )
        self.scan_label.pack(side=tk.LEFT)
        ttk.Button(actions, text="Đóng", command=self.close).pack(side=tk.RIGHT)

        if self.camera is None:
            messagebox.showerror(
                "Lỗi camera",
                camera_result.error or "Không mở được webcam.",
                parent=self,
            )
            self.close()
            return

        self.after_id = self.after(20, self.update_frame)

    def update_frame(self) -> None:
        ok, frame = self.camera.read()
        if ok:
            self.current_frame = frame
            display_frame = self.process_scan_frame(frame)
            self.preview_image = image_to_photo(display_frame, max_width=660, max_height=430)
            self.video_label.configure(image=self.preview_image, text="")
        if self.winfo_exists():
            self.after_id = self.after(25, self.update_frame)

    def process_scan_frame(self, frame):
        display = frame.copy()
        height, width = display.shape[:2]
        guide_w = int(width * 0.42)
        guide_h = int(height * 0.58)
        center_x = width // 2
        center_y = height // 2
        guide_x1 = center_x - guide_w // 2
        guide_y1 = center_y - guide_h // 2
        guide_x2 = center_x + guide_w // 2
        guide_y2 = center_y + guide_h // 2

        pose, _label = POSES[self.pose_index]
        pose_check = self.evaluate_pose(frame, pose, (guide_x1, guide_y1, guide_x2, guide_y2))
        face = pose_check["face"]
        is_ready = pose_check["ready"]
        message = pose_check["message"]
        guide_color = (0, 180, 255)

        if face is not None:
            x, y, w, h = face
            color = (40, 190, 80) if is_ready else (0, 180, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            if is_ready:
                guide_color = (40, 190, 80)

        cv2.ellipse(
            display,
            (center_x, center_y),
            (guide_w // 2, guide_h // 2),
            0,
            0,
            360,
            guide_color,
            2,
        )
        draw_unicode_text(display, self.current_instruction_overlay(), (24, 12), font_size=28)

        self.update_scan_state(is_ready, message)
        return display

    def evaluate_pose(self, frame, expected_pose: str, guide: tuple[int, int, int, int]) -> dict:
        guide_x1, guide_y1, guide_x2, guide_y2 = guide
        height, width = frame.shape[:2]
        min_face = int(min(width, height) * 0.18)
        max_face = int(min(width, height) * 0.58)

        frontal = self.detect_largest_face(frame, mode="front")
        profile_left = self.detect_largest_face(frame, mode="profile_left")
        profile_right = self.detect_largest_face(frame, mode="profile_right")

        if expected_pose == "front":
            if frontal is None:
                return {"ready": False, "message": "Nhìn thẳng vào camera", "face": None}
            ready, message = self.validate_common_face(frontal, guide, min_face, max_face)
            if not ready:
                return {"ready": False, "message": message, "face": frontal}
            eyes = self.count_eyes(frame, frontal)
            if eyes < 2:
                return {"ready": False, "message": "Cần thấy rõ 2 mắt", "face": frontal}
            return {"ready": True, "message": "Đúng chính diện, giữ yên...", "face": frontal}

        if expected_pose == "left":
            face = profile_left
            wrong_face = profile_right
            direction_message = "Xoay mặt sang trái rõ hơn"
        else:
            face = profile_right
            wrong_face = profile_left
            direction_message = "Xoay mặt sang phải rõ hơn"

        if face is None:
            if wrong_face is not None:
                return {"ready": False, "message": "Đang sai hướng xoay", "face": wrong_face}
            if frontal is not None and self.count_eyes(frame, frontal) >= 2:
                return {"ready": False, "message": direction_message, "face": frontal}
            return {"ready": False, "message": direction_message, "face": None}

        ready, message = self.validate_common_face(face, guide, min_face, max_face)
        if not ready:
            return {"ready": False, "message": message, "face": face}
        if frontal is not None and self.count_eyes(frame, frontal) >= 2:
            return {"ready": False, "message": direction_message, "face": frontal}
        return {"ready": True, "message": "Đúng hướng, giữ yên...", "face": face}

    def validate_common_face(
        self,
        face: tuple[int, int, int, int],
        guide: tuple[int, int, int, int],
        min_face: int,
        max_face: int,
    ) -> tuple[bool, str]:
        guide_x1, guide_y1, guide_x2, guide_y2 = guide
        x, y, w, h = face
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        centered = guide_x1 < face_center_x < guide_x2 and guide_y1 < face_center_y < guide_y2
        good_size = min_face <= max(w, h) <= max_face
        if not centered:
            return False, "Căn mặt vào giữa khung"
        if not good_size:
            return False, "Tiến lại gần hơn hoặc lùi ra xa"
        return True, "Giữ yên..."

    def update_scan_state(self, is_ready: bool, message: str) -> None:
        now = time.monotonic()
        if not is_ready:
            self.stable_since = None
            self.scan_label.configure(text=message)
            return

        if self.stable_since is None:
            self.stable_since = now

        elapsed = now - self.stable_since
        remaining = max(0.0, self.required_stable_seconds - elapsed)
        self.scan_label.configure(text=f"{message} {remaining:.1f}s")

        if is_ready and self.current_frame is not None:
            self.collect_pose_sample(self.current_frame)

        if (
            elapsed >= self.required_stable_seconds
            and len(self.pose_buffer) >= self.samples_per_pose
            and now - self.last_auto_capture_at > 0.8
        ):
            self.last_auto_capture_at = now
            self.capture_current_pose()

    def detect_largest_face(self, frame, mode: str = "front"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if mode == "front":
            cascade = self.face_cascade
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        elif mode == "profile_left":
            cascade = self.profile_cascade
            faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4)
        else:
            cascade = self.profile_cascade
            flipped = cv2.flip(gray, 1)
            raw_faces = cascade.detectMultiScale(flipped, scaleFactor=1.08, minNeighbors=4)
            faces = []
            width = gray.shape[1]
            for x, y, w, h in raw_faces:
                faces.append((width - x - w, y, w, h))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda item: item[2] * item[3])

    def count_eyes(self, frame, face: tuple[int, int, int, int]) -> int:
        x, y, w, h = face
        roi = frame[y : y + max(1, h // 2), x : x + w]
        if roi.size == 0:
            return 0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return min(2, len(eyes))

    def capture_manual(self) -> None:
        self.capture_current_pose()

    def capture_current_pose(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("Chưa có ảnh", "Camera chưa lấy được khung hình.", parent=self)
            return
        pose, _label = POSES[self.pose_index]
        if self.pose_buffer:
            self.pose_images[pose] = [frame.copy() for frame in self.pose_buffer[-self.samples_per_pose :]]
        else:
            self.pose_images[pose] = [self.current_frame.copy()]
        self.pose_index += 1
        self.stable_since = None
        self.pose_buffer = []
        self.progress_label.configure(text=f"{self.pose_index}/3")

        if self.pose_index < len(POSES):
            self.instruction_label.configure(text=self.current_instruction())
            self.scan_label.configure(text="Chuẩn bị góc tiếp theo")
            return

        saved_paths = save_pose_set(self.name, self.pose_images, KNOWN_FACES_DIR)
        self.parent.refresh_faces()
        self.parent.select_path(saved_paths[0].parent)
        messagebox.showinfo("Thành công", f"Đã thêm {self.name} với 3 góc mặt.", parent=self)
        self.close()

    def current_instruction(self, short: bool = False) -> str:
        _pose, label = POSES[self.pose_index]
        if short:
            return label
        return f"{label}: đưa mặt vào khung, giữ yên để hệ thống tự chụp"

    def current_instruction_overlay(self) -> str:
        pose, _label = POSES[self.pose_index]
        labels = {
            "front": "Chính diện",
            "left": "Xoay trái",
            "right": "Xoay phải",
        }
        return labels.get(pose, pose)

    def collect_pose_sample(self, frame) -> None:
        self.pose_buffer.append(frame.copy())
        if len(self.pose_buffer) > self.samples_per_pose * 2:
            self.pose_buffer = self.pose_buffer[-self.samples_per_pose :]

    def close(self) -> None:
        if self.after_id:
            try:
                self.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        self.destroy()


def image_to_photo(image_bgr, max_width: int, max_height: int, cover: bool = False) -> tk.PhotoImage:
    height, width = image_bgr.shape[:2]
    if cover:
        scale = max(max_width / width, max_height / height)
        resized = cv2.resize(
            image_bgr,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        resized_h, resized_w = resized.shape[:2]
        start_x = max(0, (resized_w - max_width) // 2)
        start_y = max(0, (resized_h - max_height) // 2)
        resized = resized[start_y : start_y + max_height, start_x : start_x + max_width]
    else:
        scale = min(max_width / width, max_height / height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".png", resized)
    if not ok:
        raise ValueError("Không thể mã hóa ảnh xem trước.")
    data = base64.b64encode(buffer).decode("ascii")
    return tk.PhotoImage(data=data)


def draw_unicode_text(
    frame,
    text: str,
    position: tuple[int, int],
    font_size: int = 26,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = load_overlay_font(font_size)
    x, y = position
    shadow_color = (20, 24, 31)
    for offset_x, offset_y in ((-1, 0), (1, 0), (0, -1), (0, 1), (2, 2)):
        draw.text((x + offset_x, y + offset_y), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=fill)
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def load_overlay_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in OVERLAY_FONT_PATHS:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), font_size)
    return ImageFont.load_default()


def main() -> None:
    app = KnownFacesAdminApp()
    app.mainloop()


if __name__ == "__main__":
    if "--monitor" in sys.argv:
        sys.argv = [argument for argument in sys.argv if argument != "--monitor"]
        from main import main as monitor_main

        raise SystemExit(monitor_main())
    main()
