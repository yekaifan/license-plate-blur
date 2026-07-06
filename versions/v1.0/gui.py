"""
车牌自动打码工具 - GUI 版
PySide6 桌面应用，支持拖入图片、设置参数、批量处理、日志查看。
"""

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal, Qt, QMimeData, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QFileDialog, QSplitter, QGroupBox, QLineEdit, QFrame,
    QAbstractItemView, QMessageBox, QProgressBar,
)
from ultralytics import YOLO

# ── 模型 ────────────────────────────────────────────
HF_REPO_ID = "Koushim/yolov8-license-plate-detection"
HF_FILENAME = "best.pt"
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / HF_FILENAME

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── 日志 ────────────────────────────────────────────
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
log = logging.getLogger("plate_blur")
log.addHandler(log_handler)
log.setLevel(logging.INFO)
log.propagate = False


# ══════════════════════════════════════════════════════
# 后台处理线程
# ══════════════════════════════════════════════════════

class BlurWorker(QThread):
    """后台处理线程，避免阻塞 UI。"""
    progress = Signal(int, int)          # current, total
    item_done = Signal(int, str, int, str)  # row, status, count, detail
    log_msg = Signal(str)
    all_done = Signal()

    def __init__(self, items, output_dir, blur_kernel, confidence_threshold, margin):
        super().__init__()
        self.items = items          # [(row, path), ...]
        self.output_dir = output_dir
        self.blur_kernel = blur_kernel
        self.confidence_threshold = confidence_threshold
        self.margin = margin

    def run(self):
        model = YOLO(str(MODEL_PATH))
        total = len(self.items)

        for idx, (row, src) in enumerate(self.items):
            self.progress.emit(idx + 1, total)

            try:
                src = Path(src)
                img = cv2.imread(str(src))
                if img is None:
                    self.item_done.emit(row, "❌ 读取失败", 0, "")
                    self.log_msg.emit(f"❌ 无法读取：{src.name}")
                    continue

                results = model(img, verbose=False)
                boxes = results[0].boxes

                plate_count = 0
                detail_parts = []

                if boxes is not None:
                    h, w = img.shape[:2]
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf < self.confidence_threshold:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1 = max(0, int(x1) - self.margin)
                        y1 = max(0, int(y1) - self.margin)
                        x2 = min(w, int(x2) + self.margin)
                        y2 = min(h, int(y2) + self.margin)

                        roi = img[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue

                        k = self.blur_kernel
                        if k % 2 == 0:
                            k += 1
                        roi = cv2.GaussianBlur(roi, (k, k), 0)
                        img[y1:y2, x1:x2] = roi

                        plate_count += 1
                        detail_parts.append(f"({x1},{y1},{x2},{y2}) {conf:.0%}")

                # 输出路径
                if self.output_dir:
                    out_dir = Path(self.output_dir)
                else:
                    out_dir = src.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                stem, ext = src.stem, src.suffix
                out_path = out_dir / f"{stem}_blurred{ext}"
                cv2.imwrite(str(out_path), img)

                if plate_count > 0:
                    status = f"✅ 打码 {plate_count}"
                    detail = "; ".join(detail_parts)
                    self.log_msg.emit(f"  ✅ {src.name} → 检测到 {plate_count} 个车牌 → {out_path.name}")
                else:
                    status = "⚠️ 未检测到"
                    detail = ""
                    self.log_msg.emit(f"  ⚠️  {src.name} → 未检测到车牌，原样保存")

                self.item_done.emit(row, status, plate_count, detail)

            except Exception as e:
                self.item_done.emit(row, "❌ 失败", 0, str(e))
                self.log_msg.emit(f"  ❌ {src.name} → {e}")

        self.all_done.emit()


# ══════════════════════════════════════════════════════
# 拖放区域
# ══════════════════════════════════════════════════════

class DropZone(QFrame):
    """支持拖入图片 / 点击浏览的拖放区域。"""
    files_added = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setMinimumHeight(160)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #888;
                border-radius: 12px;
                background-color: #f8f9fa;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.icon = QLabel("📁")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setStyleSheet("font-size: 48px; border: none; background: transparent;")

        self.label = QLabel("拖入图片到此区域\n或点击选择文件")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 14px; color: #666; border: none; background: transparent;")

        layout.addWidget(self.icon)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QFrame {
                    border: 2px dashed #4a90d9;
                    border-radius: 12px;
                    background-color: #e8f0fe;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #888;
                border-radius: 12px;
                background-color: #f8f9fa;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #888;
                border-radius: 12px;
                background-color: #f8f9fa;
            }
        """)
        files = self._collect_images(event.mimeData())
        if files:
            self.files_added.emit(files)

    def mousePressEvent(self, event):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tiff);;所有文件 (*)"
        )
        if files:
            self.files_added.emit(list(files))

    def _collect_images(self, mime: QMimeData) -> list:
        images = []
        for url in mime.urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in IMAGE_EXTS:
                images.append(path)
        return images


# ══════════════════════════════════════════════════════
# 主窗口
# ══════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 车牌自动打码工具")
        self.setMinimumSize(900, 720)
        self.resize(1000, 780)

        self.items: list[tuple[str, str]] = []  # [(path, status), ...]
        self.worker: BlurWorker | None = None
        self.output_dir: str = ""
        self.last_input_dir: str = ""

        self._build_ui()
        self._load_model()

    # ── UI 构建 ────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # 标题
        title = QLabel("🚗 车牌自动打码工具")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        root.addWidget(title)

        # 上半部分：拖放区 + 设置
        top = QHBoxLayout()

        self.drop_zone = DropZone()
        self.drop_zone.files_added.connect(self.add_files)
        top.addWidget(self.drop_zone, stretch=3)

        # 设置面板
        settings = QGroupBox("⚙️ 处理设置")
        settings_layout = QVBoxLayout(settings)
        settings_layout.setSpacing(6)

        self._add_setting_row(settings_layout, "🔲 扩边像素", "margin", 0, 200, 10)
        self._add_setting_row(settings_layout, "🌫️ 模糊强度", "blur", 1, 99, 45)
        self._add_confidence_row(settings_layout)

        # 输出目录
        out_group = QHBoxLayout()
        out_group.addWidget(QLabel("📂 输出目录："))
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("留空 = 图片所在目录")
        self.out_dir_edit.setReadOnly(True)
        out_group.addWidget(self.out_dir_edit)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_output_dir)
        out_group.addWidget(btn_browse)
        btn_reset = QPushButton("默认")
        btn_reset.clicked.connect(lambda: self._set_output_dir(""))
        out_group.addWidget(btn_reset)
        settings_layout.addLayout(out_group)

        settings_layout.addStretch()
        top.addWidget(settings, stretch=2)

        root.addLayout(top)

        # 图片列表
        list_label = QLabel("📋 图片列表")
        list_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        root.addWidget(list_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["文件名", "状态", "检测数", "详情"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(self._open_image)
        root.addWidget(self.table)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        # 按钮栏
        btn_row = QHBoxLayout()

        self.btn_start = QPushButton("🚀 开始打码")
        self.btn_start.setMinimumHeight(36)
        self.btn_start.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.btn_start.clicked.connect(self.start_blur)
        btn_row.addWidget(self.btn_start)

        self.btn_open = QPushButton("📂 打开输出文件夹")
        self.btn_open.setMinimumHeight(36)
        self.btn_open.clicked.connect(self._open_output)
        btn_row.addWidget(self.btn_open)

        self.btn_clear = QPushButton("🗑️ 清空列表")
        self.btn_clear.setMinimumHeight(36)
        self.btn_clear.clicked.connect(self._clear_list)
        btn_row.addWidget(self.btn_clear)

        btn_row.addStretch()

        root.addLayout(btn_row)

        # 日志区域
        log_label = QLabel("📜 运行日志")
        log_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        root.addWidget(log_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(160)
        self.log_view.setFont(QFont("Consolas", 9))
        root.addWidget(self.log_view)

        # 状态栏
        self.statusBar().showMessage("✅ 就绪")

    def _add_setting_row(self, parent, label_text, attr_name, min_v, max_v, default):
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))

        spin = QSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(default)
        spin.setFixedWidth(70)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)

        row.addWidget(slider)
        row.addWidget(spin)
        parent.addLayout(row)

        setattr(self, f"spin_{attr_name}", spin)
        setattr(self, f"slider_{attr_name}", slider)

    def _add_confidence_row(self, parent):
        row = QHBoxLayout()
        row.addWidget(QLabel("🎯 置信度阈值"))

        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(1, 9)
        self.slider_conf.setValue(3)
        row.addWidget(self.slider_conf)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 0.9)
        self.spin_conf.setValue(0.3)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setFixedWidth(70)
        self.slider_conf.valueChanged.connect(lambda v: self.spin_conf.setValue(v / 10))
        self.spin_conf.valueChanged.connect(lambda v: self.slider_conf.setValue(int(v * 10)))
        row.addWidget(self.spin_conf)
        parent.addLayout(row)

    # ── 模型加载 ────────────────────────────────────

    def _load_model(self):
        if MODEL_PATH.exists():
            self._log("📦 使用已缓存的模型")
            return

        self._log("📥 正在下载模型（约 6MB）...")
        try:
            from huggingface_hub import hf_hub_download
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME,
                            local_dir=MODEL_PATH.parent, local_dir_use_symlinks=False)
            self._log("✅ 模型下载完成")
        except Exception as e:
            self._log(f"❌ 模型下载失败：{e}")
            QMessageBox.warning(self, "模型下载失败",
                                f"无法下载模型。请检查网络或手动下载。\n\n{e}")

    # ── 文件管理 ────────────────────────────────────

    def add_files(self, files: list):
        added = 0
        for f in files:
            f = str(Path(f).absolute())
            if f not in [p for p, _ in self.items]:
                self.items.append((f, "⏳ 等待中"))
                self.last_input_dir = str(Path(f).parent)
                added += 1
        self._refresh_table()
        self._log(f"📥 已添加 {added} 张图片（共 {len(self.items)} 张）")
        self.statusBar().showMessage(f"已添加 {added} 张图片，共 {len(self.items)} 张")

    def _clear_list(self):
        self.items.clear()
        self._refresh_table()
        self.progress_bar.setVisible(False)
        self._log("🗑️ 列表已清空")

    def _refresh_table(self):
        self.table.setRowCount(len(self.items))
        for row, (path, status) in enumerate(self.items):
            name = Path(path).name
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(status))
            self.table.setItem(row, 2, QTableWidgetItem(""))
            self.table.setItem(row, 3, QTableWidgetItem(""))

    # ── 输出目录 ─────────────────────────────────────

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.last_input_dir)
        if d:
            self._set_output_dir(d)

    def _set_output_dir(self, path):
        self.output_dir = path
        self.out_dir_edit.setText(path if path else "(图片所在目录)")
        self._log(f"📂 输出目录：{path or '与输入图片同目录'}")

    # ── 处理逻辑 ─────────────────────────────────────

    def start_blur(self):
        if not self.items:
            QMessageBox.information(self, "提示", "请先添加图片（拖入或点击上方区域）")
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "提示", "正在处理中，请等待完成")
            return

        self.btn_start.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.items))
        self._log("-" * 40)
        self._log(f"🚀 开始处理 {len(self.items)} 张图片")

        # 重置状态
        for i in range(len(self.items)):
            self.items[i] = (self.items[i][0], "🔄 处理中...")
        self._refresh_table()

        # 参数
        blur_k = self.spin_blur.value()
        margin = self.spin_margin.value()
        conf_th = self.spin_conf.value()
        task_items = [(i, p) for i, (p, _) in enumerate(self.items)]

        self.worker = BlurWorker(task_items, self.output_dir, blur_k, conf_th, margin)
        self.worker.progress.connect(self._on_progress)
        self.worker.item_done.connect(self._on_item_done)
        self.worker.log_msg.connect(self._log)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    def _on_progress(self, current, total):
        self.progress_bar.setValue(current)

    def _on_item_done(self, row, status, count, detail):
        self.items[row] = (self.items[row][0], status)
        self.table.setItem(row, 1, QTableWidgetItem(status))
        self.table.setItem(row, 2, QTableWidgetItem(str(count) if count else ""))
        self.table.setItem(row, 3, QTableWidgetItem(detail))

    def _on_all_done(self):
        self.btn_start.setEnabled(True)
        self.progress_bar.setVisible(False)
        success = sum(1 for _, s in self.items if s.startswith("✅"))
        total = len(self.items)
        self._log(f"🎉 处理完成！{success}/{total} 张图片检测到车牌")
        self.statusBar().showMessage(f"完成！{success}/{total} 检测到车牌")

    # ── 工具方法 ────────────────────────────────────

    def _open_output(self):
        target = self.output_dir or self.last_input_dir
        if target and Path(target).exists():
            subprocess.Popen(f'explorer "{target}"')
        else:
            QMessageBox.information(self, "提示", "请先处理图片或设置输出目录")

    def _open_image(self, index):
        path = self.items[index.row()][0]
        out_candidates = []
        stem = Path(path).stem
        ext = Path(path).suffix
        # 查找可能存在的输出文件
        out_dir = Path(self.output_dir) if self.output_dir else Path(path).parent
        for suffix in ["_blurred", ""]:
            cand = out_dir / f"{stem}{suffix}{ext}"
            if cand.exists() and cand != Path(path):
                out_candidates.append(str(cand))

        if out_candidates:
            subprocess.Popen(f'explorer /select,"{out_candidates[0]}"')
        else:
            subprocess.Popen(f'explorer /select,"{path}"')

    def _log(self, msg: str):
        self.log_view.append(msg)
        # 自动滚动到底部
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


# ══════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════

def main():
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局样式
    app.setStyleSheet("""
        QMainWindow { background-color: #ffffff; }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 16px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }
        QTableWidget {
            gridline-color: #eee;
        }
        QPushButton {
            padding: 6px 14px;
            border-radius: 4px;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #ddd;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 16px;
            height: 16px;
            margin: -6px 0;
            background: #4a90d9;
            border-radius: 8px;
        }
        QProgressBar {
            border: 1px solid #ddd;
            border-radius: 4px;
            height: 8px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4a90d9;
            border-radius: 3px;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
