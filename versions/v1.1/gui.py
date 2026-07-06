"""
车牌自动打码工具 - GUI 版 v1.1
新增：人脸打码 · CheckBox 开关 · 拍照导入 · 卡片按钮样式 · 响应式布局
"""

import logging
import os
import subprocess
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QFileDialog, QSplitter, QGroupBox, QLineEdit, QFrame,
    QAbstractItemView, QMessageBox, QProgressBar, QDialog, QDialogButtonBox,
)
from ultralytics import YOLO

# ── 模型 ────────────────────────────────────────────
HF_REPO_ID = "Koushim/yolov8-license-plate-detection"
HF_FILENAME = "best.pt"
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / HF_FILENAME

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── 人脸检测（YuNet ONNX，~340KB，自动下载） ────────
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = PROJECT_DIR / "models" / "face_detection_yunet_2023mar.onnx"

def _ensure_yunet():
    """确保 YuNet 模型已下载。"""
    if YUNET_PATH.exists():
        return
    import urllib.request
    YUNET_PATH.parent.mkdir(exist_ok=True)
    log.info("📥 下载 YuNet 人脸检测模型（~340KB）...")
    urllib.request.urlretrieve(YUNET_URL, str(YUNET_PATH))
    log.info("✅ YuNet 下载完成")

def detect_faces(img_rgb):
    """返回 [(x, y, w, h), ...]。首次调用自动下载 YuNet。"""
    _ensure_yunet()
    h, w = img_rgb.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "", (w, h),
        score_threshold=0.6, nms_threshold=0.3, top_k=500
    )
    _, faces = detector.detect(img_rgb)
    if faces is None:
        return []
    # faces: [n, 15] 每行 [x1,y1,w,h, landmarks...]
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]

# ── 日志 ────────────────────────────────────────────
log = logging.getLogger("plate_blur")
log.setLevel(logging.INFO)
log.propagate = False


# ══════════════════════════════════════════════════════
# 后台处理线程
# ══════════════════════════════════════════════════════

class BlurWorker(QThread):
    progress = Signal(int, int)
    item_done = Signal(int, str, int, str)   # row, status, plate_count+face_count, detail
    log_msg = Signal(str)
    all_done = Signal()

    def __init__(self, items, output_dir, blur_kernel,
                 confidence_threshold, margin,
                 blur_plates=True, blur_faces=True):
        super().__init__()
        self.items = items
        self.output_dir = output_dir
        self.blur_kernel = blur_kernel
        self.confidence_threshold = confidence_threshold
        self.margin = margin
        self.blur_plates = blur_plates
        self.blur_faces = blur_faces

    def _blur_region(self, img, x1, y1, x2, y2):
        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return
        k = self.blur_kernel
        if k % 2 == 0:
            k += 1
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    def run(self):
        yolo = YOLO(str(MODEL_PATH))
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

                plate_count = 0
                face_count = 0
                detail_parts = []
                h, w = img.shape[:2]

                # ── 车牌检测 ──
                if self.blur_plates:
                    results = yolo(img, verbose=False)
                    boxes = results[0].boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf[0])
                            if conf < self.confidence_threshold:
                                continue
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            self._blur_region(img,
                                              int(x1) - self.margin,
                                              int(y1) - self.margin,
                                              int(x2) + self.margin,
                                              int(y2) + self.margin)
                            plate_count += 1
                            detail_parts.append(f"🚗 ({int(x1)},{int(y1)},{int(x2)},{int(y2)}) {conf:.0%}")

                # ── 人脸检测 ──
                if self.blur_faces:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = detect_faces(rgb)
                    for (fx, fy, fw, fh) in faces:
                        self._blur_region(img,
                                          fx - self.margin,
                                          fy - self.margin,
                                          fx + fw + self.margin,
                                          fy + fh + self.margin)
                        face_count += 1
                        detail_parts.append(f"👤 ({fx},{fy},{fx+fw},{fy+fh})")

                # ── 输出 ──
                out_dir = Path(self.output_dir) if self.output_dir else src.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{src.stem}_blurred{src.suffix}"
                cv2.imwrite(str(out_path), img)

                total_detected = plate_count + face_count
                if total_detected > 0:
                    parts = []
                    if plate_count:
                        parts.append(f"{plate_count}车牌")
                    if face_count:
                        parts.append(f"{face_count}人脸")
                    status = f"✅ 打码 {'+'.join(parts)}"
                    detail = "; ".join(detail_parts)
                    self.log_msg.emit(f"  ✅ {src.name} → {status} → {out_path.name}")
                else:
                    status = "⚠️ 未检测到"
                    detail = ""
                    self.log_msg.emit(f"  ⚠️  {src.name} → 未检测到目标，原样保存")

                self.item_done.emit(row, status, total_detected, detail)

            except Exception as e:
                self.item_done.emit(row, "❌ 失败", 0, str(e))
                self.log_msg.emit(f"  ❌ {src.name} → {e}")

        self.all_done.emit()


# ══════════════════════════════════════════════════════
# 拍照对话框
# ══════════════════════════════════════════════════════

class CameraDialog(QDialog):
    """打开摄像头拍照，返回图片路径。"""
    image_captured = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📷 拍照导入")
        self.setMinimumSize(500, 420)
        self.cap = None
        self.captured_path = ""

        layout = QVBoxLayout(self)

        self.view_label = QLabel("正在打开摄像头...")
        self.view_label.setAlignment(Qt.AlignCenter)
        self.view_label.setMinimumHeight(300)
        self.view_label.setStyleSheet("background: #222; border-radius: 8px; color: #aaa;")
        layout.addWidget(self.view_label)

        btn_row = QHBoxLayout()
        self.btn_capture = QPushButton("📸 拍照")
        self.btn_capture.setMinimumHeight(40)
        self.btn_capture.setStyleSheet("""
            QPushButton { font-size:14px; font-weight:bold; background:#4a90d9; color:#fff;
                          border-radius:10px; padding:8px 24px; }
            QPushButton:hover { background:#357abd; }
            QPushButton:disabled { background:#ccc; }
        """)
        self.btn_capture.clicked.connect(self._capture)
        btn_row.addWidget(self.btn_capture)

        self.btn_retry = QPushButton("🔄 重试")
        self.btn_retry.setMinimumHeight(40)
        self.btn_retry.setStyleSheet("""
            QPushButton { font-size:13px; border:1px solid #ccc; border-radius:10px; padding:8px 16px; }
            QPushButton:hover { background:#eee; }
        """)
        self.btn_retry.clicked.connect(self._restart_camera)
        btn_row.addWidget(self.btn_retry)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self._start_camera()

    def _start_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.view_label.setText("❌ 无法打开摄像头")
            self.btn_capture.setEnabled(False)
            return
        self.timer.start(33)  # ~30 fps

    def _restart_camera(self):
        self.view_label.setText("正在打开摄像头...")
        self.btn_capture.setEnabled(True)
        self.captured_path = ""
        self._start_camera()

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # scale to fit label
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.view_label.width(), self.view_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view_label.setPixmap(pix)

    def _capture(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.cap = None

        out = PROJECT_DIR / "input" / "_camera_capture.jpg"
        out.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out), frame)
        self.captured_path = str(out)

        # show preview
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.view_label.width(), self.view_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view_label.setPixmap(pix)
        self.btn_capture.setText("✅ 已拍摄")
        self.btn_capture.setEnabled(False)

        QTimer.singleShot(800, lambda: (self.image_captured.emit(self.captured_path), self.accept()))

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════
# 拖放区域
# ══════════════════════════════════════════════════════

class DropZone(QFrame):
    files_added = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(160)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #888; border-radius: 12px; background-color: #f8f9fa;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.icon = QLabel("📁")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setStyleSheet("font-size: 40px; border: none; background: transparent;")

        self.label = QLabel("拖入图片到此区域\n或点击选择文件")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 13px; color: #666; border: none; background: transparent;")
        layout.addWidget(self.icon)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("QFrame { border:2px dashed #4a90d9; border-radius:12px; background:#e8f0fe; }")

    def dragLeaveEvent(self, event):
        self.setStyleSheet("QFrame { border:2px dashed #888; border-radius:12px; background:#f8f9fa; }")

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("QFrame { border:2px dashed #888; border-radius:12px; background:#f8f9fa; }")
        imgs = [u.toLocalFile() for u in event.mimeData().urls()
                if Path(u.toLocalFile()).suffix.lower() in IMAGE_EXTS]
        if imgs:
            self.files_added.emit(imgs)

    def mousePressEvent(self, event):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tiff);;所有文件 (*)"
        )
        if files:
            self.files_added.emit(list(files))


# ══════════════════════════════════════════════════════
# 主窗口
# ══════════════════════════════════════════════════════

CARD_BTN = """
    QPushButton {
        border: 1.5px solid #ddd; border-radius: 10px;
        padding: 8px 16px; background: #fff;
    }
    QPushButton:hover { border-color: #4a90d9; background: #e8f0fe; }
    QPushButton:pressed { background: #d0e1f9; }
    QPushButton:disabled { color: #aaa; border-color: #eee; background: #f5f5f5; }
"""

PRIMARY_BTN = """
    QPushButton {
        font-size: 14px; font-weight: bold;
        background: #4a90d9; color: #fff;
        border: none; border-radius: 10px;
        padding: 10px 20px;
    }
    QPushButton:hover { background: #357abd; }
    QPushButton:disabled { background: #aaa; }
"""

CHECKBOX_STYLE = """
    QCheckBox {
        font-size: 13px; spacing: 8px;
        padding: 6px 10px;
        border: 1px solid #ddd; border-radius: 8px;
        background: #fff;
    }
    QCheckBox:hover { border-color: #4a90d9; background: #f5f9ff; }
    QCheckBox::indicator {
        width: 20px; height: 20px;
    }
"""


class MainWindow(QMainWindow):
    BREAKPOINT = 760

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 车牌 & 人脸打码工具")
        self.setMinimumSize(400, 500)
        self.resize(1000, 780)

        self.items: list = []
        self.worker: BlurWorker | None = None
        self.output_dir = ""
        self.last_input_dir = ""

        self._build_ui()
        self._load_model()

    def resizeEvent(self, event):
        w = self.width()
        target = Qt.Vertical if w < self.BREAKPOINT else Qt.Horizontal
        if self.top_splitter.orientation() != target:
            self.top_splitter.setOrientation(target)
        super().resizeEvent(event)

    # ── UI ───────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # 标题
        title = QLabel("🚗 车牌 & 人脸自动打码工具")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        root.addWidget(title)

        # ── 上半部分：拖放区 + 设置（QSplitter） ──
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)

        # 拖放区
        drop_container = QWidget()
        drop_layout = QVBoxLayout(drop_container)
        drop_layout.setContentsMargins(0, 0, 0, 0)
        self.drop_zone = DropZone()
        self.drop_zone.files_added.connect(self.add_files)
        drop_layout.addWidget(self.drop_zone)

        # 拍照按钮（卡片样式）
        btn_camera = QPushButton("📷 拍照导入")
        btn_camera.setMinimumHeight(40)
        btn_camera.setStyleSheet(CARD_BTN)
        btn_camera.clicked.connect(self._open_camera)
        drop_layout.addWidget(btn_camera)

        self.top_splitter.addWidget(drop_container)

        # 设置面板
        settings = QGroupBox("⚙️ 处理设置")
        settings_layout = QVBoxLayout(settings)
        settings_layout.setSpacing(6)

        # CheckBox 开关
        chk_row = QHBoxLayout()
        self.chk_plates = QCheckBox("🚗 车牌打码")
        self.chk_plates.setChecked(True)
        self.chk_plates.setStyleSheet(CHECKBOX_STYLE)
        chk_row.addWidget(self.chk_plates)

        self.chk_faces = QCheckBox("👤 人脸打码")
        self.chk_faces.setChecked(True)
        self.chk_faces.setStyleSheet(CHECKBOX_STYLE)
        chk_row.addWidget(self.chk_faces)
        chk_row.addStretch()
        settings_layout.addLayout(chk_row)

        self._add_setting_row(settings_layout, "🔲 扩边像素", "margin", 0, 200, 10)
        self._add_setting_row(settings_layout, "🌫️ 模糊强度", "blur", 1, 99, 45)
        self._add_confidence_row(settings_layout)

        # 输出目录
        out_group = QHBoxLayout()
        out_group.addWidget(QLabel("📂 输出目录："))
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("留空 = 图片所在目录")
        self.out_dir_edit.setReadOnly(True)
        self.out_dir_edit.setStyleSheet("border-radius:6px; padding:4px 8px; border:1px solid #ddd;")
        out_group.addWidget(self.out_dir_edit)

        btn_browse = QPushButton("浏览...")
        btn_browse.setStyleSheet(CARD_BTN)
        btn_browse.clicked.connect(self._browse_output_dir)
        out_group.addWidget(btn_browse)
        btn_reset = QPushButton("默认")
        btn_reset.setStyleSheet(CARD_BTN)
        btn_reset.clicked.connect(lambda: self._set_output_dir(""))
        out_group.addWidget(btn_reset)
        settings_layout.addLayout(out_group)
        settings_layout.addStretch()

        self.top_splitter.addWidget(settings)
        self.top_splitter.setStretchFactor(0, 3)
        self.top_splitter.setStretchFactor(1, 2)
        root.addWidget(self.top_splitter)

        # ── 图片列表 ──
        list_label = QLabel("📋 图片列表")
        list_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        root.addWidget(list_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["文件名", "状态", "检测数", "详情"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(self._open_image)
        root.addWidget(self.table)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        # ── 按钮栏（卡片样式） ──
        btn_row = QHBoxLayout()

        self.btn_start = QPushButton("🚀 开始打码")
        self.btn_start.setMinimumHeight(42)
        self.btn_start.setStyleSheet(PRIMARY_BTN)
        self.btn_start.clicked.connect(self.start_blur)
        btn_row.addWidget(self.btn_start)

        self.btn_open = QPushButton("📂 打开输出文件夹")
        self.btn_open.setMinimumHeight(42)
        self.btn_open.setStyleSheet(CARD_BTN)
        self.btn_open.clicked.connect(self._open_output)
        btn_row.addWidget(self.btn_open)

        self.btn_clear = QPushButton("🗑️ 清空列表")
        self.btn_clear.setMinimumHeight(42)
        self.btn_clear.setStyleSheet(CARD_BTN)
        self.btn_clear.clicked.connect(self._clear_list)
        btn_row.addWidget(self.btn_clear)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # ── 日志 ──
        log_label = QLabel("📜 运行日志")
        log_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        root.addWidget(log_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(160)
        self.log_view.setFont(QFont("Consolas", 9))
        root.addWidget(self.log_view)

        self.statusBar().showMessage("✅ 就绪 — 默认同时打码车牌和人脸")

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

    # ── 模型 ─────────────────────────────────────────

    def _load_model(self):
        if MODEL_PATH.exists():
            self._log("📦 使用已缓存的模型")
            return
        self._log("📥 正在下载车牌检测模型（约 6MB）...")
        try:
            from huggingface_hub import hf_hub_download
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME,
                            local_dir=MODEL_PATH.parent, local_dir_use_symlinks=False)
            self._log("✅ 模型下载完成")
        except Exception as e:
            self._log(f"❌ 模型下载失败：{e}")
            QMessageBox.warning(self, "模型下载失败", f"无法下载模型。\n\n{e}")

    # ── 相机 ─────────────────────────────────────────

    def _open_camera(self):
        dlg = CameraDialog(self)
        dlg.image_captured.connect(self.add_files)
        dlg.exec()

    # ── 文件管理 ─────────────────────────────────────

    def add_files(self, files):
        if isinstance(files, str):
            files = [files]
        added = 0
        for f in files:
            f = str(Path(f).absolute())
            if f not in [p for p, _ in self.items]:
                self.items.append((f, "⏳ 等待中"))
                self.last_input_dir = str(Path(f).parent)
                added += 1
        self._refresh_table()
        self._log(f"📥 已添加 {added} 张图片（共 {len(self.items)} 张）")
        self.statusBar().showMessage(f"已添加 {added} 张，共 {len(self.items)} 张")

    def _clear_list(self):
        self.items.clear()
        self._refresh_table()
        self.progress_bar.setVisible(False)
        self._log("🗑️ 列表已清空")

    def _refresh_table(self):
        self.table.setRowCount(len(self.items))
        for row, (path, status) in enumerate(self.items):
            self.table.setItem(row, 0, QTableWidgetItem(Path(path).name))
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

    # ── 处理 ─────────────────────────────────────────

    def start_blur(self):
        if not self.items:
            QMessageBox.information(self, "提示", "请先添加图片")
            return
        if not self.chk_plates.isChecked() and not self.chk_faces.isChecked():
            QMessageBox.information(self, "提示", "请至少勾选一种打码类型（车牌/人脸）")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "提示", "正在处理中，请等待完成")
            return

        self.btn_start.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.items))
        self._log("-" * 40)
        self._log(f"🚀 开始处理 {len(self.items)} 张图片（车牌={'✓' if self.chk_plates.isChecked() else '✗'} 人脸={'✓' if self.chk_faces.isChecked() else '✗'}）")

        for i in range(len(self.items)):
            self.items[i] = (self.items[i][0], "🔄 处理中...")
        self._refresh_table()

        task_items = [(i, p) for i, (p, _) in enumerate(self.items)]
        self.worker = BlurWorker(
            task_items, self.output_dir,
            self.spin_blur.value(), self.spin_conf.value(), self.spin_margin.value(),
            blur_plates=self.chk_plates.isChecked(),
            blur_faces=self.chk_faces.isChecked(),
        )
        self.worker.progress.connect(lambda cur, tot: self.progress_bar.setValue(cur))
        self.worker.item_done.connect(self._on_item_done)
        self.worker.log_msg.connect(self._log)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    def _on_item_done(self, row, status, count, detail):
        self.items[row] = (self.items[row][0], status)
        self.table.setItem(row, 1, QTableWidgetItem(status))
        self.table.setItem(row, 2, QTableWidgetItem(str(count) if count else ""))
        self.table.setItem(row, 3, QTableWidgetItem(detail))

    def _on_all_done(self):
        self.btn_start.setEnabled(True)
        self.progress_bar.setVisible(False)
        success = sum(1 for _, s in self.items if s.startswith("✅"))
        self._log(f"🎉 完成！{success}/{len(self.items)} 张检测到目标")
        self.statusBar().showMessage(f"完成！{success}/{len(self.items)} 检测到目标")

    # ── 工具 ─────────────────────────────────────────

    def _open_output(self):
        target = self.output_dir or self.last_input_dir
        if target and Path(target).exists():
            subprocess.Popen(f'explorer "{target}"')
        else:
            QMessageBox.information(self, "提示", "请先处理图片或设置输出目录")

    def _open_image(self, index):
        path = self.items[index.row()][0]
        out_dir = Path(self.output_dir) if self.output_dir else Path(path).parent
        stem, ext = Path(path).stem, Path(path).suffix
        cand = out_dir / f"{stem}_blurred{ext}"
        subprocess.Popen(f'explorer /select,"{cand if cand.exists() else path}"')

    def _log(self, msg: str):
        self.log_view.append(msg)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())


# ══════════════════════════════════════════════════════

def main():
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow { background: #fff; }
        QGroupBox {
            font-weight: bold; border: 1px solid #ddd; border-radius: 8px;
            margin-top: 12px; padding-top: 16px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        QTableWidget { gridline-color: #eee; }
        QSlider::groove:horizontal { height: 6px; background: #ddd; border-radius: 3px; }
        QSlider::handle:horizontal {
            width: 18px; height: 18px; margin: -7px 0;
            background: #4a90d9; border-radius: 9px;
        }
        QProgressBar {
            border: 1px solid #ddd; border-radius: 4px; height: 8px; text-align: center;
        }
        QProgressBar::chunk { background: #4a90d9; border-radius: 3px; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
