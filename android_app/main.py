"""
车牌 & 人脸自动打码工具 — Android Kivy 版
触屏优化 · 响应式布局 · 拍照导入 · 自选打码类型
"""
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import (
    BooleanProperty, NumericProperty, StringProperty,
    ListProperty, ObjectProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.utils import platform

# ── 项目路径 ─────────────────────────────────────────
# Android APK 内文件在 app 目录；桌面端在父目录
_APP_DIR = Path(__file__).parent
_MODEL_CANDIDATES = [
    _APP_DIR / "models",              # APK 内（android_app/models/）
    _APP_DIR.parent / "models",       # 桌面端（license-plate-blur/models/）
]
MODELS_DIR = next((d for d in _MODEL_CANDIDATES if d.exists()), _MODEL_CANDIDATES[0])
MODELS_DIR.mkdir(exist_ok=True)
YOLO_MODEL = MODELS_DIR / "best.pt"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
PROJECT_DIR = MODELS_DIR.parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── 人脸检测（YuNet ONNX）────────────────────────────
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

def _ensure_yunet():
    if YUNET_MODEL.exists():
        return
    import urllib.request
    MODELS_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(YUNET_URL, str(YUNET_MODEL))

def detect_faces(img_rgb):
    _ensure_yunet()
    h, w = img_rgb.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (w, h),
        score_threshold=0.6, nms_threshold=0.3, top_k=500
    )
    _, faces = detector.detect(img_rgb)
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


# ══════════════════════════════════════════════════════
# KV 样式定义
# ══════════════════════════════════════════════════════

KV = """
#:import platform kivy.utils.platform

<MainScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: dp(12)
        spacing: dp(8)

        # ── 标题 ──
        Label:
            text: '🚗 车牌 & 人脸打码工具'
            font_size: dp(20)
            bold: True
            size_hint_y: None
            height: dp(44)
            color: 0.2, 0.2, 0.2, 1

        # ── 滚动区域 ──
        ScrollView:
            do_scroll_x: False
            BoxLayout:
                orientation: 'vertical'
                spacing: dp(10)
                size_hint_y: None
                height: self.minimum_height

                # ── 文件选择按钮 ──
                CardBox:
                    orientation: 'vertical'
                    spacing: dp(8)
                    Label:
                        text: '📁 选择图片'
                        font_size: dp(14)
                        bold: True
                        size_hint_y: None
                        height: dp(28)
                        halign: 'left'
                        text_size: self.width, None
                    BoxLayout:
                        spacing: dp(8)
                        size_hint_y: None
                        height: dp(52)
                        ActionButton:
                            text: '📂 浏览文件'
                            on_release: root.open_file_chooser()
                        ActionButton:
                            text: '📷 拍照导入'
                            on_release: root.open_camera()
                    Label:
                        id: file_count
                        text: '尚未选择图片'
                        font_size: dp(12)
                        color: 0.5, 0.5, 0.5, 1
                        size_hint_y: None
                        height: dp(22)

                # ── 打码开关 ──
                CardBox:
                    orientation: 'horizontal'
                    spacing: dp(16)
                    size_hint_y: None
                    height: dp(56)
                    ToggleRow:
                        id: chk_plates
                        text: '🚗 车牌'
                        active: True
                    ToggleRow:
                        id: chk_faces
                        text: '👤 人脸'
                        active: True

                # ── 设置滑块 ──
                CardBox:
                    orientation: 'vertical'
                    spacing: dp(6)
                    Label:
                        text: '⚙️ 处理设置'
                        font_size: dp(14)
                        bold: True
                        size_hint_y: None
                        height: dp(28)
                    SettingSlider:
                        id: slider_margin
                        label: '🔲 扩边像素'
                        min_val: 0; max_val: 200; default: 10; step: 1
                    SettingSlider:
                        id: slider_blur
                        label: '🌫️ 模糊强度'
                        min_val: 1; max_val: 99; default: 45; step: 2
                    SettingSlider:
                        id: slider_conf
                        label: '🎯 置信度阈值'
                        min_val: 0.1; max_val: 0.9; default: 0.3; step: 0.05

                # ── 输出目录 ──
                CardBox:
                    orientation: 'horizontal'
                    spacing: dp(8)
                    size_hint_y: None
                    height: dp(48)
                    Label:
                        text: '📂 输出'
                        font_size: dp(13)
                        size_hint_x: None
                        width: dp(60)
                    Label:
                        id: out_dir_label
                        text: '(图片所在目录)'
                        font_size: dp(12)
                        color: 0.5, 0.5, 0.5, 1
                    ActionButton:
                        text: '浏览...'
                        size_hint_x: None
                        width: dp(80)
                        on_release: root.choose_output_dir()
                    ActionButton:
                        text: '默认'
                        size_hint_x: None
                        width: dp(60)
                        on_release: root.reset_output_dir()

        # ── 图片列表 ──
        Label:
            text: '📋 图片列表'
            font_size: dp(14)
            bold: True
            size_hint_y: None
            height: dp(28)
            halign: 'left'
            text_size: self.width, None

        ScrollView:
            do_scroll_x: False
            size_hint_y: 0.3
            GridLayout:
                id: image_list
                cols: 1
                spacing: dp(2)
                size_hint_y: None
                height: self.minimum_height

        # ── 进度条 ──
        ProgressBar:
            id: progress_bar
            max: 100
            value: 0
            size_hint_y: None
            height: dp(8)
            opacity: 0

        # ── 按钮栏 ──
        BoxLayout:
            spacing: dp(8)
            size_hint_y: None
            height: dp(54)
            PrimaryButton:
                id: btn_start
                text: '🚀 开始打码'
                on_release: root.start_blur()
            ActionButton:
                text: '📂 打开输出'
                on_release: root.open_output()
            ActionButton:
                text: '🗑️ 清空'
                on_release: root.clear_list()

        # ── 日志 ──
        Label:
            text: '📜 运行日志'
            font_size: dp(13)
            bold: True
            size_hint_y: None
            height: dp(26)
        TextInput:
            id: log_view
            readonly: True
            font_size: dp(11)
            size_hint_y: 0.2
            background_color: 0.1, 0.1, 0.1, 1
            foreground_color: 0, 1, 0, 1
            text: '✅ 就绪 — 默认同时打码车牌和人脸\\n'

# ── 自定义组件 ──

<CardBox@BoxLayout>:
    padding: dp(12)
    canvas.before:
        Color:
            rgba: 0.97, 0.97, 0.97, 1
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(10)]

<ActionButton@Button>:
    font_size: dp(14)
    background_normal: ''
    background_color: 1, 1, 1, 1
    color: 0.2, 0.2, 0.2, 1
    size_hint_y: None
    height: dp(48)
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(10)]
        Color:
            rgba: 0.85, 0.85, 0.85, 1
        Line:
            rounded_rectangle: self.x+1, self.y+1, self.width-2, self.height-2, dp(10)
            width: 1.5
    on_press: self.background_color = (0.9, 0.95, 1, 1)
    on_release: self.background_color = (1, 1, 1, 1)

<PrimaryButton@Button>:
    font_size: dp(15)
    bold: True
    background_normal: ''
    background_color: 0.29, 0.56, 0.85, 1
    color: 1, 1, 1, 1
    size_hint_y: None
    height: dp(50)
    canvas.before:
        Color:
            rgba: 0.29, 0.56, 0.85, 1
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(10)]
    on_press: self.background_color = (0.21, 0.48, 0.74, 1)
    on_release: self.background_color = (0.29, 0.56, 0.85, 1)

<ToggleRow@BoxLayout>:
    text: ''
    active: False
    spacing: dp(6)
    size_hint_y: None
    height: dp(44)
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(8)]
        Color:
            rgba: 0.85, 0.85, 0.85, 1
        Line:
            rounded_rectangle: self.x+1, self.y+1, self.width-2, self.height-2, dp(8)
            width: 1.5
    CheckBox:
        id: cb
        active: root.active
        size_hint_x: None
        width: dp(36)
        on_active: root.active = self.active
    Label:
        text: root.text
        font_size: dp(14)
        size_hint_x: None
        width: dp(60)
        halign: 'left'
        text_size: self.width, None

<SettingSlider@BoxLayout>:
    label: ''
    min_val: 0
    max_val: 100
    default: 50
    step: 1
    spacing: dp(8)
    size_hint_y: None
    height: dp(44)
    Label:
        text: root.label
        font_size: dp(12)
        size_hint_x: None
        width: dp(90)
        halign: 'left'
        text_size: self.width, None
    Slider:
        id: sl
        min: root.min_val
        max: root.max_val
        value: root.default
        step: root.step
    Label:
        id: val_label
        text: str(root.default)
        font_size: dp(12)
        size_hint_x: None
        width: dp(44)
        halign: 'right'
        text_size: self.width, None
"""

Builder.load_string(KV)


# ══════════════════════════════════════════════════════
# 自定义组件类
# ══════════════════════════════════════════════════════

class CardBox(BoxLayout):
    pass

class ActionButton(Button):
    pass

class PrimaryButton(Button):
    pass

class ToggleRow(BoxLayout):
    text = StringProperty('')
    active = BooleanProperty(False)

class SettingSlider(BoxLayout):
    label = StringProperty('')
    min_val = NumericProperty(0)
    max_val = NumericProperty(100)
    default = NumericProperty(50)
    step = NumericProperty(1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_once(self._bind, 0)

    def _bind(self, dt):
        sl = self.ids.sl
        vl = self.ids.val_label
        sl.bind(value=lambda inst, v: setattr(vl, 'text', str(round(v, 2))))


# ══════════════════════════════════════════════════════
# 后台处理线程
# ══════════════════════════════════════════════════════

class BlurThread(threading.Thread):
    def __init__(self, files, output_dir, margin, blur_kernel, confidence,
                 blur_plates, blur_faces, callback):
        super().__init__(daemon=True)
        self.files = files
        self.output_dir = output_dir
        self.margin = margin
        self.blur_kernel = blur_kernel
        self.confidence = confidence
        self.blur_plates = blur_plates
        self.blur_faces = blur_faces
        self.callback = callback  # fn(status_dict)
        self._stop = False

    def stop(self):
        self._stop = True

    def _blur_region(self, img, x1, y1, x2, y2):
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return
        k = self.blur_kernel
        if k % 2 == 0:
            k += 1
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    def run(self):
        from ultralytics import YOLO
        yolo = YOLO(str(YOLO_MODEL))
        total = len(self.files)

        for idx, src in enumerate(self.files):
            if self._stop:
                break
            src = Path(src)
            result = {"idx": idx, "filename": src.name, "status": "processing"}
            self.callback(result)

            try:
                img = cv2.imread(str(src))
                if img is None:
                    result["status"] = "read_error"
                    self.callback(result)
                    continue

                plate_count = 0
                face_count = 0
                details = []

                # 车牌检测
                if self.blur_plates:
                    results = yolo(img, verbose=False)
                    boxes = results[0].boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf[0])
                            if conf < self.confidence:
                                continue
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            self._blur_region(
                                img,
                                int(x1) - self.margin,
                                int(y1) - self.margin,
                                int(x2) + self.margin,
                                int(y2) + self.margin,
                            )
                            plate_count += 1
                            details.append(f"🚗 {conf:.0%}")

                # 人脸检测
                if self.blur_faces:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = detect_faces(rgb)
                    for (fx, fy, fw, fh) in faces:
                        self._blur_region(
                            img,
                            fx - self.margin,
                            fy - self.margin,
                            fx + fw + self.margin,
                            fy + fh + self.margin,
                        )
                        face_count += 1
                        details.append("👤")

                # 输出
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
                    result["status"] = "ok"
                    result["detail"] = f"{'+'.join(parts)} — {'; '.join(details)}"
                    result["out_path"] = str(out_path)
                else:
                    result["status"] = "none"
                    result["detail"] = ""

                self.callback(result)

            except Exception as e:
                result["status"] = "error"
                result["detail"] = str(e)
                self.callback(result)

        result = {"idx": -1, "filename": "", "status": "done"}
        self.callback(result)


# ══════════════════════════════════════════════════════
# 主界面
# ══════════════════════════════════════════════════════

class MainScreen(Screen):
    files = ListProperty([])  # [(path, status, detail), ...]
    output_dir = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thread = None
        self._pending_results = []
        Clock.schedule_interval(self._drain_results, 0.1)

    # ── 文件选择 ─────────────────────────────────────

    def open_file_chooser(self):
        """弹出文件选择器"""
        if platform == 'android':
            from plyer import filechooser
            filechooser.open_file(
                on_selection=self._on_files_selected,
                filters=[("图片", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")],
                multiple=True,
            )
        else:
            # 桌面端用 Kivy 内置 FileChooser
            content = BoxLayout(orientation='vertical', spacing=8, padding=8)
            fc = FileChooserIconView(
                filters=['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.tiff'],
                multiselect=True,
            )
            content.add_widget(fc)
            btn_box = BoxLayout(size_hint_y=None, height=48, spacing=8)
            popup = Popup(title='选择图片', content=content, size_hint=(0.9, 0.8))

            def select(instance):
                popup.dismiss()
                self._on_files_selected(fc.selection)

            btn_cancel = Button(text='取消', size_hint_y=None, height=44)
            btn_cancel.bind(on_release=popup.dismiss)
            btn_ok = Button(text='确定', size_hint_y=None, height=44,
                            background_color=(0.29, 0.56, 0.85, 1))
            btn_ok.bind(on_release=select)
            btn_box.add_widget(btn_cancel)
            btn_box.add_widget(btn_ok)
            content.add_widget(btn_box)
            popup.open()

    def _on_files_selected(self, selection):
        if not selection:
            return
        added = 0
        for f in selection:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                if f not in [p for p, _, _ in self.files]:
                    self.files.append((f, '⏳ 等待中', ''))
                    added += 1
        self._refresh_ui()
        self.log(f'📥 已添加 {added} 张图片（共 {len(self.files)} 张）')

    # ── 拍照 ─────────────────────────────────────────

    def open_camera(self):
        if platform == 'android':
            from plyer import camera
            try:
                camera.take_picture(
                    filename=str(PROJECT_DIR / 'input' / '_camera_capture.jpg'),
                    on_complete=self._on_camera_done,
                )
            except Exception as e:
                self.log(f'❌ 拍照失败：{e}')
        else:
            # 桌面端用 OpenCV 摄像头
            self.log('📷 正在打开摄像头...')
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.log('❌ 无法打开摄像头')
                return
            # 简单实现：拍一张就关
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.log('❌ 拍照失败')
                return
            out = PROJECT_DIR / 'input' / '_camera_capture.jpg'
            out.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(out), frame)
            self._on_camera_done(str(out))

    def _on_camera_done(self, path):
        if path and Path(path).exists():
            self._on_files_selected([path])
            self.log('📸 拍照完成')

    # ── 输出目录 ─────────────────────────────────────

    def choose_output_dir(self):
        if platform == 'android':
            from plyer import filechooser
            filechooser.choose_dir(on_selection=self._on_output_selected)
        else:
            content = BoxLayout(orientation='vertical', spacing=8, padding=8)
            fc = FileChooserIconView(dirselect=True)
            content.add_widget(fc)
            popup = Popup(title='选择输出目录', content=content, size_hint=(0.9, 0.8))

            def select(instance):
                popup.dismiss()
                if fc.selection:
                    self._on_output_selected(fc.selection)

            btn_box = BoxLayout(size_hint_y=None, height=48, spacing=8)
            btn_cancel = Button(text='取消', size_hint_y=None, height=44)
            btn_cancel.bind(on_release=popup.dismiss)
            btn_ok = Button(text='选择', size_hint_y=None, height=44,
                            background_color=(0.29, 0.56, 0.85, 1))
            btn_ok.bind(on_release=select)
            btn_box.add_widget(btn_cancel)
            btn_box.add_widget(btn_ok)
            content.add_widget(btn_box)
            popup.open()

    def _on_output_selected(self, selection):
        if selection:
            d = selection[0] if isinstance(selection, list) else selection
            self.output_dir = str(Path(d).absolute())
            self.ids.out_dir_label.text = self.output_dir

    def reset_output_dir(self):
        self.output_dir = ''
        self.ids.out_dir_label.text = '(图片所在目录)'

    # ── 处理 ─────────────────────────────────────────

    def start_blur(self):
        if not self.files:
            self.log('⚠️ 请先选择图片')
            return
        if not self._chk_plates_active() and not self._chk_faces_active():
            self.log('⚠️ 请至少勾选一种打码类型')
            return

        self.ids.btn_start.disabled = True
        self.ids.progress_bar.opacity = 1
        self.ids.progress_bar.value = 0
        self.ids.progress_bar.max = len(self.files)

        for i in range(len(self.files)):
            self.files[i] = (self.files[i][0], '🔄 处理中...', '')
        self._refresh_image_list()

        self.log('—' * 30)
        plates = '✓' if self._chk_plates_active() else '✗'
        faces = '✓' if self._chk_faces_active() else '✗'
        self.log(f'🚀 开始处理 {len(self.files)} 张（车牌={plates} 人脸={faces}）')

        self.thread = BlurThread(
            files=[f for f, _, __ in self.files],
            output_dir=self.output_dir,
            margin=self._get_slider('margin'),
            blur_kernel=self._get_slider('blur'),
            confidence=self._get_slider('conf'),
            blur_plates=self._chk_plates_active(),
            blur_faces=self._chk_faces_active(),
            callback=self._on_blur_result,
        )
        self.thread.start()

    def _on_blur_result(self, result):
        """从后台线程回调，加入队列由主线程处理"""
        self._pending_results.append(result)

    def _drain_results(self, dt):
        while self._pending_results:
            r = self._pending_results.pop(0)
            self._handle_result(r)

    def _handle_result(self, r):
        idx = r["idx"]
        if r["status"] == "done":
            self.ids.btn_start.disabled = False
            self.ids.progress_bar.opacity = 0
            ok_count = sum(1 for _, s, _ in self.files if s.startswith('✅'))
            self.log(f'🎉 完成！{ok_count}/{len(self.files)} 张检测到目标')
            return

        if r["status"] == "ok":
            count = r.get("detail", "").split("—")[0].strip() if "—" in r.get("detail", "") else ""
            self.files[idx] = (self.files[idx][0], f'✅ {count}', r.get("detail", ""))
            self.log(f'  ✅ {r["filename"]} → {count} → {Path(r.get("out_path","")).name}')
        elif r["status"] == "none":
            self.files[idx] = (self.files[idx][0], '⚠️ 未检测到', '')
            self.log(f'  ⚠️  {r["filename"]} → 未检测到目标')
        elif r["status"] == "read_error":
            self.files[idx] = (self.files[idx][0], '❌ 读取失败', '')
            self.log(f'  ❌ {r["filename"]} → 读取失败')
        elif r["status"] == "error":
            self.files[idx] = (self.files[idx][0], '❌ 失败', r.get("detail", ""))
            self.log(f'  ❌ {r["filename"]} → {r.get("detail", "")}')
        else:
            self.files[idx] = (self.files[idx][0], '🔄 处理中...', '')

        self.ids.progress_bar.value = sum(
            1 for _, s, __ in self.files if s not in ('⏳ 等待中', '🔄 处理中...')
        )
        self._refresh_image_list()

    # ── 打开输出 ─────────────────────────────────────

    def open_output(self):
        target = self.output_dir or (
            Path(self.files[0][0]).parent if self.files else None
        )
        if target and Path(target).exists():
            if platform == 'android':
                from plyer import filechooser
                import android
                # Fallback: just log the path
                self.log(f'📂 输出目录: {target}')
            else:
                import subprocess
                subprocess.Popen(f'explorer "{target}"', shell=True)
        else:
            self.log('⚠️ 请先处理图片或设置输出目录')

    # ── 清空 ─────────────────────────────────────────

    def clear_list(self):
        if self.thread and self.thread.is_alive():
            self.thread.stop()
        self.files.clear()
        self.ids.progress_bar.opacity = 0
        self._refresh_image_list()
        self._refresh_file_count()
        self.log('🗑️ 列表已清空')

    # ── 刷新 UI ─────────────────────────────────────

    def _refresh_ui(self):
        self._refresh_file_count()
        self._refresh_image_list()

    def _refresh_file_count(self):
        n = len(self.files)
        self.ids.file_count.text = f'已选择 {n} 张图片' if n else '尚未选择图片'

    def _refresh_image_list(self):
        grid = self.ids.image_list
        grid.clear_widgets()
        for path, status, detail in self.files:
            row = BoxLayout(
                size_hint_y=None, height=36, spacing=4,
                padding=[4, 0, 4, 0],
            )
            row.add_widget(Label(
                text=Path(path).name,
                font_size=12,
                size_hint_x=0.4,
                halign='left',
                text_size=(None, None),
                shorten=True,
            ))
            row.add_widget(Label(
                text=status,
                font_size=11,
                size_hint_x=0.3,
            ))
            row.add_widget(Label(
                text=detail[:40] if detail else '',
                font_size=10,
                color=(0.5, 0.5, 0.5, 1),
                size_hint_x=0.3,
                halign='left',
                text_size=(None, None),
                shorten=True,
            ))
            grid.add_widget(row)

    # ── 日志 ─────────────────────────────────────────

    def log(self, msg):
        self.ids.log_view.text += msg + '\n'

    # ── 取值辅助 ─────────────────────────────────────

    def _chk_plates_active(self):
        return self.ids.chk_plates.ids.cb.active

    def _chk_faces_active(self):
        return self.ids.chk_faces.ids.cb.active

    def _get_slider(self, name):
        sl = self.ids[f'slider_{name}'].ids.sl
        val = sl.value
        if name == 'conf':
            return float(val)
        return int(val)


# ══════════════════════════════════════════════════════
# App 入口
# ══════════════════════════════════════════════════════

class BlurApp(App):
    title = '车牌打码工具'

    def build(self):
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE,
            ])

        # 确保模型存在
        self._check_models()

        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        return sm

    def _check_models(self):
        MODELS_DIR.mkdir(exist_ok=True)
        if not YOLO_MODEL.exists():
            try:
                from huggingface_hub import hf_hub_download
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                hf_hub_download(
                    repo_id='Koushim/yolov8-license-plate-detection',
                    filename='best.pt',
                    local_dir=str(MODELS_DIR),
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                print(f'模型下载失败: {e}')


if __name__ == '__main__':
    BlurApp().run()
