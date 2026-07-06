[app]

# 应用基本信息
title = 车牌打码工具
package.name = plateblur
package.domain = com.plateblur.app
source.dir = .
main.py = main.py
version = 1.0.0

# 应用要求
requirements = python3,kivy,numpy,opencv,pillow,ultralytics,huggingface_hub,plyer,android
# opencv 通过 p4a 配方编译（Android arm64 支持）
# 启用摄像头
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,INTERNET

# Android 配置
android.api = 31
android.minapi = 26
android.ndk = 25b
android.gradle_dependencies = 'androidx.core:core:1.9.0'
android.arch = arm64-v8a
android.allow_backup = True
android.add_src = models/

# 图标 — 指向 android_icons 目录
icon.filename = ../android_icons/ic_launcher-playstore.png

# 如果使用自适应图标
android.adaptive_icon.foreground = ../android_icons/mipmap-xhdpi/ic_launcher_foreground.png
android.adaptive_icon.background = ../android_icons/mipmap-xhdpi/ic_launcher_background.png

# 启动画面（可选）
# Presplash 颜色：黑色背景
android.presplash_color = #000000

# 横竖屏
orientation = portrait

# 全屏
fullscreen = 0

# 日志级别
log_level = 2

# 自动接受 SDK 许可
android.accept_sdk_license = True

# 使用本地 numpy 配方（v1.26.4，兼容 Android NDK）
p4a.local_recipes = p4a_recipes/

# 启用 AndroidX
android.enable_androidx = True

# 使用 Java 8
android.add_java8 = True

# 忽略检查（避免因文件缺失中断）
warn_on_root = 0

# 复制模型文件到 APK
source.include_exts = py,png,jpg,kv,atlas,ttf,pt,onnx
source.include_patterns = models/**

[buildozer]
log_level = 2
