"""
车牌自动打码工具 🚗🔒
自动检测图片中的车牌，并打上高斯模糊（马赛克效果）。

用法：
    python blur.py                    # 处理 input/ 下所有图片
    python blur.py -i car.jpg         # 处理单张图片
    python blur.py -i input_dir -o output_dir  # 批量处理

首次运行会自动下载模型（约 6MB），需要网络。
如果 HuggingFace 访问不了，设置环境变量后重试：
    set HF_ENDPOINT=https://hf-mirror.com    (Windows CMD)
    或 $env:HF_ENDPOINT="https://hf-mirror.com"  (PowerShell)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ── 模型 ────────────────────────────────────────────
# YOLOv8n 车牌检测模型，约 6MB，HuggingFace 托管
# 国内自动使用 hf-mirror.com 镜像
HF_REPO_ID = "Koushim/yolov8-license-plate-detection"
HF_FILENAME = "best.pt"
MODEL_CACHE_DIR = Path(__file__).parent / "models"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# ── 支持的图片格式 ──────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def download_model():
    """下载模型到本地 models/ 目录，自动使用国内镜像。"""
    local_path = MODEL_CACHE_DIR / HF_FILENAME

    if local_path.exists():
        print(f"📦 使用已缓存的模型：{local_path}")
        return local_path

    # 优先使用国内镜像（hf-mirror.com），失败则回退到官方
    for endpoint in ["https://hf-mirror.com", "https://huggingface.co"]:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"📥 正在下载模型（{endpoint}）...")
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir=MODEL_CACHE_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"✅ 模型已下载到：{downloaded}")
            return Path(downloaded)
        except Exception as e:
            print(f"   ⚠️  {endpoint} 下载失败：{e}")
            continue

    print("\n❌ 所有下载方式均失败。请手动下载模型：")
    print(f"   https://hf-mirror.com/{HF_REPO_ID}/resolve/main/{HF_FILENAME}")
    print(f"   放入 {MODEL_CACHE_DIR.absolute()}/ 目录后重新运行")
    sys.exit(1)


def load_model():
    """加载 YOLO 车牌检测模型。"""
    print(f"📦 模型仓库：{HF_REPO_ID}")
    model_path = download_model()
    model = YOLO(str(model_path))
    print("✅ 模型加载成功\n")
    return model


def blur_license_plates(model, image_path, output_path, blur_kernel=45):
    """
    检测图片中的车牌并打码。

    Args:
        model: YOLO 模型
        image_path: 输入图片路径
        output_path: 输出图片路径
        blur_kernel: 模糊核大小（必须为奇数，越大越模糊，默认 45）
    """
    # 确保模糊核是奇数
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠️  无法读取图片：{image_path}")
        return False

    h, w = img.shape[:2]
    print(f"  🖼️  图片尺寸：{w}x{h}")

    # YOLO 检测
    results = model(img, verbose=False)

    plate_count = 0
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # 获取坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 边界检查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # 提取车牌区域
            plate_roi = img[y1:y2, x1:x2]
            if plate_roi.size == 0:
                continue

            # 高斯模糊
            plate_roi = cv2.GaussianBlur(plate_roi, (blur_kernel, blur_kernel), 0)
            img[y1:y2, x1:x2] = plate_roi

            plate_count += 1
            confidence = float(box.conf[0])
            print(f"  🎯 检测到车牌：置信度 {confidence:.1%}，位置 [{x1},{y1},{x2},{y2}]")

    if plate_count == 0:
        print("  ℹ️  未检测到车牌，原样保存")
    else:
        print(f"  ✅ 已对 {plate_count} 个车牌打码")

    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"  💾 已保存：{output_path}\n")
    return True


def process_images(model, input_path, output_dir, blur_kernel=45):
    """
    处理单个图片或整个目录。

    Args:
        model: YOLO 模型
        input_path: 输入图片路径或目录路径
        output_dir: 输出目录
        blur_kernel: 模糊强度
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # 收集要处理的图片
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTS:
            images = [input_path]
            print(f"📷 处理单张图片：{input_path.name}")
        else:
            print(f"❌ 不支持的文件格式：{input_path.suffix}")
            return
    elif input_path.is_dir():
        images = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        ])
        if not images:
            print(f"❌ 目录 {input_path} 中没有找到图片文件")
            return
        print(f"📁 在 {input_path} 中找到 {len(images)} 张图片\n")
    else:
        print(f"❌ 路径不存在：{input_path}")
        return

    # 逐张处理
    total = len(images)
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{total}]", end=" ")
        out_path = output_dir / img_path.name
        blur_license_plates(model, img_path, out_path, blur_kernel)

    print(f"🎉 完成！共处理 {total} 张图片，结果保存在 {output_dir.absolute()}/")


def main():
    parser = argparse.ArgumentParser(
        description="车牌自动打码工具 - 自动检测并模糊图片中的车牌",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python blur.py                          # 处理 input/ 下所有图片 → output/
  python blur.py -i car.jpg               # 处理单张图片
  python blur.py -i photos/ -o blurred/   # 批量处理
  python blur.py -i car.jpg -b 65         # 更重的模糊效果
        """,
    )
    parser.add_argument(
        "-i", "--input",
        default="input",
        help="输入图片路径或目录（默认：input/）",
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="输出目录（默认：output/）",
    )
    parser.add_argument(
        "-b", "--blur",
        type=int,
        default=45,
        help="模糊强度，奇数，越大越模糊（默认：45）",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("  🚗 车牌自动打码工具 v1.0")
    print("  模型：YOLOv8n (~6MB)")
    print("=" * 50 + "\n")

    model = load_model()
    process_images(model, args.input, args.output, args.blur)


if __name__ == "__main__":
    main()
