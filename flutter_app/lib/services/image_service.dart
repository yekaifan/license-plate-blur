import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// 图像预处理 + 高斯模糊服务
class ImageService {
  /// 将图像预处理为 YOLO 输入格式 [1, 3, 640, 640] NCHW
  static Float32List preprocessYolo(img.Image image) {
    // Resize to 640x640
    final resized = img.copyResize(image, width: 640, height: 640);
    
    final input = Float32List(3 * 640 * 640);
    int idx = 0;
    
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < 640; y++) {
        for (int x = 0; x < 640; x++) {
          final pixel = resized.getPixel(x, y);
          double val;
          if (c == 0) {
            val = pixel.r / 255.0; // R
          } else if (c == 1) {
            val = pixel.g / 255.0; // G
          } else {
            val = pixel.b / 255.0; // B
          }
          input[idx++] = val;
        }
      }
    }
    return input;
  }

  /// 将图像预处理为 YuNet 输入格式 [1, 3, H, W] NCHW
  static Float32List preprocessFace(img.Image image) {
    final w = image.width;
    final h = image.height;
    
    final input = Float32List(3 * h * w);
    int idx = 0;
    
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final pixel = image.getPixel(x, y);
          double val;
          if (c == 0) {
            val = pixel.r / 255.0;
          } else if (c == 1) {
            val = pixel.g / 255.0;
          } else {
            val = pixel.b / 255.0;
          }
          input[idx++] = val;
        }
      }
    }
    return input;
  }

  /// 对图像指定区域应用高斯模糊
  static img.Image blurRegion(
    img.Image image,
    int x, int y, int w, int h,
    int kernelSize, int margin,
  ) {
    final imgW = image.width;
    final imgH = image.height;
    
    // 扩展边距
    int x1 = (x - margin).clamp(0, imgW);
    int y1 = (y - margin).clamp(0, imgH);
    int x2 = (x + w + margin).clamp(0, imgW);
    int y2 = (y + h + margin).clamp(0, imgH);
    
    if (x2 <= x1 || y2 <= y1) return image;
    
    // 确保核大小为奇数
    int k = kernelSize;
    if (k % 2 == 0) k++;
    
    // 直接在原图上模糊指定区域
    final result = img.Image(width: imgW, height: imgH);
    // 先复制原图
    for (int py = 0; py < imgH; py++) {
      for (int px = 0; px < imgW; px++) {
        result.setPixel(px, py, image.getPixel(px, py));
      }
    }
    
    // 对 ROI 区域高斯模糊
    final roi = img.copyCrop(image, x: x1, y: y1, width: x2 - x1, height: y2 - y1);
    final blurred = img.gaussianBlur(roi, radius: k ~/ 2);
    img.compositeImage(result, blurred, dstX: x1, dstY: y1);
    
    return result;
  }

  /// 从 Uint8List 解码图像
  static img.Image? decode(Uint8List bytes) {
    return img.decodeImage(bytes);
  }

  /// 编码为 JPEG
  static Uint8List encodeJpg(img.Image image, {int quality = 95}) {
    return Uint8List.fromList(img.encodeJpg(image, quality: quality));
  }
}
