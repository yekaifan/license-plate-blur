import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

/// ONNX 推理服务 — 加载和管理车牌/人脸检测模型
class OnnxService {
  OrtSession? _plateSession;
  OrtSession? _faceSession;
  bool _initialized = false;

  bool get isInitialized => _initialized;

  /// 从 assets 加载模型
  Future<void> initialize() async {
    if (_initialized) return;

    // 车牌检测模型
    final plateBytes = await rootBundle.load('assets/models/plate_detector.onnx');
    _plateSession = OrtSession.fromBuffer(
      plateBytes.buffer.asUint8List(),
    );

    // 人脸检测模型
    final faceBytes = await rootBundle.load(
      'assets/models/face_detection_yunet_2023mar.onnx',
    );
    _faceSession = OrtSession.fromBuffer(
      faceBytes.buffer.asUint8List(),
    );

    _initialized = true;
  }

  /// 车牌检测 (YOLOv8n)
  /// 返回: List of [x1, y1, x2, y2, confidence]
  Future<List<PlateBox>> detectPlates(Float32List input, int imgW, int imgH, {double threshold = 0.3}) async {
    if (_plateSession == null) return [];

    final shape = [1, 3, 640, 640];
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, shape);
    final outputs = _plateSession!.run({'images': inputTensor});

    final output = outputs['output0']?.value as List<List<double>>?;
    inputTensor.release();
    for (final o in outputs.values) {
      o.release();
    }

    if (output == null || output.isEmpty || output[0].isEmpty) return [];

    return _parseYoloOutput(output[0], imgW, imgH, threshold);
  }

  /// 人脸检测 (YuNet)
  Future<List<FaceBox>> detectFaces(Float32List input, int imgW, int imgH) async {
    if (_faceSession == null) return [];

    final shape = [1, 3, imgH, imgW];
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, shape);
    final outputs = _faceSession!.run({'input': inputTensor});

    // YuNet outputs: 'output' with shape [n, 15]
    final outputData = outputs.values.first.value;
    inputTensor.release();
    for (final o in outputs.values) {
      o.release();
    }

    if (outputData == null) return [];

    // Parse face boxes from flat list
    final faces = <FaceBox>[];
    final data = outputData is List ? outputData : (outputData as List<List<double>>);
    
    if (data.isEmpty) return faces;
    
    // YuNet output is [n, 15] where [0:4]=bbox, [4]=score, [5:15]=landmarks
    final row = data is List<List<double>> ? data[0] : data;
    if (row is List<double>) {
      // Single face or flattened
      final score = row.length > 4 ? row[4] : 0.0;
      if (score > 0.5) {
        faces.add(FaceBox(
          x: (row[0] as double) * imgW,
          y: (row[1] as double) * imgH,
          w: (row[2] as double) * imgW,
          h: (row[3] as double) * imgH,
          score: score,
        ));
      }
    }

    return faces;
  }

  /// 解析 YOLOv8 输出 — 将 [1, 5, 8400] 转为检测框列表
  List<PlateBox> _parseYoloOutput(List<double> flat, int imgW, int imgH, double threshold) {
    const numClasses = 1; // 只检测车牌
    const numBoxes = 8400;
    
    final boxes = <PlateBox>[];
    final scaleX = imgW / 640.0;
    final scaleY = imgH / 640.0;
    
    for (int i = 0; i < numBoxes; i++) {
      final offset = i * (4 + numClasses);
      if (offset + 4 >= flat.length) break;
      
      final conf = flat[offset + 4];
      if (conf < threshold) continue;
      
      final cx = flat[offset] / 640.0 * imgW;
      final cy = flat[offset + 1] / 640.0 * imgH;
      final w = flat[offset + 2] / 640.0 * imgW;
      final h = flat[offset + 3] / 640.0 * imgH;
      
      final x = cx - w / 2;
      final y = cy - h / 2;
      
      boxes.add(PlateBox(
        x: x.clamp(0, imgW.toDouble()),
        y: y.clamp(0, imgH.toDouble()),
        w: (x + w).clamp(0, imgW.toDouble()) - x,
        h: (y + h).clamp(0, imgH.toDouble()) - y,
        confidence: conf,
      ));
    }
    
    return boxes;
  }

  void dispose() {
    _plateSession?.release();
    _faceSession?.release();
    _initialized = false;
  }
}

class PlateBox {
  final double x, y, w, h, confidence;
  PlateBox({required this.x, required this.y, required this.w, required this.h, required this.confidence});
}

class FaceBox {
  final double x, y, w, h, score;
  FaceBox({required this.x, required this.y, required this.w, required this.h, required this.score});
}
