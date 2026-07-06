import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

/// ONNX 推理服务
class OnnxService {
  OrtSession? _plateSession;
  OrtSession? _faceSession;
  bool _initialized = false;

  bool get isInitialized => _initialized;

  Future<void> initialize() async {
    if (_initialized) return;

    final plateBytes = await rootBundle.load('assets/models/plate_detector.onnx');
    _plateSession = OrtSession.fromBuffer(
      plateBytes.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    final faceBytes = await rootBundle.load(
      'assets/models/face_detection_yunet_2023mar.onnx',
    );
    _faceSession = OrtSession.fromBuffer(
      faceBytes.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    _initialized = true;
  }

  /// 车牌检测 (YOLOv8n)
  Future<List<PlateBox>> detectPlates(
    Float32List input, int imgW, int imgH, {double threshold = 0.3}) async {
    if (_plateSession == null) return [];

    final shape = [1, 3, 640, 640];
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, shape);

    final runOptions = OrtRunOptions();
    runOptions.addInput('images', inputTensor);
    runOptions.addOutput('output0');

    final outputs = _plateSession!.run(runOptions);
    inputTensor.release();
    runOptions.release();

    if (outputs.isEmpty) return [];

    final raw = outputs[0]?.tensorData as List<double>?;
    for (final o in outputs) {
      o?.release();
    }

    if (raw == null) return [];
    return _parseYoloOutput(raw, imgW, imgH, threshold);
  }

  /// 人脸检测 (YuNet)
  Future<List<FaceBox>> detectFaces(Float32List input, int imgW, int imgH) async {
    if (_faceSession == null) return [];

    final shape = [1, 3, imgH, imgW];
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, shape);

    final runOptions = OrtRunOptions();
    runOptions.addInput('input', inputTensor);
    runOptions.addOutput('output');

    final outputs = _faceSession!.run(runOptions);
    inputTensor.release();
    runOptions.release();

    if (outputs.isEmpty) return [];

    final data = outputs[0]?.tensorData;
    for (final o in outputs) {
      o?.release();
    }

    if (data == null) return [];

    // YuNet: [n, 15] → 找 score > 0.5 的人脸
    final faces = <FaceBox>[];
    final list = data is List<double>
        ? data
        : (data as List<num>).map((e) => e.toDouble()).toList();

    const step = 15;
    for (int i = 0; i + step <= list.length; i += step) {
      final score = list[i + 4];
      if (score > 0.5) {
        faces.add(FaceBox(
          x: list[i] * imgW,
          y: list[i + 1] * imgH,
          w: list[i + 2] * imgW,
          h: list[i + 3] * imgH,
          score: score,
        ));
      }
    }
    return faces;
  }

  List<PlateBox> _parseYoloOutput(List<double> flat, int imgW, int imgH, double threshold) {
    const numBoxes = 8400;
    final boxes = <PlateBox>[];

    for (int i = 0; i < numBoxes; i++) {
      final offset = i * 5; // [cx, cy, w, h, conf]
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
