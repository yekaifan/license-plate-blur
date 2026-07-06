import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/onnx_service.dart';
import '../services/image_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final OnnxService _onnx = OnnxService();
  final ImagePicker _picker = ImagePicker();
  
  final List<_ImageItem> _items = [];
  bool _loading = false;
  bool _modelReady = false;
  String _status = '正在加载模型...';
  
  // 设置
  double _threshold = 0.3;
  int _margin = 10;
  int _blurKernel = 45;
  bool _blurPlates = true;
  bool _blurFaces = true;
  String? _outputDir;
  
  final List<String> _logs = [];

  @override
  void initState() {
    super.initState();
    _initModels();
  }

  Future<void> _initModels() async {
    try {
      await _onnx.initialize();
      setState(() {
        _modelReady = true;
        _status = '✅ 就绪 — 默认同时打码车牌和人脸';
      });
      _addLog('✅ 模型加载成功');
    } catch (e) {
      setState(() => _status = '❌ 模型加载失败: $e');
      _addLog('❌ $e');
    }
  }

  // ── 文件选择 ─────────────────────────────────────

  Future<void> _pickFiles() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: true,
    );
    if (result == null) return;
    
    for (final file in result.files) {
      if (file.path != null) {
        _items.add(_ImageItem(path: file.path!, name: file.name));
      }
    }
    setState(() {});
    _addLog('📥 已添加 ${result.files.length} 张图片（共 ${_items.length} 张）');
  }

  Future<void> _takePhoto() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      _addLog('❌ 相机权限未授权');
      return;
    }
    
    final photo = await _picker.pickImage(source: ImageSource.camera);
    if (photo == null) return;
    
    _items.add(_ImageItem(path: photo.path, name: '拍照_${_items.length + 1}.jpg'));
    setState(() {});
    _addLog('📸 拍照完成');
  }

  // ── 处理 ─────────────────────────────────────────

  Future<void> _startBlur() async {
    if (_items.isEmpty) {
      _addLog('⚠️ 请先选择图片');
      return;
    }
    if (!_blurPlates && !_blurFaces) {
      _addLog('⚠️ 请至少勾选一种打码类型');
      return;
    }
    
    setState(() => _loading = true);
    _addLog('—' * 20);
    _addLog('🚀 开始处理 ${_items.length} 张图片');

    int done = 0;
    int detected = 0;

    for (final item in _items) {
      item.status = '🔄 处理中...';
      setState(() {});
      
      try {
        final bytes = await File(item.path).readAsBytes();
        final image = ImageService.decode(bytes);
        if (image == null) {
          item.status = '❌ 读取失败';
          continue;
        }
        
        int plateCount = 0;
        int faceCount = 0;
        var result = image;

        // 车牌检测
        if (_blurPlates) {
          final input = ImageService.preprocessYolo(image);
          final plates = await _onnx.detectPlates(
            input, image.width, image.height, threshold: _threshold,
          );
          
          for (final box in plates) {
            result = ImageService.blurRegion(
              result, box.x.toInt(), box.y.toInt(),
              box.w.toInt(), box.h.toInt(),
              _blurKernel, _margin,
            );
            plateCount++;
          }
        }

        // 人脸检测
        if (_blurFaces) {
          final input = ImageService.preprocessFace(image);
          final faces = await _onnx.detectFaces(
            input, image.width, image.height,
          );
          
          for (final face in faces) {
            result = ImageService.blurRegion(
              result, face.x.toInt(), face.y.toInt(),
              face.w.toInt(), face.h.toInt(),
              _blurKernel, _margin,
            );
            faceCount++;
          }
        }

        // 保存
        final outDir = _outputDir ?? 
            '${(await getApplicationDocumentsDirectory()).path}/blurred';
        Directory(outDir).createSync(recursive: true);
        
        final outPath = '$outDir/${item.name.replaceAll(RegExp(r'\.[^.]+$'), '_blurred')}.jpg';
        final outBytes = ImageService.encodeJpg(result);
        await File(outPath).writeAsBytes(outBytes);
        item.outPath = outPath;

        final total = plateCount + faceCount;
        if (total > 0) {
          final parts = <String>[];
          if (plateCount > 0) parts.add('$plateCount车牌');
          if (faceCount > 0) parts.add('$faceCount人脸');
          item.status = '✅ ${parts.join('+')}';
          detected++;
        } else {
          item.status = '⚠️ 未检测到';
        }
        
        _addLog('  ${item.status} → ${item.name}');
        
      } catch (e) {
        item.status = '❌ $e';
        _addLog('  ❌ ${item.name}: $e');
      }
      
      done++;
      setState(() {});
    }

    setState(() => _loading = false);
    _addLog('🎉 完成！$detected/${_items.length} 张检测到目标');
  }

  // ── 辅助 ─────────────────────────────────────────

  void _clearList() {
    _items.clear();
    _logs.clear();
    setState(() {});
  }

  void _addLog(String msg) {
    _logs.add(msg);
    setState(() {});
  }

  // ── UI ───────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🚗 车牌 & 人脸打码工具'),
        centerTitle: true,
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: Column(
        children: [
          // 状态栏
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            color: _modelReady ? Colors.green.shade50 : Colors.orange.shade50,
            child: Text(_status, textAlign: TextAlign.center),
          ),
          
          // 主内容区
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // 文件操作按钮
                  _buildCard([
                    Row(children: [
                      Expanded(child: _buildBtn('📂 浏览文件', _pickFiles)),
                      const SizedBox(width: 8),
                      Expanded(child: _buildBtn('📷 拍照导入', _takePhoto)),
                    ]),
                  ]),
                  
                  const SizedBox(height: 8),
                  
                  // 打码开关
                  _buildCard([
                    Row(children: [
                      _buildToggle('🚗 车牌', _blurPlates, (v) => setState(() => _blurPlates = v)),
                      const SizedBox(width: 16),
                      _buildToggle('👤 人脸', _blurFaces, (v) => setState(() => _blurFaces = v)),
                    ]),
                  ]),
                  
                  const SizedBox(height: 8),
                  
                  // 设置滑块
                  _buildCard([
                    _buildSlider('🔲 扩边像素', _margin.toDouble(), 0, 200, 1, (v) => _margin = v.toInt()),
                    _buildSlider('🌫️ 模糊强度', _blurKernel.toDouble(), 1, 99, 2, (v) => _blurKernel = v.toInt()),
                    _buildSlider('🎯 置信度阈值', _threshold, 0.1, 0.9, 0.05, (v) => _threshold = v),
                  ]),
                  
                  const SizedBox(height: 8),
                  
                  // 图片列表
                  if (_items.isNotEmpty) ...[
                    Text('📋 图片列表 (${_items.length})',
                      style: Theme.of(context).textTheme.titleSmall),
                    const SizedBox(height: 4),
                    ..._items.map((item) => Card(
                      child: ListTile(
                        dense: true,
                        title: Text(item.name, style: const TextStyle(fontSize: 13)),
                        trailing: Text(item.status, style: TextStyle(
                          fontSize: 12,
                          color: item.status.startsWith('✅') ? Colors.green : Colors.grey,
                        )),
                      ),
                    )),
                  ],
                  
                  const SizedBox(height: 8),
                  
                  // 进度条
                  if (_loading) const LinearProgressIndicator(),
                  
                  const SizedBox(height: 8),
                  
                  // 按钮栏
                  Row(children: [
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: _loading ? null : _startBlur,
                        icon: const Icon(Icons.auto_fix_high),
                        label: const Text('开始打码'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    OutlinedButton.icon(
                      onPressed: _clearList,
                      icon: const Icon(Icons.clear_all),
                      label: const Text('清空'),
                    ),
                  ]),
                  
                  const SizedBox(height: 8),
                  
                  // 日志
                  if (_logs.isNotEmpty) ...[
                    Text('📜 运行日志',
                      style: Theme.of(context).textTheme.titleSmall),
                    const SizedBox(height: 4),
                    Container(
                      height: 120,
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.grey.shade900,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: ListView.builder(
                        itemCount: _logs.length,
                        itemBuilder: (_, i) => Text(
                          _logs[i],
                          style: const TextStyle(
                            color: Colors.lightGreen,
                            fontSize: 11,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCard(List<Widget> children) {
    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(children: children),
      ),
    );
  }

  Widget _buildBtn(String label, VoidCallback onTap) {
    return OutlinedButton(
      onPressed: _modelReady ? onTap : null,
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(0, 48),
      ),
      child: Text(label),
    );
  }

  Widget _buildToggle(String label, bool value, ValueChanged<bool> onChanged) {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Checkbox(value: value, onChanged: (v) => onChanged(v ?? false)),
      Text(label),
    ]);
  }

  Widget _buildSlider(String label, double value, double min, double max, double step, ValueChanged<double> onChanged) {
    return Row(children: [
      SizedBox(width: 100, child: Text(label, style: const TextStyle(fontSize: 13))),
      Expanded(
        child: Slider(
          value: value,
          min: min,
          max: max,
          divisions: ((max - min) / step).round(),
          onChanged: onChanged,
        ),
      ),
      SizedBox(
        width: 44,
        child: Text(value.toStringAsFixed(step < 1 ? 2 : 0), textAlign: TextAlign.right),
      ),
    ]);
  }
}

class _ImageItem {
  final String path;
  final String name;
  String status = '⏳ 等待中';
  String? outPath;
  
  _ImageItem({required this.path, required this.name});
}
