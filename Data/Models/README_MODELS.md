# 深度学习模型说明文档

## 概述

本文档描述了VisualRobot项目中使用的深度学习模型配置和使用方法。

## 支持的模型格式

DLProcessor支持以下模型格式：

### 1. ONNX模型 (.onnx)
- **推荐格式**，跨平台兼容性好
- 支持从PyTorch、TensorFlow等框架导出
- 使用方法：`initModel("model.onnx")`

### 2. TensorFlow模型
- `.pb` 文件（frozen graph）
- 使用方法：`initModel("model.pb")`

### 3. Caffe模型
- 需要 `.caffemodel` 和 `.prototxt` 文件
- 使用方法：`initModel("model.caffemodel", "config.prototxt")`

### 4. Darknet模型
- 需要 `.weights` 和 `.cfg` 文件
- 使用方法：`initModel("model.weights", "config.cfg")`

## 二分类模型要求

### 输入要求
- **图像尺寸**：默认224x224，可通过`setInputSize()`调整
- **颜色通道**：支持RGB和灰度图像
- **数据类型**：CV_32F（浮点型）

### 输出格式
支持两种输出格式：

#### 1. Sigmoid输出（单节点）
```
输出形状: [1, 1]
输出值: 0.0 ~ 1.0
阈值: > 0.5 为类别1，<= 0.5 为类别0
```

#### 2. Softmax输出（双节点）
```
输出形状: [1, 2]
输出值: [prob_class0, prob_class1]
选择: 概率最大的类别
```

## 模型配置示例

### 1. 基础配置
```cpp
DLProcessor processor;

// 加载模型
processor.initModel("path/to/model.onnx");

// 设置参数
processor.setModelParams(0.7f, 0.4f);  // 置信度阈值, NMS阈值

// 设置输入尺寸
processor.setInputSize(cv::Size(224, 224));

// 加载类别标签
processor.loadClassLabels("Data/Labels/class_labels.txt");
```

### 2. 自定义预处理参数
```cpp
// 对于不同的预训练模型，可能需要不同的预处理参数

// ImageNet预训练模型
meanValues = cv::Scalar(104.0, 177.0, 123.0);
scaleFactor = 1.0;

// 归一化到[0,1]
meanValues = cv::Scalar(0, 0, 0);
scaleFactor = 1.0/255.0;

// 标准化到[-1,1]
meanValues = cv::Scalar(127.5, 127.5, 127.5);
scaleFactor = 1.0/127.5;
```

## 使用示例

### 1. 单张图像分类
```cpp
cv::Mat image = cv::imread("test_image.jpg");
ClassificationResult result;

if (processor.classifyImage(image, result)) {
    std::cout << "类别: " << result.className << std::endl;
    std::cout << "置信度: " << result.confidence << std::endl;
    std::cout << "类别ID: " << result.classId << std::endl;
}
```

### 2. 批量图像分类
```cpp
std::vector<cv::Mat> images;
// ... 加载多张图像到images向量

std::vector<ClassificationResult> results;
if (processor.classifyBatch(images, results)) {
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "图像 " << i << ": " 
                  << results[i].className << " (" 
                  << results[i].confidence << ")" << std::endl;
    }
}
```

### 3. 与Qt信号槽集成
```cpp
// 连接信号
connect(&processor, &DLProcessor::classificationComplete,
        this, &MainWindow::onClassificationComplete);

connect(&processor, &DLProcessor::errorOccurred,
        this, &MainWindow::onDLError);

// 槽函数
void MainWindow::onClassificationComplete(const ClassificationResult& result) {
    QString message = QString("分类结果: %1 (置信度: %2)")
                     .arg(QString::fromStdString(result.className))
                     .arg(result.confidence);
    appendLog(message, INFO);
}
```

## 模型训练建议

### 1. 数据准备
- **数据平衡**：确保两个类别的样本数量相对平衡
- **数据增强**：使用旋转、缩放、翻转等增强技术
- **质量控制**：确保标注准确，图像质量良好

### 2. 模型选择
推荐的二分类模型架构：
- **轻量级**：MobileNetV2, EfficientNet-B0
- **高精度**：ResNet50, DenseNet121
- **实时性**：SqueezeNet, ShuffleNet

### 3. 训练参数
```python
# PyTorch示例
model = torchvision.models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 二分类

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### 4. 模型导出
```python
# 导出为ONNX格式
torch.onnx.export(model, dummy_input, "model.onnx", 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                               'output': {0: 'batch_size'}})
```

## 性能优化

### 1. 硬件加速
```cpp
// CPU优化
net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

// GPU加速（需要CUDA支持）
net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

// Intel推理引擎（需要OpenVINO）
net_.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
```

### 2. 批处理优化
- 对于多张图像，使用`classifyBatch()`比多次调用`classifyImage()`更高效
- 批处理可以更好地利用GPU并行计算能力

### 3. 内存优化
- 及时释放不需要的cv::Mat对象
- 对于大批量处理，考虑分批处理避免内存溢出

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型格式与OpenCV DNN模块兼容
   - 查看错误日志获取详细信息

2. **推理结果异常**
   - 检查输入图像预处理是否正确
   - 确认模型输出格式与后处理逻辑匹配
   - 验证类别标签文件是否正确

3. **性能问题**
   - 尝试不同的后端和目标设备
   - 检查输入图像尺寸是否过大
   - 考虑使用更轻量级的模型

### 调试技巧

1. **启用详细日志**
```cpp
// 在Qt应用中查看qDebug输出
qDebug() << "Model loaded:" << processor.isModelLoaded();
qDebug() << "Input size:" << processor.getInputSize().width << "x" << processor.getInputSize().height;
```

2. **验证模型输出**
```cpp
// 检查网络输出层名称
auto layerNames = net_.getUnconnectedOutLayersNames();
for (const auto& name : layerNames) {
    qDebug() << "Output layer:" << QString::fromStdString(name);
}
```

3. **测试简单样本**
- 使用已知结果的简单图像进行测试
- 逐步增加复杂度，定位问题所在

## 更新日志

- **v1.0.0** (2025-01-09): 初始版本，支持基础二分类功能
- 后续版本将添加更多模型格式支持和优化功能

## 联系方式

如有问题或建议，请通过项目仓库提交Issue或联系开发团队。
