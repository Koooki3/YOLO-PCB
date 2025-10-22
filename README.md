# VisualRobot

基于 Qt 和 C++ 开发的工业视觉机器人控制系统，提供完整的相机控制、图像处理、深度学习解决方案和高级视觉分析功能。

## ✨ 主要特性

- 🎥 **工业相机控制** - 支持 GigE 和 USB3.0 接口的工业相机，实时图像采集和参数调节
- 🖼️ **数字图像处理** - 强大的图像预处理、增强、特征检测和匹配功能
- 🤖 **深度学习集成** - 支持 ONNX、TensorFlow、Caffe、Darknet 等多种模型格式的二分类任务
- 🌐 **多语言界面** - 完整的中英文双语界面支持
- 📊 **实时系统监控** - CPU、内存、温度监控，图像清晰度实时计算
- 🛠️ **可配置参数** - 灵活的相机参数和算法参数配置
- 📈 **特征检测优化** - 并行化特征检测和匹配，支持批量处理
- 🔍 **Halcon 集成** - 专业的 Halcon 图像处理算法集成
- 📷 **相机标定与去畸变** - 完整的相机标定流程和图像畸变校正
- 🎯 **交互式图像处理** - 多边形绘制、图像裁剪、尺寸测量等交互功能
- 📏 **精确测量分析** - 物件长度、宽度、倾角等精确测量计算
- 🔄 **实时缺陷检测** - 基于模板匹配和特征对齐的实时缺陷检测
- 📐 **坐标变换系统** - 世界坐标与像素坐标的精确转换
- 🖱️ **鼠标交互操作** - 丰富的鼠标事件处理和键盘快捷键支持

## 📁 项目结构

```
VisualRobot/
├── Data/                          # 数据文件目录
│   ├── Labels/                    # 标签文件
│   │   └── class_labels.txt       # 类别标签文件
│   └── Models/                    # 模型文件
│       ├── resnet34_cat_dog_classifier.onnx  # ONNX 格式的猫狗分类模型
│       └── README_MODELS.md       # 模型说明文档
├── Doc/                           # 项目文档
│   ├── 调试信息手册.md             # 系统调试和维护手册 (Markdown)
│   ├── 调试信息手册.pdf            # 系统调试和维护手册 (PDF)
│   ├── 深度学习二分类使用指南.md    # 深度学习使用指南
│   ├── feature_alignment_integration_guide.md  # 特征对齐集成指南
│   └── feature_detect_optimization_analysis.md # 特征检测优化分析
├── HalconCode/                    # Halcon 图像处理代码
│   ├── Halcon.cpp                 # Halcon 功能实现
│   └── Halcon.h                   # Halcon 功能头文件
├── Img/                           # 图像资源目录
│   ├── capture.jpg                # 相机采集的原始图像
│   ├── cat.jpg                    # 猫示例图像
│   ├── circle_detected.jpg        # 圆形检测的结果图像
│   ├── cropped_polygon.jpg        # 多边形裁剪结果图像
│   ├── dog.jpg                    # 狗示例图像
│   ├── test1.jpg                  # 测试图像1
│   ├── test2.jpg                  # 测试图像2
│   ├── test3.jpg                  # 测试图像3
│   ├── test4.jpg                  # 测试图像4
│   ├── undistorted_processed.jpg  # 去畸变处理后的图像
│   └── undistorted.jpg            # 去畸变原始图像
├── ImgData/                       # 图像数据目录（标定图像等）
├── Test/                          # 测试代码目录
│   ├── feature_alignment_test.cpp # 特征对齐测试
│   ├── feature_detect_benchmark.cpp # 特征检测性能测试
│   ├── neu_pipeline.cpp          # NEU 缺陷检测流水线
│   ├── neu_visual.py             # NEU 可视化工具
│   ├── README_NEU.md             # NEU 测试说明
│   └── TestBench.cpp             # 测试基准
├── VisualRobot/                   # 核心源代码目录
│   ├── DataProcessor.cpp          # 数据处理实现（图像增强、特征提取等）
│   ├── DataProcessor.h            # 数据处理头文件
│   ├── DefectDetection.cpp        # 缺陷检测实现（PCA+SVM）
│   ├── DefectDetection.h          # 缺陷检测头文件
│   ├── DIP.cpp                    # 数字图像处理算法实现
│   ├── DIP.h                      # 数字图像处理头文件
│   ├── DLExample.cpp              # 深度学习示例实现
│   ├── DLExample.h                # 深度学习示例头文件
│   ├── DLProcessor.cpp            # 深度学习处理实现
│   ├── DLProcessor.h              # 深度学习处理头文件
│   ├── FeatureAlignment.cpp       # 特征对齐实现
│   ├── FeatureAlignment.h         # 特征对齐头文件
│   ├── featureDetect.cpp          # 特征检测实现
│   ├── featureDetect.h            # 特征检测头文件
│   ├── featureDetect_optimized.cpp # 优化特征检测实现（并行化）
│   ├── featureDetect_optimized.h  # 优化特征检测头文件
│   ├── main.cpp                   # 程序入口点
│   ├── mainwindow.cpp             # 主窗口实现
│   ├── mainwindow.h               # 主窗口头文件
│   ├── mainwindow.ui              # 主窗口界面文件
│   ├── mainwindow_systemstats.cpp # 系统统计界面实现
│   ├── MvCamera.cpp               # 工业相机控制实现
│   ├── MvCamera.h                 # 工业相机控制头文件
│   ├── SystemMonitor.cpp          # 系统监控实现
│   ├── SystemMonitor.h            # 系统监控头文件
│   ├── Undistort.cpp              # 图像去畸变处理实现
│   ├── Undistort.h                # 图像去畸变处理头文件
│   ├── VisualRobot_zh_EN.qm       # 编译后的翻译文件
│   ├── VisualRobot_zh_EN.ts       # 中英文翻译源文件
│   ├── VisualRobot.pro            # Qt 项目配置文件
│   └── VisualRobot.pro.user       # Qt 项目用户配置
├── calibration_parameters.yml     # 相机标定参数文件
├── detectedImg.jpg                # 最新检测结果图像
├── feature_matches.jpg            # 特征匹配结果图像
├── matrix.bin                     # 系统配置矩阵文件
├── test.jpg                       # 测试图像
├── README.md                      # 项目说明文档
└── log/                           # 系统日志目录
    └── log_*.txt                  # 系统运行日志文件
```

### 目录说明

#### 📂 核心源代码 (VisualRobot/)

- **相机控制** - `MvCamera.*` 工业相机控制，支持 GigE/USB3.0
- **图像处理** - `DIP.*` 数字图像处理，包含图像增强和特征检测
- **深度学习** - `DLProcessor.*`, `DLExample.*` 深度学习图像分类处理
- **特征检测** - `featureDetect.*`, `featureDetect_optimized.*` 图像特征检测和匹配（支持并行化）
- **缺陷检测** - `DefectDetection.*` 基于PCA+SVM的缺陷检测系统
- **特征对齐** - `FeatureAlignment.*` 图像特征对齐和配准
- **系统监控** - `SystemMonitor.*` 系统监控，追踪性能和资源使用
- **数据处理** - `DataProcessor.*` 通用数据处理功能（图像增强、特征提取等）
- **图像去畸变** - `Undistort.*` 相机畸变校正处理
- **用户界面** - `mainwindow.*` 主窗口实现，包含界面逻辑
- **系统统计** - `mainwindow_systemstats.cpp` 系统统计界面实现

#### 📂 Halcon 代码 (HalconCode/)

- **Halcon 集成** - `Halcon.*` Halcon 图像处理功能集成

#### 📂 数据文件 (Data/)

- **Labels** - 标签文件，定义类别名称
- **Models** - 预训练模型文件，包含 ResNet34 猫狗分类器

#### 📂 文档 (Doc/)

- 系统调试和维护手册
- 深度学习使用指南
- 特征对齐集成指南
- 特征检测优化分析

#### 📂 测试代码 (Test/)

- **NEU 缺陷检测流水线** - `neu_pipeline.cpp` 支持训练和评估模式
- **特征对齐测试** - `feature_alignment_test.cpp`
- **性能基准测试** - `TestBench.cpp`, `feature_detect_benchmark.cpp`
- **可视化工具** - `neu_visual.py`

#### 📂 图像资源 (Img/)

- 测试图像文件
- 示例图像
- 处理结果图像

### 关键文件说明

- `VisualRobot.pro` - Qt 项目配置，包含构建设置、依赖配置、资源管理和编译选项
- `main.cpp` - 程序入口点，负责初始化应用程序、配置日志系统和启动主界面
- `matrix.bin` - 系统配置矩阵文件，存储相机和算法参数
- `calibration_parameters.yml` - 相机标定参数文件
- `resnet34_cat_dog_classifier.onnx` - 预训练的深度学习模型，用于猫狗图像分类
- `Halcon.cpp/.h` - Halcon 图像处理功能集成
- `Undistort.cpp/.h` - 相机畸变校正处理模块
- `DefectDetection.cpp/.h` - 缺陷检测系统，支持PCA降维和SVM分类

## 💡 核心模块

### 相机控制模块 (MvCamera)
- 工业相机初始化与配置
- 实时图像采集
- 相机参数动态调节（曝光、增益、帧率等）
- 触发模式支持（连续模式、触发模式、软件触发）

### 图像处理模块 (DIP)
- 图像预处理与增强
- 特征检测与提取
- 目标识别与定位
- 尺寸测量和角度计算

### 深度学习模块 (DLProcessor)
- ONNX 模型加载和推理
- 多格式模型支持（TensorFlow、Caffe、Darknet）
- 图像分类处理
- 批量处理支持
- 结果后处理和分析

### 特征检测模块 (featureDetect)
- 关键点检测（ORB、SIFT等）
- 特征描述符提取
- 图像匹配和比对
- 并行化优化版本支持

### 缺陷检测模块 (DefectDetection)
- 基于PCA的特征降维
- SVM分类器训练和预测
- 模板匹配和特征对齐
- 实时缺陷检测

### 特征对齐模块 (FeatureAlignment)
- 图像配准和对齐
- 特征点匹配和几何验证
- 重投影误差计算
- 快速对齐算法

### Halcon 处理模块 (Halcon)
- Halcon 图像算法集成
- 高级图像处理功能
- 工业视觉算法应用

### 去畸变模块 (Undistort)
- 相机畸变校正
- 图像几何变换
- 标定参数应用
- 自动标定流程

### 系统监控模块 (SystemMonitor)
- 实时性能监控（CPU、内存、温度）
- 系统资源管理
- 运行状态追踪
- 图像清晰度实时计算和显示

### 交互式图像处理模块 (MainWindow)
- 直观的用户交互界面
- 相机实时预览
- 多语言支持
- 处理结果显示
- **交互式多边形绘制**：支持鼠标点击绘制多边形区域
- **智能图像裁剪**：自动裁剪多边形区域并补全背景
- **实时尺寸测量**：精确计算物件长度、宽度和倾角
- **坐标变换验证**：世界坐标与像素坐标的精确转换
- **清晰度分析**：Tenengrad算法实时计算图像清晰度
- **鼠标事件处理**：支持点击、键盘交互操作
- **实时缺陷检测**：基于模板的实时缺陷检测流程

## 🔧 环境要求

### 开发环境
- Qt 5.x 或更高版本
- C++ 编译器 (支持 C++11 标准)
- CMake 3.x 或更高版本

### 依赖库

1. **MVS (Machine Vision Software) SDK**
```cpp
INCLUDEPATH += /opt/MVS/include
LIBS += -L/opt/MVS/bin/ -L/opt/MVS/lib/64/
```

2. **HALCON 图像处理库**
```cpp
INCLUDEPATH += /home/orangepi/MVTec/HALCON-25.05-Progress/include
LIBS += -L/home/orangepi/MVTec/HALCON-25.05-Progress/lib/aarch64-linux/
```

3. **Eigen3 数学库**
```cpp
INCLUDEPATH += /usr/include/eigen3/Eigen
```

4. **OpenCV 计算机视觉库**
```cpp
# 用于图像处理、特征检测和清晰度计算
INCLUDEPATH += /usr/include/opencv4
LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_calib3d -lopencv_ml -lopencv_dnn
```

5. **ONNX Runtime (深度学习推理)**
```cpp
# 需要安装 ONNX Runtime 库用于模型推理
```

## 📦 安装与配置

### 1. 安装依赖
```bash
# 安装 Qt 开发环境
# Windows: 使用 Qt 在线安装器
# Linux: 
sudo apt-get install qt5-default
sudo apt-get install qtcreator

# 安装必要的开发工具
sudo apt-get install build-essential
sudo apt-get install cmake

# 安装 ONNX Runtime (根据平台选择)
# 参考: https://onnxruntime.ai/
```

### 2. 配置项目
```bash
# 克隆项目
git clone https://jihulab.com/visualteam/VisualRobot.git
cd VisualRobot
```

## 🚀 编译与运行

### 使用 Qt Creator
1. 使用 Qt Creator 打开 `VisualRobot/VisualRobot.pro`
2. 配置编译选项和依赖库路径
3. 编译项目
4. 运行程序

### 命令行编译
```bash
# 生成构建文件
qmake VisualRobot/VisualRobot.pro
# 编译
make
# 运行
./VisualRobot
```

## 🧪 测试功能

### NEU 缺陷检测流水线
```bash
# 训练模式
./neu_pipeline -m train -r ../Data/NEU -p 32

# 评估模式
./neu_pipeline -m eval -r ../Data/NEU -o neu_results.csv
```

### 特征检测性能测试
```bash
# 运行特征检测基准测试
./feature_detect_benchmark
```

### 特征对齐测试
```bash
# 运行特征对齐测试
./feature_alignment_test
```

## ⚠️ 注意事项

### 首次运行前确认
- 所有依赖库已正确安装（MVS、Halcon、OpenCV、Eigen3、ONNX Runtime）
- 相机驱动已安装
- 相机连接正常
- ONNX Runtime 库已配置
- 确保有足够的磁盘空间存储图像和处理结果

### 运行时注意
- 检查相机连接状态
- 确保权限设置正确
- 监控系统资源占用（特别是CPU和内存）
- 确认模型文件路径正确
- 多边形绘制功能需要先加载图像到widgetDisplay_2
- 图像裁剪功能需要至少3个点形成多边形

### 交互式功能使用说明
- **多边形绘制**：在widgetDisplay_2上点击添加顶点，按Enter键完成绘制
- **图像裁剪**：多边形绘制完成后自动裁剪并显示结果
- **尺寸测量**：支持标准模式和裁剪区域模式两种测量方式
- **清晰度分析**：实时显示在状态栏，无需额外操作
- **实时缺陷检测**：需要先设置模板图像，然后启动实时检测模式

### 调试参考
- 查看 `Doc/调试信息手册.md` 或 `Doc/调试信息手册.pdf`
- 使用系统监控工具跟踪性能
- 参考 `Doc/深度学习二分类使用指南.md`
- 查看主界面日志区域获取详细操作信息
- 使用测试代码验证各模块功能

## 📝 开发规范

### 代码风格
- 使用统一的代码格式化工具
- 遵循 Qt 编码规范
- 保持注释的完整性
- 使用有意义的变量和函数命名

### 版本控制
- 使用语义化版本号
- 提交信息要清晰明确
- 保持分支管理的规范性
- 定期同步远程仓库

### 测试要求
- 新功能需要提供相应的测试代码
- 性能关键模块需要基准测试
- 确保向后兼容性
- 文档与代码同步更新

## 🔄 更新日志

### v1.0.0 (2025-10-22)
- 完整的工业视觉系统框架
- 支持多种相机接口和图像处理算法
- 深度学习模型集成
- 实时系统监控
- 交互式图像处理功能
- 完整的文档和测试套件

## 📄 许可证

待添加许可证信息

## 👥 维护者

待添加维护者信息

## 🤝 贡献

欢迎提交问题和改进建议！

### 贡献流程
1. Fork 项目仓库
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

### 报告问题
- 使用清晰的问题描述
- 提供复现步骤
- 包含相关日志和截图
- 说明期望的行为和实际行为

---

**最后更新日期：2025年10月22日**

**项目仓库：https://jihulab.com/visualteam/VisualRobot.git**
