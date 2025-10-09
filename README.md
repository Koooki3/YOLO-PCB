# VisualRobot

基于 Qt 和 C++ 开发的工业视觉机器人控制系统，提供完整的相机控制、图像处理、深度学习解决方案和高级视觉分析功能。

## ✨ 主要特性

- 🎥 支持 GigE 和 USB3.0 接口的工业相机控制
- 🖼️ 强大的数字图像处理功能
- 🤖 深度学习图像分类支持
- 🌐 支持中英文双语界面
- 📊 实时系统状态监控（CPU、内存、温度、图像清晰度）
- 🛠️ 可配置的相机参数设置
- 📈 特征检测和匹配功能
- 🔍 Halcon 图像处理集成
- 📷 图像去畸变处理
- 🎯 交互式多边形绘制和图像裁剪
- 📏 精确的尺寸测量和角度计算
- 🔄 实时图像清晰度分析
- 📐 坐标变换和矩阵计算
- 🖱️ 鼠标交互式图像操作

## 📁 项目结构

```plaintext
VisualRobot/
├── Data/                          # 数据文件目录
│   ├── Datasets/                  # 训练数据集
│   │   ├── cats/                  # 猫类图像
│   │   └── dogs/                  # 狗类图像
│   ├── Labels/                    # 标签文件
│   │   └── class_labels.txt       # 类别标签文件
│   └── Models/                    # 模型文件
│       ├── resnet34_cat_dog_classifier.onnx  # ONNX 格式的猫狗分类模型
│       └── README_MODELS.md       # 模型说明文档
├── Doc/                           # 项目文档
│   ├── 调试信息手册.md             # 系统调试和维护手册 (Markdown)
│   ├── 调试信息手册.pdf            # 系统调试和维护手册 (PDF)
│   └── 深度学习二分类使用指南.md    # 深度学习使用指南
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
├── VisualRobot/                   # 核心源代码目录
│   ├── DataProcessor.cpp          # 数据处理实现
│   ├── DataProcessor.h            # 数据处理头文件
│   ├── DIP.cpp                    # 数字图像处理算法实现
│   ├── DIP.h                      # 数字图像处理头文件
│   ├── DLExample.cpp              # 深度学习示例实现
│   ├── DLExample.h                # 深度学习示例头文件
│   ├── DLProcessor.cpp            # 深度学习处理实现
│   ├── DLProcessor.h              # 深度学习处理头文件
│   ├── featureDetect.cpp          # 特征检测实现
│   ├── featureDetect.h            # 特征检测头文件
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
├── detectedImg.jpg                # 最新检测结果图像
├── feature_matches.jpg            # 特征匹配结果图像
├── matrix.bin                     # 系统配置矩阵文件
├── test.jpg                       # 测试图像
├── README.md                      # 项目说明文档
└── log/                           # 系统日志目录
    └── log_20250930_153826.txt    # 系统运行日志文件
```

### 目录说明

#### 📂 核心源代码 (VisualRobot/)

- **相机控制** - `MvCamera.*` 工业相机控制，支持 GigE/USB3.0
- **图像处理** - `DIP.*` 数字图像处理，包含图像增强和特征检测
- **深度学习** - `DLProcessor.*`, `DLExample.*` 深度学习图像分类处理
- **特征检测** - `featureDetect.*` 图像特征检测和匹配
- **系统监控** - `SystemMonitor.*` 系统监控，追踪性能和资源使用
- **数据处理** - `DataProcessor.*` 通用数据处理功能
- **图像去畸变** - `Undistort.*` 相机畸变校正处理
- **用户界面** - `mainwindow.*` 主窗口实现，包含界面逻辑
- **系统统计** - `mainwindow_systemstats.cpp` 系统统计界面实现

#### 📂 Halcon 代码 (HalconCode/)

- **Halcon 集成** - `Halcon.*` Halcon 图像处理功能集成

#### 📂 数据文件 (Data/)

- **Datasets** - 训练数据集，包含猫狗分类样本
- **Labels** - 标签文件，定义类别名称
- **Models** - 预训练模型文件，包含 ResNet34 猫狗分类器

#### 📂 文档 (Doc/)

- 系统调试和维护手册
- 深度学习使用指南
- 相关技术文档

#### 📂 图像资源 (Img/)

- 测试图像文件
- 示例图像
- 处理结果图像

### 关键文件说明

- `VisualRobot.pro` - Qt 项目配置，包含构建设置、依赖配置、资源管理和编译选项
- `main.cpp` - 程序入口点，负责初始化应用程序、配置日志系统和启动主界面
- `matrix.bin` - 系统配置矩阵文件，存储相机和算法参数
- `resnet34_cat_dog_classifier.onnx` - 预训练的深度学习模型，用于猫狗图像分类
- `Halcon.cpp/.h` - Halcon 图像处理功能集成
- `Undistort.cpp/.h` - 相机畸变校正处理模块

## 💡 核心模块

### 相机控制模块 (MvCamera)
- 工业相机初始化与配置
- 实时图像采集
- 相机参数动态调节

### 图像处理模块 (DIP)
- 图像预处理与增强
- 特征检测与提取
- 目标识别与定位

### 深度学习模块 (DLProcessor)
- ONNX 模型加载和推理
- 图像分类处理
- 结果后处理和分析

### 特征检测模块 (featureDetect)
- 关键点检测
- 特征描述符提取
- 图像匹配和比对

### Halcon 处理模块 (Halcon)
- Halcon 图像算法集成
- 高级图像处理功能
- 工业视觉算法应用

### 去畸变模块 (Undistort)
- 相机畸变校正
- 图像几何变换
- 标定参数应用

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

## 🔧 环境要求

### 开发环境
- Qt 5.x 或更高版本
- C++ 编译器 (支持 C++11 标准)
- CMake 3.x 或更高版本

### 依赖库

1. MVS (Machine Vision Software) SDK
```cpp
INCLUDEPATH += /opt/MVS/include
LIBS += -L/opt/MVS/bin/ -L/opt/MVS/lib/64/
```

2. HALCON 图像处理库
```cpp
INCLUDEPATH += /home/orangepi/MVTec/HALCON-25.05-Progress/include
LIBS += -L/home/orangepi/MVTec/HALCON-25.05-Progress/lib/aarch64-linux/
```

3. Eigen3 数学库
```cpp
INCLUDEPATH += /usr/include/eigen3/Eigen
```

4. OpenCV 计算机视觉库
```cpp
# 用于图像处理、特征检测和清晰度计算
INCLUDEPATH += /usr/include/opencv4
LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
```

5. ONNX Runtime (深度学习推理)
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

1. 使用 Qt Creator 打开 `VisualRobot/VisualRobot.pro`
2. 配置编译选项和依赖库路径
3. 编译项目
4. 运行程序

## ⚠️ 注意事项

1. 首次运行前确认：
   - 所有依赖库已正确安装（MVS、Halcon、OpenCV、Eigen3、ONNX Runtime）
   - 相机驱动已安装
   - 相机连接正常
   - ONNX Runtime 库已配置
   - 确保有足够的磁盘空间存储图像和处理结果

2. 运行时注意：
   - 检查相机连接状态
   - 确保权限设置正确
   - 监控系统资源占用（特别是CPU和内存）
   - 确认模型文件路径正确
   - 多边形绘制功能需要先加载图像到widgetDisplay_2
   - 图像裁剪功能需要至少3个点形成多边形

3. 交互式功能使用说明：
   - **多边形绘制**：在widgetDisplay_2上点击添加顶点，按Enter键完成绘制
   - **图像裁剪**：多边形绘制完成后自动裁剪并显示结果
   - **尺寸测量**：支持标准模式和裁剪区域模式两种测量方式
   - **清晰度分析**：实时显示在状态栏，无需额外操作

4. 调试参考：
   - 查看 `Doc/调试信息手册.md` 或 `Doc/调试信息手册.pdf`
   - 使用系统监控工具跟踪性能
   - 参考 `Doc/深度学习二分类使用指南.md`
   - 查看主界面日志区域获取详细操作信息

## 📝 开发规范

1. 代码风格
   - 使用统一的代码格式化工具
   - 遵循 Qt 编码规范
   - 保持注释的完整性

2. 版本控制
   - 使用语义化版本号
   - 提交信息要清晰明确
   - 保持分支管理的规范性

## 📄 许可证

待添加许可证信息

## 👥 维护者

待添加维护者信息

## 🤝 贡献

欢迎提交问题和改进建议！

---

最后更新日期：2025年10月9日
